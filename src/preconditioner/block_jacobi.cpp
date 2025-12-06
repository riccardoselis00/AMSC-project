#include <cmath>
#include <limits>

#include "algebra/matrixSparse.hpp" 
#include "algebra/CSR.hpp"            
#include "algebra/COO.hpp"           
#include "preconditioner/block_jacobi.hpp"

#include <typeinfo>
#include <stdexcept>

static void lu_factor_block(std::vector<double>& M, int n)
{
  for (int k = 0; k < n; ++k) {
    double pivot = M[k*n + k];
    if (std::abs(pivot) < 1e-15) {
      throw std::runtime_error("BlockJacobi::lu_factor_block: zero/near-zero pivot");
    }

    for (int i = k + 1; i < n; ++i) {
      M[i*n + k] /= pivot;
      double lik = M[i*n + k];
      for (int j = k + 1; j < n; ++j) {
        M[i*n + j] -= lik * M[k*n + j];
      }
    }
  }
}

static void lu_solve_block(const std::vector<double>& LU,
                           int n,
                           const double* b,
                           double* x)
{

  std::vector<double> y(n);
  for (int i = 0; i < n; ++i) {
    double sum = b[i];
    for (int j = 0; j < i; ++j) {
      sum -= LU[i*n + j] * y[j];
    }
    y[i] = sum;
  }

  for (int i = n - 1; i >= 0; --i) {
    double sum = y[i];
    for (int j = i + 1; j < n; ++j) {
      sum -= LU[i*n + j] * x[j];
    }
    double pivot = LU[i*n + i];
    if (std::abs(pivot) < 1e-15) {
      throw std::runtime_error("BlockJacobi::lu_solve_block: zero/near-zero pivot");
    }
    x[i] = sum / pivot;
  }
}

BlockJacobi::BlockJacobi(int block_size)
  : m_blockSize(block_size),
    m_n(0),
    m_nparts(0)
{
    if (m_blockSize <= 0) {
        throw std::invalid_argument("BlockJacobi: block_size must be > 0");
    }
}

void BlockJacobi::setup(const MatrixSparse& A)
{
    using Index  = MatrixSparse::Index;
    using Scalar = MatrixSparse::Scalar;

    m_n = static_cast<int>(A.rows());
    if (A.cols() != static_cast<std::size_t>(m_n)) {
        throw std::runtime_error("BlockJacobi::setup requires a square matrix.");
    }

    // 1) Define blocks: contiguous rows of size m_blockSize (last may be smaller)
    m_nparts = (m_n + m_blockSize - 1) / m_blockSize; // ceil(n / blockSize)

    m_starts.resize(static_cast<std::size_t>(m_nparts + 1));
    m_blockSizes.resize(static_cast<std::size_t>(m_nparts));

    int pos = 0;
    for (int p = 0; p < m_nparts; ++p) {
        m_starts[static_cast<std::size_t>(p)] = pos;
        int bs = std::min(m_blockSize, m_n - pos);
        m_blockSizes[static_cast<std::size_t>(p)] = bs;
        pos += bs;
    }
    m_starts[static_cast<std::size_t>(m_nparts)] = m_n;

    // 2) Allocate dense blocks and row->block map
    m_LUblocks.clear();
    m_LUblocks.resize(static_cast<std::size_t>(m_nparts));

    int max_bs = 0;
    for (int p = 0; p < m_nparts; ++p) {
        int bs = m_blockSizes[static_cast<std::size_t>(p)];
        max_bs = std::max(max_bs, bs);
        m_LUblocks[static_cast<std::size_t>(p)].assign(
            static_cast<std::size_t>(bs * bs), 0.0);
    }

    m_rowToBlock.resize(static_cast<std::size_t>(m_n));
    for (int p = 0; p < m_nparts; ++p) {
        int s = m_starts[static_cast<std::size_t>(p)];
        int e = m_starts[static_cast<std::size_t>(p + 1)];
        for (int i = s; i < e; ++i) {
            m_rowToBlock[static_cast<std::size_t>(i)] = p;
        }
    }

    // 3) Fill diagonal blocks in ONE pass over A
    A.forEachNZ([&](Index r, Index c, Scalar v) {
        int ri = static_cast<int>(r);
        int ci = static_cast<int>(c);
        int pRow = m_rowToBlock[static_cast<std::size_t>(ri)];
        int pCol = m_rowToBlock[static_cast<std::size_t>(ci)];
        if (pRow != pCol) return;  // off-diagonal block ignored

        const int p  = pRow;
        const int s  = m_starts[static_cast<std::size_t>(p)];
        const int bs = m_blockSizes[static_cast<std::size_t>(p)];

        int lr = ri - s;
        int lc = ci - s;

        m_LUblocks[static_cast<std::size_t>(p)]
            [static_cast<std::size_t>(lr * bs + lc)] = static_cast<double>(v);
    });

    // 4) Factor each block (dense LU on SMALL matrices)
    for (int p = 0; p < m_nparts; ++p) {
        int bs = m_blockSizes[static_cast<std::size_t>(p)];
        if (bs > 0) {
            lu_factor_block(m_LUblocks[static_cast<std::size_t>(p)], bs);
        }
    }

    // 5) Prepare workspaces for apply()
    m_rhs.resize(static_cast<std::size_t>(max_bs));
    m_sol.resize(static_cast<std::size_t>(max_bs));
}


void BlockJacobi::apply(const std::vector<double>& r,
                        std::vector<double>& z) const
{
    if (r.size() != static_cast<std::size_t>(m_n)) {
        throw std::runtime_error("BlockJacobi::apply: r has wrong size");
    }

    z.assign(r.size(), 0.0);

    for (int p = 0; p < m_nparts; ++p) {
        const int s  = m_starts[static_cast<std::size_t>(p)];
        const int bs = m_blockSizes[static_cast<std::size_t>(p)];
        if (bs <= 0) continue;

        const auto& LU = m_LUblocks[static_cast<std::size_t>(p)];

        double* rhs = m_rhs.data();
        double* sol = m_sol.data();

        // rhs = local residual
        for (int i = 0; i < bs; ++i) {
            rhs[i] = r[static_cast<std::size_t>(s + i)];
        }

        lu_solve_block(LU, bs, rhs, sol);

        // write back
        for (int i = 0; i < bs; ++i) {
            z[static_cast<std::size_t>(s + i)] = sol[i];
        }
    }
}
