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

BlockJacobi::BlockJacobi(int nparts)
  : m_nparts(nparts), m_n(0)
{
  if (m_nparts <= 0) {
    throw std::invalid_argument("BlockJacobi: nparts must be > 0");
  }
}

void BlockJacobi::setup(const MatrixSparse& A)
{
  using Index  = MatrixSparse::Index;
  using Scalar = MatrixSparse::Scalar;

  m_n = A.rows();
  if (A.cols() != m_n) {
    throw std::runtime_error("BlockJacobi::setup requires a square matrix.");
  }

  m_starts.resize(static_cast<std::size_t>(m_nparts) + 1);
  const Index base = m_n / static_cast<Index>(m_nparts);
  Index rem = m_n % static_cast<Index>(m_nparts);
  Index pos = 0;
  for (int p = 0; p < m_nparts; ++p) {
    m_starts[static_cast<std::size_t>(p)] = static_cast<int>(pos);
    Index sz = base + (static_cast<Index>(p) < rem ? 1 : 0);
    pos += sz;
  }
  m_starts[static_cast<std::size_t>(m_nparts)] = static_cast<int>(m_n);

  m_blockSizes.resize(static_cast<std::size_t>(m_nparts));
  m_LUblocks.clear();
  m_LUblocks.resize(static_cast<std::size_t>(m_nparts));

  for (int p = 0; p < m_nparts; ++p) {
    const int s  = m_starts[static_cast<std::size_t>(p)];
    const int e  = m_starts[static_cast<std::size_t>(p + 1)];
    const int bs = e - s;

    m_blockSizes[static_cast<std::size_t>(p)] = bs;
    auto& blockLU = m_LUblocks[static_cast<std::size_t>(p)];
    blockLU.assign(static_cast<std::size_t>(bs * bs), 0.0);

    A.forEachNZ([&](Index r, Index c, Scalar v) {
      int ri = static_cast<int>(r);
      int ci = static_cast<int>(c);
      if (ri >= s && ri < e && ci >= s && ci < e) {
        int lr = ri - s; 
        int lc = ci - s; 
        blockLU[lr * bs + lc] = static_cast<double>(v);
      }
    });

    lu_factor_block(blockLU, bs);
  }
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
    const int e  = m_starts[static_cast<std::size_t>(p + 1)];
    const int bs = m_blockSizes[static_cast<std::size_t>(p)];

    const auto& LU = m_LUblocks[static_cast<std::size_t>(p)];
    if (bs <= 0) continue;

    std::vector<double> rhs(bs);
    std::vector<double> sol(bs);

    for (int i = 0; i < bs; ++i) {
      rhs[static_cast<std::size_t>(i)] = r[static_cast<std::size_t>(s + i)];
    }

    lu_solve_block(LU, bs, rhs.data(), sol.data());

    for (int i = 0; i < bs; ++i) {
      z[static_cast<std::size_t>(s + i)] = sol[static_cast<std::size_t>(i)];
    }
  }
}
