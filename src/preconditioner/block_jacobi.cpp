#include "algebra/matrixSparse.hpp" 
#include "algebra/CSR.hpp"            
#include "algebra/COO.hpp"           
#include "preconditioner/block_jacobi.hpp"

#include <typeinfo>
#include <stdexcept>


BlockJacobi::BlockJacobi(int nparts)
    : m_nparts(nparts), m_n(0) {
  if (m_nparts <= 0) {
    throw std::invalid_argument("BlockJacobi: nparts must be > 0");
  }
}

static double diag_from_csr(const MatrixCSR& A, int i) {
  using Index = MatrixSparse::Index;
  const auto& rowPtr = A.rowPtr();
  const auto& colIdx = A.colIndex();
  const auto& vals   = A.values();

  const Index ii  = static_cast<Index>(i);
  const Index beg = rowPtr[ii];
  const Index end = rowPtr[ii + 1];
  for (Index k = beg; k < end; ++k) {
    if (colIdx[k] == ii) return static_cast<double>(vals[k]);
  }
  return 0.0;
}

double BlockJacobi::diag_at(const MatrixSparse& A, int i) {
  if (auto csr = dynamic_cast<const MatrixCSR*>(&A)) {
    return diag_from_csr(*csr, i);
  }

  double aii = 0.0;
  using Index  = MatrixSparse::Index;
  using Scalar = MatrixSparse::Scalar;

  A.forEachNZ([&](Index r, Index c, Scalar v) {
    if (static_cast<int>(r) == i && static_cast<int>(c) == i) {
      aii = static_cast<double>(v);
    }
  });
  return aii;
}


void BlockJacobi::setup(const MatrixSparse& A) {
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

  m_inv_diag.assign(static_cast<std::size_t>(m_n), 0.0);
  for (Index i = 0; i < m_n; ++i) {
    Scalar aii = diag_at(A, i);
    if (aii == Scalar{0}) {
      throw std::runtime_error("BlockJacobi::setup: zero diagonal at row " + std::to_string(i));
    }
    m_inv_diag[static_cast<std::size_t>(i)] = Scalar{1} / aii;
  }
}

void BlockJacobi::apply(const std::vector<double>& r,
                        std::vector<double>& z) const {
  if (r.size() != static_cast<std::size_t>(m_n)) {
    throw std::runtime_error("BlockJacobi::apply: r has wrong size");
  }
  z.assign(r.size(), 0.0);
  for (int p = 0; p < m_nparts; ++p) {
    int s = m_starts[p];
    int e = m_starts[p + 1];
    for (int i = s; i < e; ++i) {
      z[static_cast<std::size_t>(i)] = m_inv_diag[static_cast<std::size_t>(i)] * r[static_cast<std::size_t>(i)];
    }
  }
}