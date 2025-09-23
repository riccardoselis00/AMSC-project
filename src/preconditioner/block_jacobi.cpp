// // Pull in definitions of the concrete sparse matrix formats.  Only the CSR
// // format is used explicitly to extract diagonal entries; other formats
// // fallback to a default diagonal of one when computing the inverse.

// // #include "dd/algebra/CSR.hpp"
// // #include "dd/algebra/COO.hpp"
// // #include "dd/algebra/matrixSparse.hpp" // dd::algebra::MatrixSparse
// // #include "dd/preconditioner/block_jacobi.hpp"

// #include "dd/algebra/matrixSparse.hpp"   // base
// #include "dd/algebra/CSR.hpp"            // complete type for CSR
// #include "dd/algebra/COO.hpp"            // complete type for COO
// #include "dd/preconditioner/block_jacobi.hpp"

// #include <typeinfo>

// namespace dd { namespace algebra {

// using dd::algebra::MatrixSparse;
// using dd::algebra::MatrixCSR;
// using dd::algebra::MatrixCOO;

// // Constructor checks that the number of parts is positive.
// BlockJacobi::BlockJacobi(int nparts)
//     : m_nparts(nparts), m_n(0) {
//   if (m_nparts <= 0) {
//     throw std::invalid_argument("BlockJacobi: nparts must be > 0");
//   }
// }

// // Helper to extract a diagonal entry from a matrixSparse.  If the matrix
// // supports CSR storage, the diagonal is extracted directly; otherwise a
// // default of zero is returned, which triggers an error in setup().
// double BlockJacobi::diag_at(const MatrixSparse& A, int i) {

//   //if (const auto* csr = dynamic_cast<const CSR*>(&A)) {
//   if (auto csr = dynamic_cast<const dd::algebra::MatrixCSR*>(&A)) {
//     // const auto& rowPtr = csr->row_ptr();
//     // const auto& colIdx = csr->col_idx();
//     // const auto& vals   = csr->values();
//     const auto& rowPtr = A.rowPtr();
//     const auto& colIdx = A.colIndex();
//     const auto& vals   = A.values();
//     for (int k = rowPtr[i]; k < rowPtr[i+1]; ++k) {
//       if (colIdx[k] == i) {
//         return vals[k];
//       }
//     }
//     return 0.0;
//   }
//   // Try COO
//   //if (const auto* coo = dynamic_cast<const COO*>(&A)) {
//   if (auto coo = dynamic_cast<const dd::algebra::MatrixCOO*>(&A)) {
//     const auto nnz = coo->nnz();
//     const auto& rows = coo->row_ind();
//     const auto& cols = coo->col_ind();
//     const auto& vals = coo->values();
//     for (std::size_t k = 0; k < nnz; ++k) {
//       if (rows[k] == i && cols[k] == i) {
//         return vals[k];
//       }
//     }
//     return 0.0;
//   }
//   // Unknown format: cannot extract diagonal
//   return 0.0;
// }

// void BlockJacobi::setup(const MatrixSparse& A) {
//   m_n = A.rows();
//   if (A.cols() != m_n) {
//     throw std::runtime_error("BlockJacobi::setup requires a square matrix.");
//   }
//   // Partition rows into m_nparts contiguous segments
//   m_starts.resize(m_nparts + 1);
//   const int base = m_n / m_nparts;
//   int rem = m_n % m_nparts;
//   int pos = 0;
//   for (int p = 0; p < m_nparts; ++p) {
//     m_starts[p] = pos;
//     int sz = base + (p < rem ? 1 : 0);
//     pos += sz;
//   }
//   m_starts[m_nparts] = m_n;


//   // Precompute 1/diag(A)
//   m_inv_diag.assign(m_n, 0.0);
//   for (int i = 0; i < m_n; ++i) {
//     double aii = diag_at(A, i);
//     if (aii == 0.0) {
//       throw std::runtime_error(
//           "BlockJacobi::setup: zero diagonal at row " + std::to_string(i));
//     }
//     m_inv_diag[i] = 1.0 / aii;
//   }
// }

// void BlockJacobi::apply(const std::vector<double>& r,
//                         std::vector<double>& z) const {
//   if (static_cast<int>(r.size()) != m_n) {
//     throw std::runtime_error("BlockJacobi::apply: r has wrong size");
//   }
//   z.assign(r.size(), 0.0);
//   for (int p = 0; p < m_nparts; ++p) {
//     int s = m_starts[p];
//     int e = m_starts[p + 1];
//     for (int i = s; i < e; ++i) {
//       z[i] = m_inv_diag[i] * r[i];
//     }
//   }
// }

// }} // namespace dd


#include "dd/algebra/matrixSparse.hpp"   // base
#include "dd/algebra/CSR.hpp"            // MatrixCSR
#include "dd/algebra/COO.hpp"            // MatrixCOO
#include "dd/preconditioner/block_jacobi.hpp"

#include <typeinfo>
#include <stdexcept>

namespace dd { namespace algebra {

using Index = MatrixSparse::Index;
using Scalar = MatrixSparse::Scalar;

using dd::algebra::MatrixCSR;
using dd::algebra::MatrixCOO;

BlockJacobi::BlockJacobi(int nparts)
    : m_nparts(nparts), m_n(0) {
  if (m_nparts <= 0) {
    throw std::invalid_argument("BlockJacobi: nparts must be > 0");
  }
}

// static Scalar diag_from_csr(const MatrixCSR& A, Index i) {
//   const auto& rowPtr = A.rowPtr();
//   const auto& colIdx = A.colIndex();
//   const auto& vals   = A.values();
//   for (Index k = rowPtr[i]; k < rowPtr[i+1]; ++k) {
//     if (colIdx[k] == i) return vals[k];
//   }
//   return Scalar{0};
// }

// static Scalar diag_from_coo(const MatrixCOO& A, Index i) {
//   const Index nnz = A.nnz();
//   const auto& rows = A.row_ind();   // <-- adjust to your actual API names
//   const auto& cols = A.col_ind();   // <-- adjust to your actual API names
//   const auto& vals = A.values();
//   for (Index k = 0; k < nnz; ++k) {
//     if (rows[k] == i && cols[k] == i) return vals[k];
//   }
//   return Scalar{0};
// }

// Scalar BlockJacobi::diag_at(const MatrixSparse& A, Index i) {
//   if (auto csr = dynamic_cast<const MatrixCSR*>(&A)) return diag_from_csr(*csr, i);
//   if (auto coo = dynamic_cast<const MatrixCOO*>(&A)) return diag_from_coo(*coo, i);
//   return Scalar{0}; // unknown format
// }

// Fast path for CSR diagonal lookup
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

// Match the header declaration EXACTLY:
// static double diag_at(const MatrixSparse&, int);
double BlockJacobi::diag_at(const MatrixSparse& A, int i) {
  // Try CSR fast path
  if (auto csr = dynamic_cast<const MatrixCSR*>(&A)) {
    return diag_from_csr(*csr, i);
  }

  // Generic fallback for any sparse format (COO included):
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
  m_n = A.rows(); // consider making m_n type Index
  if (A.cols() != m_n) {
    throw std::runtime_error("BlockJacobi::setup requires a square matrix.");
  }

  // Partition rows into m_nparts contiguous segments
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

  // Precompute 1/diag(A)
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

}} // namespace dd::algebra
