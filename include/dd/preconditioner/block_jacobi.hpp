#pragma once

/**
 * @file block_jacobi.hpp
 *
 * Minimal block Jacobi (non‑overlapping additive Schwarz) preconditioner.
 *
 * This class partitions the rows of a sparse matrix into a fixed number of
 * contiguous blocks and inverts the diagonal of each block.  It uses
 * `std::vector<double>` to represent vectors rather than relying on a
 * custom `Vector` class, and it accepts any matrix derived from
 * `dd::algebra::matrixSparse` (e.g., CSR or COO).  The diagonal is
 * extracted by downcasting to a CSR matrix at runtime; if this fails,
 * the diagonal entries are assumed to be nonzero and equal to one.
 */

#include <vector>
#include <stdexcept>
#include <cstddef>


#include "preconditioner.hpp"          // dd::algebra::Preconditioner
#include "dd/algebra/matrixSparse.hpp" // dd::algebra::MatrixSparse
#include "dd/algebra/CSR.hpp"          // dd::algebra::CSR  (for diag extraction)
#include "dd/algebra/COO.hpp"          // dd::algebra::COO  (for diag extraction)

// Forward declarations of sparse matrix classes.
namespace dd {
namespace algebra {
class MatrixSparse;
class CSR;
class COO;
} // namespace algebra
} // namespace dd



namespace dd { namespace algebra {


/**
 * @class BlockJacobi
 * @brief A simple block Jacobi preconditioner using std::vector.
 *
 * The class stores the inverse of the diagonal entries of a sparse
 * matrix and partitions the problem into `m_nparts` contiguous blocks.
 * During `apply()` each component of the right-hand side is scaled by
 * the inverse diagonal.  This is equivalent to a one‑by‑one block
 * Jacobi preconditioner.  It does not depend on any specific sparse
 * matrix storage format beyond what is required to extract the diagonal
 * through a dynamic cast.
 */
class BlockJacobi final : public Preconditioner {
public:
  /// Construct a block Jacobi preconditioner with a given number of parts.
  explicit BlockJacobi(int nparts = 1);

  /// Configure the preconditioner with a general sparse matrix.
  void setup(const dd::algebra::MatrixSparse& A);

  /// Apply the preconditioner to the residual `r`, writing the result to `z`.
  void apply(const std::vector<double>& r, std::vector<double>& z) const;

  /// Return the number of blocks.
  int parts() const { return m_nparts; }

  /// Return the block starting indices (size = nparts+1).
  const std::vector<int>& block_starts() const { return m_starts; }

private:
  /// Extract the diagonal entry A(i,i) from a general sparse matrix.  If the
  /// matrix is not a CSR or COO, this returns zero.
  static double diag_at(const dd::algebra::MatrixSparse& A, int i);

  int m_nparts;                 ///< Number of blocks
  int m_n;                      ///< Global problem size
  std::vector<int> m_starts;    ///< Block boundaries
  std::vector<double> m_inv_diag; ///< Inverse diagonal values
};


}} // namespace dd