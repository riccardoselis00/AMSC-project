#pragma once

#include <vector>
#include <cstddef>
#include <string>
#include <initializer_list>
#include <stdexcept>
#include <fstream>
#include <cctype>
#include <limits>

namespace dd { namespace algebra {

/**
 * @brief Simple row‑major dense matrix (double precision) with a minimal, HPC‑friendly API.
 *
 * The matrix stores values contiguously in row‑major order. It provides
 * element accessors, basic construction, algebraic operations, simple
 * utilities such as extracting the diagonal and computing norms, and
 * supports resizing with or without preserving existing contents. No
 * external BLAS library is required; all operations use straightforward
 * triply nested loops where appropriate.
 */
class MatrixDense {
public:
    using Scalar = double;
    using Index  = std::size_t;

    /**
     * @brief Default constructor, yields a 0×0 empty matrix.
     */
    MatrixDense() : m_rows(0), m_cols(0) {}

    /**
     * @brief Construct a matrix of dimensions @p rows × @p cols, filling all
     * elements with @p init_value.
     */
    MatrixDense(Index rows, Index cols, Scalar init_value = Scalar{0});

    /**
     * @brief Construct from a nested initializer list. All rows must
     * contain the same number of columns or a std::runtime_error is thrown.
     */
    MatrixDense(std::initializer_list<std::initializer_list<Scalar>> rows);

    /// Create a rows×cols zero matrix.
    static MatrixDense Zero(Index rows, Index cols) { return MatrixDense(rows, cols, Scalar{0}); }

    /// Create an identity matrix of size n×n.
    static MatrixDense Identity(Index n);

    // ----- Accessors -----

    /// Number of rows.    /// Direct constructor from ptr/col/val arrays (no copies). Validates sizes.

    inline Index rows() const noexcept { return m_rows; }
    /// Number of columns.
    inline Index cols() const noexcept { return m_cols; }
    /// Total number of elements (rows × cols).
    inline Index size() const noexcept { return m_data.size(); }
    /// Whether the matrix has no elements.
    inline bool empty() const noexcept { return m_data.empty(); }

    /// Direct access to raw storage.
    inline Scalar*       data()       noexcept { return m_data.data(); }
    inline const Scalar* data() const noexcept { return m_data.data(); }

    /// Unchecked element access (fast path). Caller must ensure valid indices.
    inline Scalar& operator()(Index i, Index j) noexcept { return m_data[i * m_cols + j]; }
    inline const Scalar& operator()(Index i, Index j) const noexcept { return m_data[i * m_cols + j]; }

    /// Bounds‑checked element access; throws std::out_of_range on invalid indices.
    Scalar& at(Index i, Index j);
    const Scalar& at(Index i, Index j) const;

    // ----- Mutators -----

    /// Set every element to the value @p v.
    void fill(Scalar v) noexcept;
    /// Set all elements to zero.
    void setZero() noexcept { fill(Scalar{0}); }
    /// Make the matrix an identity matrix; requires it to be square or throws.
    void setIdentity();
    /**
     * @brief Resize the matrix to @p rows × @p cols.
     *
     * If @p keep is false, the matrix contents are discarded and the new
     * elements are filled with @p pad_value. If @p keep is true, the
     * overlapping part of the old matrix is copied into the new matrix and
     * any newly created elements are set to @p pad_value.
     */
    void resize(Index rows, Index cols, bool keep = false, Scalar pad_value = Scalar{0});

    // ----- Algebraic operations -----

    /**
     * @brief Compute y = α·A·x + β·y in place (raw pointers version).
     *
     * The vectors x and y must have length cols() and rows() respectively.
     */
    void gemv(const Scalar* x, Scalar* y, Scalar alpha = 1.0, Scalar beta = 0.0) const noexcept;

    /// Vector overload: compute y = α·A·x + β·y; allocates y if necessary.
    void gemv(const std::vector<Scalar>& x, std::vector<Scalar>& y, Scalar alpha = 1.0, Scalar beta = 0.0) const;

    /// Convenience overload: returns y = A·x.
    std::vector<Scalar> gemv(const std::vector<Scalar>& x) const;

    /**
     * @brief Compute C = α A·B + β·C.
     *
     * Dimensions must match: A(m×k), B(k×n), C(m×n). Uses a naive
     * triple‑nested loop; does not rely on external BLAS.
     */
    static void gemm(const MatrixDense& A, const MatrixDense& B, MatrixDense& C,
                     Scalar alpha = 1.0, Scalar beta = 0.0);

    /// In‑place scaling: A = α·A.
    void scale(Scalar alpha) noexcept;
   
    /// A = A + α·B; throws if dimensions mismatch.
    void axpy(Scalar alpha, const MatrixDense& B);

    // ----- Transposition -----

    /// Return the transpose of the matrix.
    MatrixDense transpose() const;
    
    /// In‑place transpose; requires a square matrix.
    void transposeInPlace();

    // ----- Norms & diagnostics -----

    /// Frobenius norm: sqrt(∑|a_ij|^2).
    Scalar frobeniusNorm() const noexcept;
    
    /// Maximum absolute value of any element.
    Scalar maxAbs() const noexcept;

    // ----- Misc -----

    /**
     * @brief Fill @p d with the diagonal of the matrix (length min(rows,cols)).
     */
    void extractDiagonal(std::vector<Scalar>& d) const;

    /**
     * @brief A human‑readable string preview of the matrix contents.
     *
     * Displays up to @p max_rows and @p max_cols entries; additional rows or
     * columns are elided with an ellipsis.
     */
    std::string toString(Index max_rows = 6, Index max_cols = 6) const;

    /**
     * @brief Copy and return a subblock of this matrix.
     *
     * Returns an r×c dense block starting at (i0,j0). Throws if the block
     * extends beyond the bounds of the matrix.
     */
    MatrixDense block(Index i0, Index j0, Index r, Index c) const;

    // === Matrix Market (array) I/O ===
    // Load from a Matrix Market 'array' file (double, real/integer). Throws std::runtime_error on parse errors.
    static MatrixDense from_mm_array_file(const std::string& filename);
    static MatrixDense from_mm_array_stream(std::istream& is);

    // Save to Matrix Market 'array' format. symmetry ∈ {"general","symmetric","skew-symmetric"}.
    // Returns true on success; throws on shape violations (e.g., symmetry on non-square).
    bool to_mm_array_file(const std::string& filename, const std::string& symmetry = "general") const;
    bool to_mm_array_stream(std::ostream& os, const std::string& symmetry = "general") const;

private:
    Index m_rows{};
    Index m_cols{};
    std::vector<Scalar> m_data; // row‑major storage
};

}} // namespace dd::algebra