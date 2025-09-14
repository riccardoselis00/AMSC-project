#pragma once

#include <vector>
#include <cstddef>
#include <memory>
#include <stdexcept>

#include "dd/algebra/matrixSparse.hpp"
// #include "dd/algebra/COO.hpp"

namespace dd { namespace algebra {

    class MatrixCOO;

/**
 * @brief Sparse matrix in Compressed Sparse Row (CSR) format.
 *
 * Stores three arrays: row pointers (ptr), column indices (col) and
 * values (val). Provides efficient row traversal and matrix‑vector
 * multiplication. Constructible from COO.
 */
class MatrixCSR final : public MatrixSparse {
public:
    using Scalar = MatrixSparse::Scalar;
    using Index  = MatrixSparse::Index;

    using MatrixSparse::gemv;


    MatrixCSR() = default;
    
    /// Direct constructor from ptr/col/val arrays (no copies). Validates sizes.
    
    MatrixCSR(Index rows, Index cols,
              std::vector<Index> ptr,
              std::vector<Index> col,
              std::vector<Scalar> val)
        : m_rows(rows), m_cols(cols), m_ptr(std::move(ptr)), m_col(std::move(col)), m_val(std::move(val))
    {
        if (m_ptr.size() != m_rows + 1)
            throw std::runtime_error("MatrixCSR: ptr size must equal rows+1");
        if (m_col.size() != m_val.size())
            throw std::runtime_error("MatrixCSR: col and val sizes mismatch");
        if (m_ptr.back() != m_col.size())
            throw std::runtime_error("MatrixCSR: ptr.back() must equal nnz");
        validateIndices_();
    }

    /// Construct from a COO matrix. Performs conversion (duplicates not coalesced).
    explicit MatrixCSR(const MatrixCOO& A);

    /// Create a zero matrix with given shape.
    static MatrixCSR Zero(Index rows, Index cols)
    {
        MatrixCSR Z;
        Z.m_rows = rows;
        Z.m_cols = cols;
        Z.m_ptr.assign(rows + 1, 0);
        return Z;
    }

    /// Create an identity matrix of size n×n.
    static MatrixCSR Identity(Index n)
    {
        MatrixCSR I;
        I.m_rows = n;
        I.m_cols = n;
        I.m_ptr.resize(n + 1);
        I.m_col.resize(n);
        I.m_val.resize(n);
        for (Index i = 0; i < n; ++i) {
            I.m_ptr[i] = i;
            I.m_col[i] = i;
            I.m_val[i] = Scalar{1};
        }
        I.m_ptr[n] = n;
        return I;
    }

    // ----- MatrixSparse interface -----
    Index rows() const noexcept override { return m_rows; }
    Index cols() const noexcept override { return m_cols; }
    Index nnz()  const noexcept override { return static_cast<Index>(m_val.size()); }
    SparseFormat format() const noexcept override { return SparseFormat::CSR; }

    void gemv(const Scalar* x, Scalar* y,
              Scalar alpha = 1.0, Scalar beta = 0.0) const override;

    void extractDiagonal(std::vector<Scalar>& d) const override;

    void forEachNZ(const TripletVisitor& f) const override
    {
        for (Index i = 0; i < m_rows; ++i) {
            for (Index k = m_ptr[i]; k < m_ptr[i+1]; ++k) {
                f(i, m_col[k], m_val[k]);
            }
        }
    }

    std::unique_ptr<MatrixSparse> clone() const override
    {
        return std::make_unique<MatrixCSR>(*this);
    }

    // ----- Extra utilities -----
    /// In-place scale all values by a factor.
    void scale(Scalar alpha) noexcept { for (auto& v : m_val) v *= alpha; }

    /**
     * @brief A = A + alpha * B, requires identical sparsity pattern.
     *
     * Patterns must match exactly (same ptr and col arrays). Throws otherwise.
     */
    void axpySamePattern(Scalar alpha, const MatrixCSR& B);

    /// Return the transpose of this matrix.
    MatrixCSR transpose() const;

    /// Access the CSR arrays (read-only).
    const std::vector<Index>& rowPtr() const noexcept { return m_ptr; }
    const std::vector<Index>& colIndex() const noexcept { return m_col; }
    const std::vector<Scalar>& values()  const noexcept { return m_val; }

private:
    void validateIndices_() const
    {
        for (Index i = 0; i < m_rows; ++i) {
            for (Index k = m_ptr[i]; k < m_ptr[i+1]; ++k) {
                if (m_col[k] >= m_cols)
                    throw std::runtime_error("MatrixCSR: column index out of bounds");
            }
        }
    }

    Index m_rows{0}, m_cols{0};
    std::vector<Index>  m_ptr; // size = rows+1
    std::vector<Index>  m_col;
    std::vector<Scalar> m_val;
};

}} // namespace dd::algebra