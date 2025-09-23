#pragma once

#include <vector>
#include <cstddef>
#include <memory>
#include <stdexcept>

#include "dd/algebra/matrixSparse.hpp"
// #include "dd/algebra/CSR.hpp"

namespace dd { namespace algebra {

    class MatrixCSR;
/**
 * @brief Sparse matrix in Coordinate (COO) format.
 *
 * Stores three parallel arrays: row indices, column indices and values.
 * Provides basic algebraic operations similar to MatrixDense where
 * appropriate, along with operations specific to COO storage.
 */
class MatrixCOO final : public MatrixSparse {
public:
    using Scalar = MatrixSparse::Scalar;
    using Index  = MatrixSparse::Index;

    using MatrixSparse::gemv;



    /// Construct an empty 0×0 matrix.
    MatrixCOO() = default;
    /// Construct an empty matrix with fixed dimension; no nonzeros initially.
    MatrixCOO(Index rows, Index cols) : m_rows(rows), m_cols(cols) {}

    /// Construct from row/col/val arrays; sizes must match and all indices must be within bounds.
    MatrixCOO(Index rows, Index cols,
              std::vector<Index> row, std::vector<Index> col,
              std::vector<Scalar> val)
        : m_rows(rows), m_cols(cols), m_row(std::move(row)), m_col(std::move(col)), m_val(std::move(val))
    {
        if (m_row.size() != m_col.size() || m_row.size() != m_val.size())
            throw std::runtime_error("MatrixCOO: row/col/value sizes mismatch");
        validateIndices_();
    }

    /// Construct from an initializer list of triplets { {i,j,v}, ... } and optional shape.
    MatrixCOO(Index rows, Index cols, std::initializer_list<std::tuple<Index,Index,Scalar>> triplets)
        : m_rows(rows), m_cols(cols)
    {
        for (auto& t : triplets) {
            Index i,j; Scalar v;
            std::tie(i,j,v) = t;
            add(i,j,v);
        }
    }

    explicit MatrixCOO(const MatrixCSR& A);

    static MatrixCOO Poisson2D(Index n);

    /// Create a zero matrix with given shape.
    static MatrixCOO Zero(Index rows, Index cols) { return MatrixCOO(rows, cols); }
    /// Create an identity matrix of size n×n.
    static MatrixCOO Identity(Index n)
    {
        MatrixCOO I(n, n);
        I.reserve(n);
        for (Index i = 0; i < n; ++i) I.add(i, i, Scalar{1});
        return I;
    }

    // ----- I/O -----

    static void lower_inplace(char* s);

    //MatrixCOO(const std::string& filename); // throws on error
    static MatrixCOO read_COO(const std::string& filename); // throws on error
    
    bool write_COO(const std::string& filename) const; // returns false on error


    // ----- Assembly -----
    /// Reserve space for nnz nonzeros.
    void reserve(Index nnz) { m_row.reserve(nnz); m_col.reserve(nnz); m_val.reserve(nnz); }
    /// Append a nonzero entry (i,j) = v. Throws if i>=rows or j>=cols.
    void add(Index i, Index j, Scalar v)
    {
        if (i >= m_rows || j >= m_cols)
            throw std::out_of_range("MatrixCOO::add: index out of bounds");
        m_row.push_back(i);
        m_col.push_back(j);
        m_val.push_back(v);
    }

    // ----- MatrixSparse interface -----
    Index rows() const noexcept override { return m_rows; }
    Index cols() const noexcept override { return m_cols; }
    Index nnz()  const noexcept override { return static_cast<Index>(m_val.size()); }
    SparseFormat format() const noexcept override { return SparseFormat::COO; }

    void gemv(const Scalar* x, Scalar* y,
              Scalar alpha = 1.0, Scalar beta = 0.0) const override;

    void extractDiagonal(std::vector<Scalar>& d) const override;

    void forEachNZ(const TripletVisitor& f) const override
    {
        for (Index k = 0; k < nnz(); ++k) f(m_row[k], m_col[k], m_val[k]);
    }

    std::unique_ptr<MatrixSparse> clone() const override
    {
        return std::make_unique<MatrixCOO>(*this);
    }

    // ----- Extra utilities -----
    /// In-place scale all values by a factor.
    void scale(Scalar alpha) noexcept { for (auto& v : m_val) v *= alpha; }

    /**
     * @brief A = A + alpha * B. Pattern is not required to match; duplicates are not coalesced.
     *
     * Complexity is O(nnz(A) + nnz(B)). Duplicated entries will remain and
     * affect subsequent operations accordingly.
     */
    void axpy(Scalar alpha, const MatrixCOO& B)
    {
        if (m_rows != B.m_rows || m_cols != B.m_cols)
            throw std::runtime_error("MatrixCOO::axpy: shape mismatch");
        reserve(nnz() + B.nnz());
        for (Index k = 0; k < B.nnz(); ++k) {
            add(B.m_row[k], B.m_col[k], alpha * B.m_val[k]);
        }
    }

    /// A = A + alpha * B, but requires identical sparsity pattern (same row/col arrays).
    void axpySamePattern(Scalar alpha, const MatrixCOO& B);

    /// Return the transpose of this matrix.
    MatrixCOO transpose() const;

    /// Access raw row indices.
    const std::vector<Index>& rowIndex() const noexcept { return m_row; }
    /// Access raw col indices.
    const std::vector<Index>& colIndex() const noexcept { return m_col; }
    /// Access raw values.
    const std::vector<Scalar>& values()  const noexcept { return m_val; }

private:
    void validateIndices_() const
    {
        for (Index k = 0; k < nnz(); ++k) {
            if (m_row[k] >= m_rows || m_col[k] >= m_cols)
                throw std::runtime_error("MatrixCOO: index out of bounds");
        }
    }

    Index m_rows{0}, m_cols{0};
    std::vector<Index>  m_row;
    std::vector<Index>  m_col;
    std::vector<Scalar> m_val;
};

}} // namespace dd::algebra