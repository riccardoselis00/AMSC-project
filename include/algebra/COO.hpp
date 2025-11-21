#pragma once

#include <vector>
#include <cstddef>
#include <memory>
#include <stdexcept>

#include "algebra/matrixSparse.hpp"
#include "algebra/COO.hpp"
#include "algebra/CSR.hpp"


class MatrixCSR;


class MatrixCOO final : public MatrixSparse {
public:
    using Scalar = MatrixSparse::Scalar;
    using Index  = MatrixSparse::Index;


    using MatrixSparse::gemv;

    MatrixCOO() = default;

    MatrixCOO(Index rows, Index cols) : m_rows(rows), m_cols(cols) {}

    MatrixCOO(Index rows, Index cols,
              std::vector<Index> row, std::vector<Index> col,
              std::vector<Scalar> val)
        : m_rows(rows), m_cols(cols), m_row(std::move(row)), m_col(std::move(col)), m_val(std::move(val))
    {
        if (m_row.size() != m_col.size() || m_row.size() != m_val.size())
            throw std::runtime_error("MatrixCOO: row/col/value sizes mismatch");
        validateIndices_();
    }

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

    static MatrixCOO Zero(Index rows, Index cols) { return MatrixCOO(rows, cols); }

    static MatrixCOO Identity(Index n)
    {
        MatrixCOO I(n, n);
        I.reserve(n);
        for (Index i = 0; i < n; ++i) I.add(i, i, Scalar{1});
        return I;
    }

    static void lower_inplace(char* s);

    static MatrixCOO read_COO(const std::string& filename); 
    
    bool write_COO(const std::string& filename) const; 

    void reserve(Index nnz) { m_row.reserve(nnz); m_col.reserve(nnz); m_val.reserve(nnz); }

    void add(Index i, Index j, Scalar v)
    {
        if (i >= m_rows || j >= m_cols)
            throw std::out_of_range("MatrixCOO::add: index out of bounds");
        m_row.push_back(i);
        m_col.push_back(j);
        m_val.push_back(v);
    }

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

    void scale(Scalar alpha) noexcept { for (auto& v : m_val) v *= alpha; }

    void axpy(Scalar alpha, const MatrixCOO& B)
    {
        if (m_rows != B.m_rows || m_cols != B.m_cols)
            throw std::runtime_error("MatrixCOO::axpy: shape mismatch");
        reserve(nnz() + B.nnz());
        for (Index k = 0; k < B.nnz(); ++k) {
            add(B.m_row[k], B.m_col[k], alpha * B.m_val[k]);
        }
    }

    void axpySamePattern(Scalar alpha, const MatrixCOO& B);

    MatrixCOO transpose() const;

    const std::vector<Index>& rowIndex() const noexcept { return m_row; }

    const std::vector<Index>& colIndex() const noexcept { return m_col; }

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