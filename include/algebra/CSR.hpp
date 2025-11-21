#pragma once

#include <vector>
#include <cstddef>
#include <memory>
#include <stdexcept>

#include "algebra/matrixSparse.hpp"
#include "algebra/CSR.hpp"
#include "algebra/COO.hpp"

class MatrixCOO;

    
class MatrixCSR final : public MatrixSparse {

public:

    using Scalar = MatrixSparse::Scalar;
    using Index  = MatrixSparse::Index;

    using MatrixSparse::gemv;


    MatrixCSR() = default;
    
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

    explicit MatrixCSR(const MatrixCOO& A);

    static MatrixCSR Zero(Index rows, Index cols)
    {
        MatrixCSR Z;
        Z.m_rows = rows;
        Z.m_cols = cols;
        Z.m_ptr.assign(rows + 1, 0);
        return Z;
    }

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


    void scale(Scalar alpha) noexcept { for (auto& v : m_val) v *= alpha; }

    void axpySamePattern(Scalar alpha, const MatrixCSR& B);

    MatrixCSR transpose() const;

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
    std::vector<Index>  m_ptr;
    std::vector<Index>  m_col;
    std::vector<Scalar> m_val;
};