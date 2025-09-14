#include "dd/algebra/CSR.hpp"
#include "dd/algebra/COO.hpp"

#include <algorithm>
#include <numeric>

namespace dd { namespace algebra {

MatrixCSR::MatrixCSR(const MatrixCOO& A)
{
    m_rows = A.rows();
    m_cols = A.cols();
    const Index nz = A.nnz();
    m_ptr.assign(m_rows + 1, 0);
    m_col.resize(nz);
    m_val.resize(nz);
    // Count nnz per row
    for (Index k = 0; k < nz; ++k) {
        ++m_ptr[A.rowIndex()[k]];
    }
    // Prefix sum to get starting positions
    Index sum = 0;
    for (Index i = 0; i < m_rows; ++i) {
        Index cnt = m_ptr[i];
        m_ptr[i] = sum;
        sum += cnt;
    }
    m_ptr[m_rows] = sum;
    // Copy of row pointer for insertion positions
    std::vector<Index> next = m_ptr;
    // Scatter
    for (Index k = 0; k < nz; ++k) {
        Index r = A.rowIndex()[k];
        Index pos = next[r]++;
        m_col[pos] = A.colIndex()[k];
        m_val[pos] = A.values()[k];
    }
}

void MatrixCSR::gemv(const Scalar* x, Scalar* y,
                      Scalar alpha, Scalar beta) const
{
    for (Index i = 0; i < m_rows; ++i) {
        Scalar sum = Scalar{0};
        for (Index k = m_ptr[i]; k < m_ptr[i+1]; ++k) {
            sum += m_val[k] * x[m_col[k]];
        }
        y[i] = alpha * sum + beta * y[i];
    }
}

void MatrixCSR::extractDiagonal(std::vector<Scalar>& d) const
{
    const Index n = std::min(m_rows, m_cols);
    d.assign(n, Scalar{0});
    for (Index i = 0; i < m_rows; ++i) {
        for (Index k = m_ptr[i]; k < m_ptr[i+1]; ++k) {
            if (m_col[k] == i) {
                d[i] += m_val[k];
            }
        }
    }
}

void MatrixCSR::axpySamePattern(Scalar alpha, const MatrixCSR& B)
{
    if (m_rows != B.m_rows || m_cols != B.m_cols)
        throw std::runtime_error("MatrixCSR::axpySamePattern: shape mismatch");
    if (m_ptr != B.m_ptr || m_col != B.m_col)
        throw std::runtime_error("MatrixCSR::axpySamePattern: pattern mismatch");
    for (Index k = 0; k < m_val.size(); ++k) {
        m_val[k] += alpha * B.m_val[k];
    }
}

MatrixCSR MatrixCSR::transpose() const
{
    MatrixCSR T;
    T.m_rows = m_cols;
    T.m_cols = m_rows;
    T.m_ptr.assign(T.m_rows + 1, 0);
    T.m_col.resize(m_val.size());
    T.m_val.resize(m_val.size());
    // Count nnz per column (becomes row in transpose)
    for (Index k = 0; k < m_col.size(); ++k) {
        ++T.m_ptr[m_col[k]];
    }
    // Prefix sum
    Index sum = 0;
    for (Index r = 0; r < T.m_rows; ++r) {
        Index cnt = T.m_ptr[r];
        T.m_ptr[r] = sum;
        sum += cnt;
    }
    T.m_ptr[T.m_rows] = sum;
    std::vector<Index> next = T.m_ptr;
    // Scatter
    for (Index i = 0; i < m_rows; ++i) {
        for (Index k = m_ptr[i]; k < m_ptr[i+1]; ++k) {
            Index r = m_col[k];
            Index pos = next[r]++;
            T.m_col[pos] = i;
            T.m_val[pos] = m_val[k];
        }
    }
    return T;
}

}} // namespace dd::algebra