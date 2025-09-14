#include "dd/algebra/COO.hpp"
#include "dd/algebra/CSR.hpp"

#include <algorithm>
#include <cmath>

namespace dd { namespace algebra {

// MatrixCOO::MatrixCOO(const MatrixCSR& A)
// {
//     m_rows = A.rows();
//     m_cols = A.cols();
//     const Index nz = A.nnz();
//     m_row.resize(nz);
//     m_col.resize(nz);
//     m_val.resize(nz);
//     // Scatter
//     for (Index i = 0; i < m_rows; ++i) {
//         for (Index k = A.rowPtr()[i]; k < A.rowPtr()[i+1]; ++k) {
//             m_row[k] = i;
//             m_col[k] = A.colIndex()[k];
//             m_val[k] = A.values()[k];
//         }
//     }
// }

MatrixCOO::MatrixCOO(const MatrixCSR& A)
: m_rows(A.rows()), m_cols(A.cols())
{
    reserve(A.nnz());

    const auto& ptr = A.rowPtr();    // size = m_rows + 1
    const auto& col = A.colIndex();  // size = nnz
    const auto& val = A.values();    // size = nnz

    for (Index i = 0; i < m_rows; ++i) {
        for (Index p = ptr[i]; p < ptr[i + 1]; ++p) {
            add(i, col[p], val[p]);  // push (i, j, a_ij) in COO
        }
    }
}

void MatrixCOO::gemv(const Scalar* x, Scalar* y, Scalar alpha, Scalar beta) const
{
    // Scale y by beta
    if (beta == Scalar{0}) {
        for (Index i = 0; i < m_rows; ++i) y[i] = Scalar{0};
    } else if (beta != Scalar{1}) {
        for (Index i = 0; i < m_rows; ++i) y[i] *= beta;
    }
    // Accumulate alpha * A * x
    for (Index k = 0; k < nnz(); ++k) {
        y[m_row[k]] += alpha * m_val[k] * x[m_col[k]];
    }
}

void MatrixCOO::extractDiagonal(std::vector<Scalar>& d) const
{
    Index n = std::min(m_rows, m_cols);
    d.assign(n, Scalar{0});
    for (Index k = 0; k < nnz(); ++k) {
        if (m_row[k] == m_col[k]) {
            d[m_row[k]] += m_val[k];
        }
    }
}

void MatrixCOO::axpySamePattern(Scalar alpha, const MatrixCOO& B)
{
    if (m_rows != B.m_rows || m_cols != B.m_cols)
        throw std::runtime_error("MatrixCOO::axpySamePattern: shape mismatch");
    if (m_row.size() != B.m_row.size() || m_col.size() != B.m_col.size())
        throw std::runtime_error("MatrixCOO::axpySamePattern: pattern size mismatch");
    for (Index k = 0; k < m_row.size(); ++k) {
        if (m_row[k] != B.m_row[k] || m_col[k] != B.m_col[k])
            throw std::runtime_error("MatrixCOO::axpySamePattern: pattern differs at element");
        m_val[k] += alpha * B.m_val[k];
    }
}

MatrixCOO MatrixCOO::transpose() const
{
    MatrixCOO T(m_cols, m_rows);
    T.reserve(nnz());
    for (Index k = 0; k < nnz(); ++k) {
        T.add(m_col[k], m_row[k], m_val[k]);
    }
    return T;
}

}} // namespace dd::algebra