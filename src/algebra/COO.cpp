#include "dd/algebra/COO.hpp"
#include "dd/algebra/CSR.hpp"

#include <algorithm>
#include <cmath>

#include <cctype>

#include <cctype>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

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

// static void lower_inplace(char* s) {
//     for (; *s; ++s) *s = static_cast<char>(std::tolower(static_cast<unsigned char>(*s)));
// }


void MatrixCOO::lower_inplace(char* s) {
    for (; *s; ++s) {
        *s = static_cast<char>(std::tolower(static_cast<unsigned char>(*s)));
    }
}



MatrixCOO MatrixCOO::read_COO(const std::string& filename)
{
    FILE* f = std::fopen(filename.c_str(), "r");
    if (!f) throw std::runtime_error("MatrixCOO::read_COO: cannot open '" + filename + "'");

    char buf[1 << 16];

    if (!std::fgets(buf, sizeof(buf), f))
        throw std::runtime_error("MatrixCOO::read_COO: empty file");

    char object[64], format[64], field[64], symmetry[64];
    if (std::sscanf(buf, "%%%%MatrixMarket %63s %63s %63s %63s",
                    object, format, field, symmetry) != 4) {
        throw std::runtime_error("MatrixCOO::read_COO: malformed header");
    }
    if (std::strncmp(buf, "%%MatrixMarket", 14) != 0)
        throw std::runtime_error("MatrixCOO::read_COO: missing %%MatrixMarket");

    lower_inplace(object); lower_inplace(format);
    lower_inplace(field);  lower_inplace(symmetry);

    if (std::strcmp(object, "matrix") != 0 || std::strcmp(format, "coordinate") != 0)
        throw std::runtime_error("MatrixCOO::read_COO: only 'matrix coordinate' supported");
    if (std::strcmp(field, "complex") == 0 || std::strcmp(symmetry, "hermitian") == 0)
        throw std::runtime_error("MatrixCOO::read_COO: complex/hermitian not supported");

    unsigned long m=0, n=0, nnz_hint=0;
    for (;;) {
        if (!std::fgets(buf, sizeof(buf), f))
            throw std::runtime_error("MatrixCOO::read_COO: missing size line");
        if (buf[0] == '%' || buf[0] == '\n' || buf[0] == '\r') continue;
        if (std::sscanf(buf, "%lu %lu %lu", &m, &n, &nnz_hint) == 3 && m>0 && n>0) break;
        throw std::runtime_error("MatrixCOO::read_COO: invalid size line");
    }

    MatrixCOO A(static_cast<std::size_t>(m), static_cast<std::size_t>(n));
    A.reserve(std::strcmp(symmetry, "general") == 0 ? nnz_hint : nnz_hint * 2);

    while (std::fgets(buf, sizeof(buf), f)) {
        if (buf[0] == '%' || buf[0] == '\n' || buf[0] == '\r') continue;

        char* p = buf; char* end = buf;
        unsigned long i1 = std::strtoul(p, &end, 10);
        if (end == p) continue;
        p = end;
        unsigned long j1 = std::strtoul(p, &end, 10);
        if (end == p) continue;
        p = end;

        double v = 1.0;
        if (std::strcmp(field, "pattern") != 0) {
            v = std::strtod(p, &end);
            if (end == p)
                throw std::runtime_error("MatrixCOO::read_COO: expected numeric value");
        }

        if (i1 == 0 || j1 == 0)
            throw std::runtime_error("MatrixCOO::read_COO: indices must be 1-based");
        const std::size_t i = static_cast<std::size_t>(i1 - 1);
        const std::size_t j = static_cast<std::size_t>(j1 - 1);

        A.add(i, j, v);
        if (i != j) {
            if (std::strcmp(symmetry, "symmetric") == 0)       A.add(j, i,  v);
            else if (std::strcmp(symmetry, "skew-symmetric") == 0) A.add(j, i, -v);
        }
    }

    std::fclose(f);
    return A;
}

bool MatrixCOO::write_COO(const std::string& filename) const
{
    FILE* f = std::fopen(filename.c_str(), "w");
    if (!f) return false;

    std::fprintf(f, "%%%MatrixMarket matrix coordinate real general\n");
    //std::fprintf(f, "%%generated by dd::algebra::MatrixCOO::write_COO\n");
    std::fprintf(f, "%zu %zu %zu\n",
                 static_cast<std::size_t>(rows()),
                 static_cast<std::size_t>(cols()),
                 static_cast<std::size_t>(nnz()));

    const std::size_t K = static_cast<std::size_t>(nnz());
    for (std::size_t k = 0; k < K; ++k) {
        const std::size_t i = static_cast<std::size_t>(m_row[k]) + 1;
        const std::size_t j = static_cast<std::size_t>(m_col[k]) + 1;
        const double      v = m_val[k];
        std::fprintf(f, "%zu %zu %.17g\n", i, j, v);
    }

    std::fclose(f);
    return true;
}

}} // namespace dd::algebra