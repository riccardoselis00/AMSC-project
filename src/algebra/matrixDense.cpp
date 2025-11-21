#include "algebra/matrixDense.hpp"
#include <sstream>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <cctype>
#include <limits>


MatrixDense::MatrixDense(Index rows, Index cols, Scalar init_value)
    : m_rows(rows), m_cols(cols), m_data(rows * cols, init_value)
{
}

MatrixDense::MatrixDense(std::initializer_list<std::initializer_list<Scalar>> rows_il)
{
    m_rows = rows_il.size();
    m_cols = m_rows ? rows_il.begin()->size() : 0;
    m_data.resize(m_rows * m_cols);
    Index i = 0;
    for (const auto& row : rows_il) {
        if (row.size() != m_cols)
            throw std::runtime_error("MatrixDense: ragged initializer_list");
        Index j = 0;
        for (Scalar v : row) {
            m_data[i * m_cols + j] = v;
            ++j;
        }
        ++i;
    }
}

MatrixDense MatrixDense::Identity(Index n)
{
    MatrixDense I(n, n, 0.0);
    for (Index i = 0; i < n; ++i) I(i, i) = 1.0;
    return I;
}

MatrixDense::Scalar& MatrixDense::at(Index i, Index j)
{
    if (i >= m_rows || j >= m_cols)
        throw std::out_of_range("MatrixDense::at out of range");
    return m_data[i * m_cols + j];
}

const MatrixDense::Scalar& MatrixDense::at(Index i, Index j) const
{
    if (i >= m_rows || j >= m_cols)
        throw std::out_of_range("MatrixDense::at out of range");
    return m_data[i * m_cols + j];
}

void MatrixDense::fill(Scalar v) noexcept
{
    std::fill(m_data.begin(), m_data.end(), v);
}

void MatrixDense::setIdentity()
{
    if (m_rows != m_cols)
        throw std::runtime_error("MatrixDense::setIdentity requires square matrix");
    std::fill(m_data.begin(), m_data.end(), Scalar{0});
    for (Index i = 0; i < m_rows; ++i)
        m_data[i * m_cols + i] = Scalar{1};
}

void MatrixDense::resize(Index rows, Index cols, bool keep, Scalar pad_value)
{
    if (!keep) {
        m_rows = rows;
        m_cols = cols;
        m_data.assign(rows * cols, pad_value);
        return;
    }
    std::vector<Scalar> new_data(rows * cols, pad_value);
    Index rmin = std::min(rows, m_rows);
    Index cmin = std::min(cols, m_cols);
    for (Index i = 0; i < rmin; ++i) {
        const Scalar* src = &m_data[i * m_cols];
        Scalar* dst = &new_data[i * cols];
        std::copy(src, src + cmin, dst);
    }
    m_rows = rows;
    m_cols = cols;
    m_data.swap(new_data);
}

void MatrixDense::gemv(const Scalar* x, Scalar* y, Scalar alpha, Scalar beta) const noexcept
{

    for (Index i = 0; i < m_rows; ++i) {
        const Scalar* row = &m_data[i * m_cols];
        Scalar sum = 0.0;
        for (Index j = 0; j < m_cols; ++j) {
            sum += row[j] * x[j];
        }
        y[i] = alpha * sum + beta * y[i];
    }
}

void MatrixDense::gemv(const std::vector<Scalar>& x, std::vector<Scalar>& y, Scalar alpha, Scalar beta) const
{
    if (x.size() != m_cols)
        throw std::runtime_error("gemv: x has wrong size");
    if (y.size() != m_rows)
        y.assign(m_rows, Scalar{0});
    gemv(x.data(), y.data(), alpha, beta);
}

std::vector<MatrixDense::Scalar> MatrixDense::gemv(const std::vector<Scalar>& x) const
{
    if (x.size() != m_cols)
        throw std::runtime_error("gemv: x has wrong size");
    std::vector<Scalar> y(m_rows);
    gemv(x.data(), y.data(), 1.0, 0.0);
    return y;
}

void MatrixDense::gemm(const MatrixDense& A, const MatrixDense& B, MatrixDense& C,
                       Scalar alpha, Scalar beta)
{
    if (A.m_cols != B.m_rows)
        throw std::runtime_error("gemm: inner dimensions mismatch");
    if (C.m_rows != A.m_rows || C.m_cols != B.m_cols)
        throw std::runtime_error("gemm: C has wrong shape");

    if (beta == Scalar{0}) {
        std::fill(C.m_data.begin(), C.m_data.end(), Scalar{0});
    } else if (beta != Scalar{1}) {
        for (auto& v : C.m_data)
            v *= beta;
    }

    for (Index i = 0; i < A.m_rows; ++i) {
        const Scalar* Ai = &A.m_data[i * A.m_cols];
        Scalar* Ci = &C.m_data[i * C.m_cols];
        for (Index k = 0; k < A.m_cols; ++k) {
            const Scalar aik = alpha * Ai[k];
            const Scalar* Bk = &B.m_data[k * B.m_cols];
            for (Index j = 0; j < B.m_cols; ++j) {
                Ci[j] += aik * Bk[j];
            }
        }
    }
}

void MatrixDense::scale(Scalar alpha) noexcept
{
    for (auto& v : m_data)
        v *= alpha;
}

void MatrixDense::axpy(Scalar alpha, const MatrixDense& B)
{
    if (m_rows != B.m_rows || m_cols != B.m_cols)
        throw std::runtime_error("axpy: shape mismatch");
    for (Index i = 0; i < m_data.size(); ++i) {
        m_data[i] += alpha * B.m_data[i];
    }
}

MatrixDense MatrixDense::transpose() const
{
    MatrixDense T(m_cols, m_rows);
    for (Index i = 0; i < m_rows; ++i) {
        for (Index j = 0; j < m_cols; ++j) {
            T(j, i) = (*this)(i, j);
        }
    }
    return T;
}

void MatrixDense::transposeInPlace()
{
    if (m_rows != m_cols)
        throw std::runtime_error("transposeInPlace: requires square matrix");
    for (Index i = 0; i < m_rows; ++i) {
        for (Index j = i + 1; j < m_cols; ++j) {
            std::swap((*this)(i, j), (*this)(j, i));
        }
    }
}

MatrixDense::Scalar MatrixDense::frobeniusNorm() const noexcept
{
    long double sum = 0.0L;
    for (const auto& v : m_data) {
        sum += static_cast<long double>(v) * v;
    }
    return static_cast<Scalar>(std::sqrt(sum));
}

MatrixDense::Scalar MatrixDense::maxAbs() const noexcept
{
    Scalar m = 0;
    for (const auto& v : m_data) {
        Scalar absv = v >= Scalar{0} ? v : -v;
        if (absv > m)
            m = absv;
    }
    return m;
}

void MatrixDense::extractDiagonal(std::vector<Scalar>& d) const
{
    Index n = std::min(m_rows, m_cols);
    d.resize(n);
    for (Index i = 0; i < n; ++i) {
        d[i] = (*this)(i, i);
    }
}

std::string MatrixDense::toString(Index max_rows, Index max_cols) const
{
    std::ostringstream oss;
    const Index r = std::min(max_rows, m_rows);
    const Index c = std::min(max_cols, m_cols);
    oss << "MatrixDense(" << m_rows << "x" << m_cols << ")\n";
    for (Index i = 0; i < r; ++i) {
        oss << "  [";
        for (Index j = 0; j < c; ++j) {
            oss << (*this)(i, j);
            if (j + 1 < c)
                oss << ", ";
        }
        if (c < m_cols)
            oss << ", ...";
        oss << "]\n";
    }
    if (r < m_rows)
        oss << "  ...\n";
    return oss.str();
}

MatrixDense MatrixDense::block(Index i0, Index j0, Index r, Index c) const
{
    if (i0 + r > m_rows || j0 + c > m_cols)
        throw std::runtime_error("block: out of range");
    MatrixDense B(r, c);
    for (Index i = 0; i < r; ++i) {
        const Scalar* src = &m_data[(i0 + i) * m_cols + j0];
        Scalar* dst = &B.m_data[i * c];
        std::copy(src, src + c, dst);
    }
    return B;
}


static std::string tolower_copy(std::string s) {
    for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

static std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}


MatrixDense MatrixDense::from_mm_array_file(const std::string& filename)
{
    std::ifstream ifs(filename);
    if (!ifs) throw std::runtime_error("from_mm_array_file: cannot open '" + filename + "'");
    return from_mm_array_stream(ifs);
}

MatrixDense MatrixDense::from_mm_array_stream(std::istream& is)
{
    std::string line;
    if (!std::getline(is, line))
        throw std::runtime_error("from_mm_array_stream: empty input");
    auto header = trim(line);
    auto header_lc = tolower_copy(header);
    if (header_lc.rfind("%%matrixmarket", 0) != 0)
        throw std::runtime_error("from_mm_array_stream: missing '%%MatrixMarket' header");

    std::istringstream hs(header);
    std::string tag, kind, fmt, field, sym;
    hs >> tag >> kind >> fmt >> field >> sym;
    kind = tolower_copy(kind);
    fmt  = tolower_copy(fmt);
    field= tolower_copy(field);
    sym  = tolower_copy(sym);
    if (kind != "matrix" || fmt != "array")
        throw std::runtime_error("from_mm_array_stream: not a Matrix Market 'array' matrix");
    if (field != "real" && field != "integer")
        throw std::runtime_error("from_mm_array_stream: only 'real' or 'integer' supported for double");
    if (sym != "general" && sym != "symmetric" && sym != "skew-symmetric" && sym != "skewsymmetric")
        throw std::runtime_error("from_mm_array_stream: unsupported symmetry (hermitian/complex not supported)");
    if (sym == "skewsymmetric") sym = "skew-symmetric";

    while (std::getline(is, line)) {
        if (!line.empty() && line[0] != '%') break;
    }
    if (!is) throw std::runtime_error("from_mm_array_stream: missing size line");

    std::istringstream ss(line);
    Index M = 0, N = 0;
    ss >> M >> N;
    if (M <= 0 || N <= 0)
        throw std::runtime_error("from_mm_array_stream: invalid matrix size");
    MatrixDense A(M, N);

    auto read_scalar = [&](double& v)->void {
        if (!(is >> v)) throw std::runtime_error("from_mm_array_stream: not enough numeric data");
    };

    if (sym == "general") {

        const std::size_t total = static_cast<std::size_t>(M) * static_cast<std::size_t>(N);
        std::vector<double> buf(total);
        for (std::size_t k = 0; k < total; ++k) read_scalar(buf[k]);
        for (Index j = 0; j < N; ++j) {
            for (Index i = 0; i < M; ++i) {
                A(i, j) = buf[static_cast<std::size_t>(j) * M + i];
            }
        }
        return A;
    } else if (sym == "symmetric" || sym == "skew-symmetric") {
        if (M != N) throw std::runtime_error("from_mm_array_stream: symmetry requires square matrix");

        const bool skew = (sym == "skew-symmetric");
        const std::size_t T = skew
            ? static_cast<std::size_t>(M) * static_cast<std::size_t>(M - 1) / 2
            : static_cast<std::size_t>(M) * static_cast<std::size_t>(M + 1) / 2;
        std::vector<double> tri(T);
        for (std::size_t k = 0; k < T; ++k) read_scalar(tri[k]);

        std::size_t k = 0;
        for (Index j = 0; j < N; ++j) {
            for (Index i = j; i < M; ++i) {
                if (skew && i == j) continue; 
                const double v = tri[k++];
                A(i, j) = v;
                if (i == j) {
                    A(i, i) = skew ? 0.0 : v;
                } else {
                    A(j, i) = skew ? -v : v;
                }
            }
        }
        return A;
    } else {
        throw std::runtime_error("from_mm_array_stream: unsupported symmetry");
    }
}

bool MatrixDense::to_mm_array_file(const std::string& filename, const std::string& symmetry) const
{
    std::ofstream ofs(filename);
    if (!ofs) return false;
    return to_mm_array_stream(ofs, symmetry);
}

bool MatrixDense::to_mm_array_stream(std::ostream& os, const std::string& symmetry) const
{
    std::string sym = tolower_copy(symmetry);
    if (sym == "skewsymmetric") sym = "skew-symmetric";
    if (sym != "general" && sym != "symmetric" && sym != "skew-symmetric")
        return false;

    const Index M = m_rows, N = m_cols;
    if ((sym == "symmetric" || sym == "skew-symmetric") && M != N)
        throw std::runtime_error("to_mm_array_stream: symmetry requires square matrix");

    os << "%%MatrixMarket matrix array real " << sym << "\n";
    os << M << " " << N << "\n";
    os.setf(std::ios::scientific);
    os.precision(17);

    if (sym == "general") {
        for (Index j = 0; j < N; ++j)
            for (Index i = 0; i < M; ++i)
                os << (*this)(i, j) << "\n";
        return static_cast<bool>(os);
    }

    const bool skew = (sym == "skew-symmetric");
    for (Index j = 0; j < N; ++j) {
        for (Index i = j; i < M; ++i) {
            if (skew && i == j) continue; 
            os << (*this)(i, j) << "\n";
        }
    }
    return static_cast<bool>(os);
}