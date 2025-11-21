#include "algebra/matrixSparse.hpp"

#include <cmath>
#include <sstream>
#include <algorithm>


void MatrixSparse::gemv(const std::vector<Scalar>& x, std::vector<Scalar>& y,
                        Scalar alpha, Scalar beta) const
{
    if (x.size() != cols())
        throw std::runtime_error("MatrixSparse::gemv: x has wrong size");
    if (y.size() != rows())
        y.assign(rows(), Scalar{0});
    gemv(x.data(), y.data(), alpha, beta);
}

std::vector<MatrixSparse::Scalar>
MatrixSparse::gemv(const std::vector<Scalar>& x) const
{
    if (x.size() != cols())
        throw std::runtime_error("MatrixSparse::gemv: x has wrong size");
    std::vector<Scalar> y(rows());
    gemv(x.data(), y.data(), Scalar{1}, Scalar{0});
    return y;
}

MatrixSparse::Scalar MatrixSparse::frobeniusNorm() const
{
    long double sum = 0.0L;
    forEachNZ([&](Index, Index, Scalar v){ sum += static_cast<long double>(v) * v; });
    return static_cast<Scalar>(std::sqrt(sum));
}

MatrixSparse::Scalar MatrixSparse::maxAbs() const
{
    Scalar m = 0.0;
    forEachNZ([&](Index, Index, Scalar v){ m = std::max(m, v >= Scalar{0} ? v : -v); });
    return m;
}

std::string MatrixSparse::toString(std::size_t max_lines) const
{
    std::ostringstream oss;
    oss << "MatrixSparse(" << rows() << "x" << cols() << ", nnz=" << nnz() << ")\n";
    std::size_t printed = 0;
    forEachNZ([&](Index i, Index j, Scalar v){
        if (printed < max_lines) {
            oss << "  (" << i << "," << j << ") = " << v << "\n";
            ++printed;
        }
    });
    if (nnz() > printed) oss << "  ...\n";
    return oss.str();
}