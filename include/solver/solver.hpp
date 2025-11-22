#pragma once

#include <cstddef>
#include <vector>
#include "algebra/matrixSparse.hpp"
#include "preconditioner/preconditioner.hpp"

class Solver {
public:
    using Scalar = double;
    using Index  = std::size_t;

    Solver(const MatrixSparse& A, Preconditioner* M = nullptr)
    : A_(A), M_(M) {}

    virtual ~Solver();

    void setMaxIters(std::size_t iters) noexcept { maxIters_ = iters; }
    std::size_t maxIters() const noexcept { return maxIters_; }

    void setTolerance(Scalar tol) noexcept { tol_ = tol; }
    Scalar tolerance() const noexcept { return tol_; }

    void setPreconditioner(Preconditioner* M) noexcept { M_ = M; }
    Preconditioner* preconditioner() const noexcept { return M_; }

    virtual std::size_t solve(const std::vector<Scalar>& b,
                              std::vector<Scalar>& x) = 0;

    void printConvergenceInfo(std::size_t iters, Scalar final_res, Scalar tolerance) const;

protected:
    const MatrixSparse& A_;
    Preconditioner*     M_{nullptr};
    std::size_t         maxIters_{10000};
    Scalar              tol_{1e-8};
};
