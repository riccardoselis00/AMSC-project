
#pragma once

#include <cstddef>
#include <vector>
#include "dd/algebra/matrixSparse.hpp"
#include "dd/preconditioner/preconditioner.hpp"


namespace dd { namespace algebra {

/**
 * @brief Abstract base class for iterative solvers of A x = b.
 *
 * A concrete solver implements @ref solve() (e.g., CG, GMRES).
 * The solver holds a non-owning reference to the system matrix A
 * and an optional non-owning pointer to a @ref Preconditioner.
 */
class Solver {
public:
    using Scalar = double;
    using Index  = std::size_t;

    Solver(const MatrixSparse& A, Preconditioner* M = nullptr)
    : A_(A), M_(M) {}

    virtual ~Solver();

    /// Set/get maximum iterations.
    void setMaxIters(std::size_t iters) noexcept { maxIters_ = iters; }
    std::size_t maxIters() const noexcept { return maxIters_; }

    /// Set/get relative residual tolerance (||r_k|| / ||r_0||).
    void setTolerance(Scalar tol) noexcept { tol_ = tol; }
    Scalar tolerance() const noexcept { return tol_; }

    /// Assign/replace the (non-owning) preconditioner pointer.
    void setPreconditioner(Preconditioner* M) noexcept { M_ = M; }
    Preconditioner* preconditioner() const noexcept { return M_; }

    /// @brief Solve A x = b starting from the current contents of x.
    /// @return Number of iterations performed (<= maxIters()).
    virtual std::size_t solve(const std::vector<Scalar>& b,
                              std::vector<Scalar>& x) = 0;

protected:
    const MatrixSparse& A_;
    Preconditioner*     M_{nullptr};
    std::size_t         maxIters_{1000};
    Scalar              tol_{1e-8};
};

}} // namespace dd::algebra
