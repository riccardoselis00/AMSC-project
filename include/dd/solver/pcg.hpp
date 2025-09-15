
#pragma once
#include "solver.hpp"

namespace dd { namespace algebra {

/**
 * @brief Preconditioned Conjugate Gradient (PCG) for SPD systems.
 *
 * Solves A x = b where A is symmetric positive definite.  Uses an
 * optional preconditioner M to accelerate convergence.  Convergence
 * test is ||r_k||_2 / ||r_0||_2 <= tol.
 */
class PCGSolver final : public Solver {
public:
    using Solver::Solver; // inherit constructors
    using Scalar = Solver::Scalar;

    std::size_t solve(const std::vector<Scalar>& b,
                      std::vector<Scalar>& x) override;

    // Last run diagnostics
    std::size_t iterations() const noexcept { return its_; }
    Scalar      lastResidualNorm() const noexcept { return last_rnorm_; }
    Scalar      lastRelResidual() const noexcept { return last_rel_; }
    bool        converged() const noexcept { return converged_; }

private:
    std::size_t its_{0};
    Scalar      last_rnorm_{0};
    Scalar      last_rel_{1};
    bool        converged_{false};
};

}} // namespace dd::algebra
