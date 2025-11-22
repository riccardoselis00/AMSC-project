
#pragma once
#include "solver.hpp"

class PCGSolver final : public Solver {
public:
    using Solver::Solver; 
    using Scalar = Solver::Scalar;

    std::size_t solve(const std::vector<Scalar>& b,
                      std::vector<Scalar>& x) override;

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
