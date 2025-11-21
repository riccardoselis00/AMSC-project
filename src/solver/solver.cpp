#include "solver/solver.hpp"

#include <stdio.h>
#include <stdlib.h>

#include <iostream>

Solver::~Solver() = default;


void Solver::printConvergenceInfo(std::size_t iters, Scalar final_res) const
{
    std::cout << "[Solver] Converged in " << iters << " iterations\n"
              << "          Final relative residual = " << final_res << "\n"
              << "          Tolerance used          = " << tol_ << "\n";
}
