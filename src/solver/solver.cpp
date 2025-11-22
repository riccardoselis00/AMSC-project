#include "solver/solver.hpp"

#include <stdio.h>
#include <stdlib.h>

#include <iostream>

Solver::~Solver() = default;


void Solver::printConvergenceInfo(std::size_t iters,
                                  Scalar final_res,
                                  Scalar tolerance) const
{
    std::cout << "Converged in " << iters
              << " iterations, final residual = " << final_res
              << ", tolerance = " << tolerance << "\n";
}

