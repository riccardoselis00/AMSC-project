#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <stdexcept>
#include <sstream>   // for std::stringstream
#include <cstdio>    // for std::remove
#include <cstdlib>   // for std::rand
#include <stdlib.h>

#include "algebra/COO.hpp"
#include "algebra/CSR.hpp"
#include "solver/pcg.hpp"
#include "preconditioner/preconditioner.hpp"
#include "preconditioner/identity.hpp"
#include "preconditioner/block_jacobi.hpp"
#include "utils/timing.hpp"
#include "utils/test_util.hpp"


int main()
{

    Registry reg;
    int its = 0;
    std::cout << "Create The Matrix!" << std::endl;   

    MatrixCOO A = MatrixCOO::Poisson2D(10000);

    printf("Matrix created: %zu x %zu, nnz=%zu\n", A.rows(), A.cols(), A.nnz());

    std::cout << "Setting the RHS!" << std::endl;
    std::vector<double> b(A.rows(), 0.0);
    std::vector<double> x(A.cols(), 0.0);

    // RHS
    //================================
    // for (size_t i = 0; i < b.size(); ++i)
    //     b[i] = 1.0;

    for (size_t i = 0; i < b.size(); ++i)
        b[i] = (i & 1) ? -1.0 : 1.0; 
    //================================

    std::cout << "Setting the preconditioner!" << std::endl;

    IdentityPreconditioner M;   

    // BlockJacobi M(8);                        // NEW: choose a partition count (tune as you like)
    // M.setup(A);                                         // NEW: build 1/diag(A) and partitions

    PCGSolver solver(A, &M);                            // (unchanged)
    solver.setMaxIters(500000);
    solver.setTolerance(1e-8);

    std::cout << "Solve The Linear System!" << std::endl;
    {
    // Optional metadata
    const std::string note = "solver=PCG; tol="  ;
    DD_TIMED_SCOPE_X("solve", reg, /*bytes=*/0, /*iters=*/0, note);
    its = solver.solve(b, x);     // <-- the call you want to time
    } // record is pushed on scope exit

    reg.print_table();              // quick check in console
    reg.to_csv("solve_perf.csv");   //

    std::cout << "Solver finished in " << its << " iterations." << std::endl;

    std::cout << "b entries" << std::endl;
    for (size_t i = 0; i < std::min((size_t)10, b.size()); ++i) {
        std::cout << "b[" << i << "] = " << b[i] << std::endl;
    }

    std::cout << "Solution x entries" << std::endl;
    for (size_t i = 0; i < std::min((size_t)10, x.size()); ++i) {
        std::cout << "x[" << i << "] = " << x[i] << std::endl;
    }

    return 0;
}
