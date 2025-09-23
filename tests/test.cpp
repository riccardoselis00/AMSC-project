#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <stdexcept>
#include <sstream>   // for std::stringstream
#include <cstdio>    // for std::remove
#include <cstdlib>   // for std::rand
#include <stdlib.h>

#include "dd/algebra/COO.hpp"
#include "dd/algebra/CSR.hpp"
#include "dd/solver/pcg.hpp"
#include "dd/preconditioner/preconditioner.hpp"
#include "dd/preconditioner/identity.hpp"
#include "dd/preconditioner/block_jacobi.hpp"
#include "timing.hpp"
#include "../tests/test_util.hpp"

using dd::algebra::MatrixCOO;
using dd::algebra::MatrixCSR;
using dd::algebra::PCGSolver;
using dd::algebra::IdentityPreconditioner;           // REMOVED (no longer used)
using dd::algebra::BlockJacobi;                         // NEW

int main()
{

    dd::util::Registry reg;
    std::cout << "Create The Matrix!" << std::endl;   

    MatrixCOO A = MatrixCOO::Poisson2D(50000);

    std::vector<double> b(A.rows(), 1.0);

    printf("Matrix created: %zu x %zu, nnz=%zu\n", A.rows(), A.cols(), A.nnz());

    std::cout << "Write the Matrix!" << std::endl;

    if (!A.write_COO("out.mtx")) {
        std::fprintf(stderr, "Failed to write out.mtx\n");
        return 1;
    }

    std::cout << "Solve The Linear System!" << std::endl;
    
    std::vector<double> x(A.cols(), 0.0);
    //IdentityPreconditioner M;                        // CHANGED (replace with BlockJacobi)
    
    BlockJacobi M(/*nparts=*/5);                        // NEW: choose a partition count (tune as you like)
    M.setup(A);                                         // NEW: build 1/diag(A) and partitions

    PCGSolver solver(A, &M);                            // (unchanged)
    solver.setMaxIters(50000);
    solver.setTolerance(1e-12);
    
    int its = 0;
    {
    // Optional metadata
    const std::string note = "solver=PCG; tol="  ;
                            // "; maxit=" + std::to_string(atol) +
                            // "; n=" + std::to_string(A.rows());
    
    DD_TIMED_SCOPE_X("solve", reg, /*bytes=*/0, /*iters=*/0, note);
    its = solver.solve(b, x);     // <-- the call you want to time
    } // record is pushed on scope exit

    reg.print_table();              // quick check in console
    reg.to_csv("solve_perf.csv");   //

    std::cout << "Solver finished in " << its << " iterations." << std::endl;

    std::cout << "Solution x entries" << std::endl;
    for (size_t i = 0; i < std::min((size_t)5, x.size()); ++i) {
        std::cout << "x[" << i << "] = " << x[i] << std::endl;
    }

    return 0;
}
