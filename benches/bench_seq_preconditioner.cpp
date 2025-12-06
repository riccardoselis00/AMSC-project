#include <iostream>
#include <vector>
#include <string>
#include <cstdio>
#include <cstdlib>

#include "algebra/COO.hpp"
#include "algebra/CSR.hpp"
#include "solver/pcg.hpp"
#include "preconditioner/identity.hpp"
#include "preconditioner/block_jacobi.hpp"
#include "preconditioner/additive_schwarz.hpp"
#include "preconditioner/diagonal_jacobi.hpp"
#include "utils/arg_parser.hpp"
#include "utils/timing.hpp"

int main(int argc, char** argv)
{
    SolverConfig cfg = SolverConfig::from_cli(argc, argv);

    Registry reg;
    int its = 0;

    std::cout << "Creating Poisson matrix of size " << cfg.n << std::endl;

    MatrixCOO A = MatrixCOO::Poisson2D(cfg.n);
    printf("Matrix created: %zu x %zu, nnz=%zu\n", A.rows(), A.cols(), A.nnz());


    std::vector<double> b(A.rows(), 0.0);
    std::vector<double> x(A.cols(), 0.0);

    for (size_t i = 0; i < b.size(); ++i)
        b[i] = (i & 1) ? -1.0 : 1.0;


    Preconditioner* M = nullptr;

    if (cfg.prec == "identity") {
        std::cout << "Using Identity Preconditioner\n";
        M = new IdentityPreconditioner();
    }
    else if (cfg.prec == "blockjac") {
        std::cout << "Using BlockJacobi with blocks = " << cfg.block_size << "\n";
        M = new BlockJacobi(cfg.block_size);
    }
    else if (cfg.prec == "diag_jacobi") {
         std::cout << "Using Diagonal Jacobi Preconditioner\n";
         M = new DiagonalJacobi(cfg.block_size);
    }
    else if (cfg.prec == "as") {
        std::cout << "Using Additive Schwarz (1 level)\n";
        M = new AdditiveSchwarz(cfg.block_size, cfg.overlap);
    }
    else {
        std::cerr << "Unknown preconditioner: " << cfg.prec << std::endl;
        return EXIT_FAILURE;
    }

    double time_setup = 0.0;

    if (cfg.prec != "identity") {
        auto t0 = std::chrono::high_resolution_clock::now();
        M->setup(A);
        auto t1 = std::chrono::high_resolution_clock::now();
        time_setup = std::chrono::duration<double>(t1 - t0).count();
    }


    PCGSolver solver(A, M);
    
    solver.setMaxIters(cfg.max_it);
    solver.setTolerance(cfg.tol);

    auto t0 = std::chrono::high_resolution_clock::now();
    its = solver.solve(b, x);
    auto t1 = std::chrono::high_resolution_clock::now();
    double time_solve = std::chrono::duration<double>(t1 - t0).count();

    double total_time = time_setup + time_solve;
        
    // reg.print_table();
    // reg.to_csv("solve_perf.csv");

    std::cout << "Solver finished in " << its << " iterations.\n";std::cout 
          << cfg.n << ","
          << cfg.prec << ","
          << its << ","
          << solver.lastRelResidual() << ","  
          << time_setup << ","
          << time_solve << ","
          << total_time           
          << std::endl;

    delete M;
    return 0;
}
