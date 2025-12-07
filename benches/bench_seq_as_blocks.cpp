#include <iostream>
#include <vector>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <chrono>

#include "algebra/COO.hpp"
#include "algebra/CSR.hpp"
#include "solver/pcg.hpp"
#include "preconditioner/identity.hpp"
#include "preconditioner/additive_schwarz.hpp"
#include "utils/arg_parser.hpp"   // SolverConfig
#include "utils/timing.hpp"       // Registry if you need it later

int main(int argc, char** argv)
{
    // --------------------------------------------------
    // 1. Parse CLI options
    //    Expected: --n, --prec [identity|as], --block-size, --overlap, --tol, --maxit
    // --------------------------------------------------
    SolverConfig cfg = SolverConfig::from_cli(argc, argv);

    if (cfg.prec != "identity" && cfg.prec != "as") {
        std::cerr << "bench_as_blocks: only prec=identity or prec=as are supported.\n";
        return EXIT_FAILURE;
    }

    std::cout << "Creating Poisson matrix of size " << cfg.n << std::endl;

    // --------------------------------------------------
    // 2. Build matrix and RHS
    // --------------------------------------------------
    MatrixCOO A = MatrixCOO::Poisson2D(cfg.n);
    std::printf("Matrix created: %zu x %zu, nnz=%zu\n", A.rows(), A.cols(), A.nnz());

    std::vector<double> b(A.rows(), 0.0);
    std::vector<double> x(A.cols(), 0.0);

    // Alternating ±1 RHS
    for (std::size_t i = 0; i < b.size(); ++i) {
        b[i] = (i & 1) ? -1.0 : 1.0;
    }

    // --------------------------------------------------
    // 3. Select preconditioner (identity or AS)
    // --------------------------------------------------
    Preconditioner* M = nullptr;

    if (cfg.prec == "identity") {
        std::cout << "Using Identity Preconditioner\n";
        M = new IdentityPreconditioner();
    }
    else { // cfg.prec == "as"
        std::cout << "Using Additive Schwarz (1 level), blocks = "
                  << cfg.block_size << ", overlap = " << cfg.overlap << "\n";
        M = new AdditiveSchwarz(cfg.block_size, cfg.overlap);
    }

    // --------------------------------------------------
    // 4. Setup preconditioner (not needed for identity)
    // --------------------------------------------------
    double time_setup = 0.0;

    if (cfg.prec != "identity") {
        auto t0 = std::chrono::high_resolution_clock::now();
        M->setup(A);
        auto t1 = std::chrono::high_resolution_clock::now();
        time_setup = std::chrono::duration<double>(t1 - t0).count();
    }

    // --------------------------------------------------
    // 5. Solve with PCG
    // --------------------------------------------------
    PCGSolver solver(A, M);
    solver.setMaxIters(cfg.max_it);
    solver.setTolerance(cfg.tol);

    int its = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    its = solver.solve(b, x);
    auto t1 = std::chrono::high_resolution_clock::now();
    double time_solve = std::chrono::duration<double>(t1 - t0).count();

    double total_time = time_setup + time_solve;

    std::cout << "Solver finished in " << its << " iterations.\n";

    // --------------------------------------------------
    // 6. CSV line (compatible with your .sh logic)
    //
    // NOTE: no block_size in the line itself, because your
    //       run_seq_as_blocks.sh reconstructs:
    //       n,prec,block_size,iters,...
    //       by cutting and prepending BLOCK from the script.
    // --------------------------------------------------
    std::cout << cfg.n << ","
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
