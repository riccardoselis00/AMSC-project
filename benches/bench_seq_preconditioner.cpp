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
#include "utils/arg_parser.hpp"
#include "utils/timing.hpp"

int main(int argc, char** argv)
{
    // ============================
    // 1. Parse CLI
    // ============================
    SolverConfig cfg = SolverConfig::from_cli(argc, argv);

    Registry reg;
    int its = 0;

    std::cout << "Creating Poisson matrix of size " << cfg.n << std::endl;

    // ============================
    // 2. Build matrix (Poisson2D)
    // ============================
    MatrixCOO A = MatrixCOO::Poisson2D(cfg.n);
    printf("Matrix created: %zu x %zu, nnz=%zu\n", A.rows(), A.cols(), A.nnz());

    // Vectors
    std::vector<double> b(A.rows(), 0.0);
    std::vector<double> x(A.cols(), 0.0);

    // RHS pattern: alternating +1/-1
    for (size_t i = 0; i < b.size(); ++i)
        b[i] = (i & 1) ? -1.0 : 1.0;

    // ============================
    // 3. Select preconditioner
    // ============================

    Preconditioner* M = nullptr;

    if (cfg.prec == "identity") {
        std::cout << "Using Identity Preconditioner\n";
        M = new IdentityPreconditioner();
    } 
    else if (cfg.prec == "blockjac") {
        std::cout << "Using BlockJacobi with blocks=" << cfg.block_size << "\n";
        M = new BlockJacobi(cfg.block_size);

        { DD_TIMED_SCOPE_X("BlockJacobi_setup", reg, 0, 0, ""); 
          M->setup(A);
        }
    }
    else if (cfg.prec == "as") {
        std::cout << "Using Additive Schwarz (1 level)\n";
        M = new AdditiveSchwarz(cfg.block_size, cfg.overlap);

        { DD_TIMED_SCOPE_X("AS_setup", reg, 0, 0, ""); 
          M->setup(A);
        }
    }
    else if (cfg.prec == "as2") {
        std::cout << "Using Additive Schwarz (2 levels)\n";
        M = new AdditiveSchwarz(cfg.block_size, cfg.overlap,
                                AdditiveSchwarz::Level::TwoLevels);

        { DD_TIMED_SCOPE_X("AS2_setup", reg, 0, 0, ""); 
          M->setup(A);
        }
    }
    else {
        std::cerr << "Unknown --prec type: " << cfg.prec << "\n";
        return EXIT_FAILURE;
    }

    // ============================
    // 4. Solve with PCG
    // ============================

    PCGSolver solver(A, M);
    solver.setMaxIters(cfg.max_it);
    solver.setTolerance(cfg.tol);

    {
        DD_TIMED_SCOPE_X("solve", reg, 0, 0, "");
        its = solver.solve(b, x);
    }

    reg.print_table();
    reg.to_csv("solve_perf.csv");

    std::cout << "Solver finished in " << its << " iterations.\n";

    delete M;
    return 0;
}
