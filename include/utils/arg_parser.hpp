#pragma once
#include <string>
#include <cstdlib>
#include <iostream>

struct SolverConfig
{
    // ------------------
    // Public parameters
    // ------------------
    int    n           = 100000;
    double tol         = 1e-8;
    int    max_it      = 5000;

    std::string prec   = "identity";

    int block_size     = 4;     // Block Jacobi
    int overlap        = 1;     // Additive Schwarz
    int coarse_dim     = 0;     // Coarse operator size

    // ------------------
    // Parse CLI (header-only)
    // ------------------
    static SolverConfig from_cli(int argc, char** argv)
    {
        SolverConfig cfg;

        for (int i = 1; i < argc; ++i) {
            std::string a(argv[i]);

            if (a == "--n" && i + 1 < argc) {
                cfg.n = std::atoi(argv[++i]);
            }
            else if (a == "--prec" && i + 1 < argc) {
                cfg.prec = argv[++i];
            }
            else if (a == "--tol" && i + 1 < argc) {
                cfg.tol = std::atof(argv[++i]);
            }
            else if (a == "--maxit" && i + 1 < argc) {
                cfg.max_it = std::atoi(argv[++i]);
            }
            else if (a == "--block-size" && i + 1 < argc) {
                cfg.block_size = std::atoi(argv[++i]);
            }
            else if (a == "--overlap" && i + 1 < argc) {
                cfg.overlap = std::atoi(argv[++i]);
            }
            else if (a == "--coarse-dim" && i + 1 < argc) {
                cfg.coarse_dim = std::atoi(argv[++i]);
            }
            else if (a == "--help") {
                std::cout << "Available parameters:\n"
                          << "  --n N\n"
                          << "  --prec [none|jacobi|blockjac|as|as_coarse]\n"
                          << "  --tol TOL\n"
                          << "  --maxit MAX_IT\n"
                          << "  --block-size B\n"
                          << "  --overlap O\n"
                          << "  --coarse-dim C\n";
                std::exit(0);
            }
        }

        return cfg;
    }
};
