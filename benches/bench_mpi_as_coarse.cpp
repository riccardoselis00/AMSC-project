#include <mpi.h>

#include <iostream>
#include <vector>
#include <cstdlib>   // std::atoi
#include <chrono>

#include "algebra/COO.hpp"
#include "preconditioner/additive_schwarz.hpp"
#include "preconditioner/identity.hpp"
#include "solver/pcg_mpi.hpp"
#include "partitioner/partitioner.hpp"
#include "utils/arg_parser.hpp"

// NOTE:
// We assume SolverConfig (from arg_parser.hpp) has at least:
//   int    n;
//   std::string prec;      // "identity", "as", "as2", ...
//   int    block_size;     // here used as nparts (number of subdomains per rank)
//   int    overlap;
//   double tol;
//   int    max_it;
// and a static SolverConfig SolverConfig::from_cli(int argc, char** argv);

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    int rank = 0, nprocs = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    // Parse common CLI options (same style as sequential bench)
    SolverConfig cfg = SolverConfig::from_cli(argc, argv);
    const int n_global = cfg.n;

    if (rank == 0) {
        std::cout << "=== bench_pcg_mpi: PCG + MPI preconditioner ===\n";
        std::cout << "n_global = " << n_global
                  << ", np = " << nprocs
                  << ", prec = " << cfg.prec
                  << ", nparts (per rank) = " << cfg.block_size
                  << ", overlap = " << cfg.overlap
                  << ", tol = " << cfg.tol
                  << ", max_it = " << cfg.max_it
                  << "\n";
    }

    // -------------------------------------------------------------------------
    // Build global matrix and RHS (replicated on all ranks, OK for moderate n)
    // -------------------------------------------------------------------------
    MatrixCOO A_global = MatrixCOO::Poisson2D(static_cast<MatrixCOO::Index>(n_global));

    std::vector<double> b_global(static_cast<std::size_t>(n_global));
    for (int i = 0; i < n_global; ++i) {
        b_global[static_cast<std::size_t>(i)] = (i & 1) ? -1.0 : 1.0;
    }

    // -------------------------------------------------------------------------
    // Partition global rows [0, n_global) among MPI ranks
    // -------------------------------------------------------------------------
    BlockRowPartitioner part(n_global, comm);
    const int ls    = part.ls();      // first global row on this rank
    const int le    = part.le();      // one-past-last global row on this rank
    const int n_loc = part.nLocal();  // local number of rows

    // Build local matrix by restricting A_global
    MatrixCOO A_loc(static_cast<MatrixCOO::Index>(n_loc),
                    static_cast<MatrixCOO::Index>(n_loc));

    A_loc.reserve(5u * static_cast<std::size_t>(n_loc)); // Poisson-like

    A_global.forEachNZ([&](MatrixCOO::Index i, MatrixCOO::Index j, MatrixCOO::Scalar v) {
        int gi = static_cast<int>(i);
        int gj = static_cast<int>(j);
        if (gi >= ls && gi < le &&
            gj >= ls && gj < le)
        {
            const int li = gi - ls;
            const int lj = gj - ls;
            A_loc.add(static_cast<MatrixCOO::Index>(li),
                      static_cast<MatrixCOO::Index>(lj),
                      v);
        }
    });

    std::vector<double> b_loc;
    part.extractLocalVector(b_global, b_loc);
    if ((int)b_loc.size() != n_loc) {
        if (rank == 0) {
            std::cerr << "bench_pcg_mpi: b_loc size != n_loc\n";
        }
        MPI_Abort(comm, 1);
    }

    std::vector<double> x_loc(static_cast<std::size_t>(n_loc), 0.0);

    // -------------------------------------------------------------------------
    // Build preconditioner
    //   - "identity"  -> IdentityPreconditioner
    //   - "as"        -> AdditiveSchwarz (MPI, 1-level)
    //   - "as2"       -> AdditiveSchwarz (MPI, 2-level / coarse)
    // -------------------------------------------------------------------------
    Preconditioner* M = nullptr;
    AdditiveSchwarz::Level as_level = AdditiveSchwarz::Level::OneLevel;

    if (cfg.prec == "identity") {
        if (rank == 0) {
            std::cout << "Using Identity preconditioner (MPI-aware CG)\n";
        }
        M = new IdentityPreconditioner();
    }
    else if (cfg.prec == "as") {
        as_level = AdditiveSchwarz::Level::OneLevel;
        if (rank == 0) {
            std::cout << "Using MPI Additive Schwarz (1-level), nparts="
                      << cfg.block_size << ", overlap=" << cfg.overlap << "\n";
        }
        M = new AdditiveSchwarz(
            n_global,             // total DOFs
            ls,                   // first global row on this rank
            le,                   // one-past-last global row
            cfg.block_size,       // number of subdomains per rank
            cfg.overlap,          // overlap
            comm,                 // communicator
            as_level              // level (1-level)
        );
    }
    else if (cfg.prec == "as2") {
        as_level = AdditiveSchwarz::Level::TwoLevels;
        if (rank == 0) {
            std::cout << "Using MPI Additive Schwarz (2-level), nparts="
                      << cfg.block_size << ", overlap=" << cfg.overlap << "\n";
        }
        M = new AdditiveSchwarz(
            n_global,
            ls,
            le,
            cfg.block_size,
            cfg.overlap,
            comm,
            as_level
        );
    }
    else {
        if (rank == 0) {
            std::cerr << "bench_pcg_mpi: unknown --prec '" << cfg.prec
                      << "'. Supported: identity, as, as2.\n";
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // Optionally tweak smoother parameters for AS
    if (cfg.prec == "as" || cfg.prec == "as2") {
        auto* as_ptr = dynamic_cast<AdditiveSchwarz*>(M);
        if (as_ptr) {
            as_ptr->setSSORSweeps(1);
            as_ptr->setOmega(1.95);
        }
    }

    // -------------------------------------------------------------------------
    // Setup preconditioner (timed with MPI_Wtime, max over ranks)
    // -------------------------------------------------------------------------
    double time_setup_local = 0.0;
    double time_setup       = 0.0;

    if (cfg.prec != "identity") {
        MPI_Barrier(comm);
        double t0 = MPI_Wtime();
        M->setup(A_loc);
        MPI_Barrier(comm);
        double t1 = MPI_Wtime();
        time_setup_local = t1 - t0;
    }

    // Reduce to global max time (rank 0)
    MPI_Reduce(&time_setup_local, &time_setup,
               1, MPI_DOUBLE, MPI_MAX, 0, comm);

    // -------------------------------------------------------------------------
    // Solve with PCGSolverMPI (timed with MPI_Wtime, max over ranks)
    // -------------------------------------------------------------------------
    PCGSolverMPI solver(A_loc, M, comm,
                        n_global, ls, le);
    solver.setMaxIters(cfg.max_it);
    solver.setTolerance(cfg.tol);

    std::size_t its = 0;
    double time_solve_local = 0.0;
    double time_solve       = 0.0;

    MPI_Barrier(comm);
    double t0 = MPI_Wtime();
    its = solver.solve(b_loc, x_loc);
    MPI_Barrier(comm);
    double t1 = MPI_Wtime();
    time_solve_local = t1 - t0;

    MPI_Reduce(&time_solve_local, &time_solve,
               1, MPI_DOUBLE, MPI_MAX, 0, comm);

    double total_time = time_setup + time_solve;

    // -------------------------------------------------------------------------
    // CSV output on rank 0
    // Format:
    //   n,prec,nprocs,iters,residual,time_setup,time_solve,total_time
    // -------------------------------------------------------------------------
    if (rank == 0) {
        std::cout << "Solver finished in " << its
                  << " iterations.\n";

        std::cout.setf(std::ios::scientific);
        std::cout.precision(6);

        std::cout << n_global << ","
                  << cfg.prec << ","
                  << nprocs << ","
                  << its << ","
                //   << solver.lastRelResidual() << ","
                  << time_setup << ","
                  << time_solve << ","
                  << total_time
                  << std::endl;
    }

    delete M;
    MPI_Finalize();
    return 0;
}
