#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>   // std::atoi

#include "algebra/COO.hpp"
#include "preconditioner/additive_schwarz.hpp"
#include "solver/pcg_mpi.hpp"
#include "partitioner/partitioner.hpp"
#include "utils/timing.hpp"

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    int rank = 0, size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    using MatrixSparse = ::MatrixSparse;
    using Index        = MatrixSparse::Index;
    using Scalar       = MatrixSparse::Scalar;

    // -------------------------------------------------------------------------
    // Global problem size
    // -------------------------------------------------------------------------
    const int n_global = (argc > 1) ? std::atoi(argv[1]) : 20000;

    if (rank == 0) {
        std::cout << "=== testMPI: PCG + AdditiveSchwarz (MPI + coarse) ===\n";
        std::cout << "Create The Matrix! (n_global = "
                  << n_global << ", np = " << size << ")\n";
    }

    // -------------------------------------------------------------------------
    // Build global matrix and RHS (same on all ranks for now)
    //   A_global ~ Poisson-like operator
    //   b_global = (+1, -1, +1, -1, ...)
    // -------------------------------------------------------------------------
    MatrixCOO A_global = MatrixCOO::Poisson2D(static_cast<Index>(n_global));

    std::vector<double> b_global(static_cast<std::size_t>(n_global));
    for (int i = 0; i < n_global; ++i) {
        b_global[static_cast<std::size_t>(i)] = (i & 1) ? -1.0 : 1.0;
    }

    if (rank == 0) {
        std::cout << "Matrix created: " << n_global << " x " << n_global
                  << ", nnz = " << A_global.nnz() << "\n";
        std::cout << "Setting the RHS!\n";
    }

    // -------------------------------------------------------------------------
    // Partition global rows [0, n_global) among MPI ranks
    // -------------------------------------------------------------------------
    BlockRowPartitioner part(n_global, comm);
    const int ls    = part.ls();      // first global row on this rank
    const int le    = part.le();      // one-past-last global row
    const int n_loc = part.nLocal();  // local number of rows

    std::cout << "[rank " << rank << "] local range [ls, le) = ["
              << ls << ", " << le << "), n_loc = " << n_loc << "\n";

    // -------------------------------------------------------------------------
    // Build local system A_loc, b_loc by restricting A_global to [ls, le)
    // -------------------------------------------------------------------------
    MatrixCOO A_loc(static_cast<Index>(n_loc),
                    static_cast<Index>(n_loc));

    // Poisson → few nonzeros per row; reserve some space
    A_loc.reserve(5u * static_cast<std::size_t>(n_loc));

    A_global.forEachNZ([&](Index i, Index j, Scalar v) {
        const int gi = static_cast<int>(i);
        const int gj = static_cast<int>(j);
        if (gi >= ls && gi < le &&
            gj >= ls && gj < le)
        {
            const int li = gi - ls;
            const int lj = gj - ls;
            A_loc.add(static_cast<Index>(li),
                      static_cast<Index>(lj),
                      v);
        }
    });

    std::vector<double> b_loc;
    part.extractLocalVector(b_global, b_loc);

    if ((int)b_loc.size() != n_loc) {
        throw std::runtime_error("testMPI: b_loc size != n_loc");
    }

    if (rank == 0) {
        std::cout << "Local matrix assembled: n_loc = "
                  << n_loc << ", nnz_loc = " << A_loc.nnz() << "\n";
    }

    // -------------------------------------------------------------------------
    // Local Additive Schwarz preconditioner (MPI + coarse) & MPI PCG
    // -------------------------------------------------------------------------
    Registry reg;   // timing registry

    const int nparts  = 8;   // number of subdomains per rank
    const int overlap = 1;   // overlap in local DOFs

    AdditiveSchwarz::Level level = AdditiveSchwarz::Level::OneLevel;

    if (rank == 0) {
        std::cout << "Setting up Additive Schwarz (nparts=" << nparts
                  << ", overlap=" << overlap
                  << ", level=" << (level == AdditiveSchwarz::Level::TwoLevels
                                     ? "TwoLevels" : "OneLevel")
                  << ")...\n";
    }

    // MPI-aware + coarse AS
    AdditiveSchwarz M_loc(n_global,  // total DOFs
                          ls,        // first global row on this rank
                          le,        // one-past-last global row
                          nparts,
                          overlap,
                          comm,
                          level);

    M_loc.setSSORSweeps(1);
    M_loc.setOmega(1.95);

    {
        DD_TIMED_SCOPE_X("setup AS preconditioner", reg,
                         /*bytes=*/0, /*iters=*/0, "note");
        // A_loc is MatrixCOO, but AdditiveSchwarz::setup takes MatrixSparse const&
        // → OK via polymorphism (MatrixCOO derives from MatrixSparse)
        M_loc.setup(A_loc);
    }

    std::vector<double> x_loc(static_cast<std::size_t>(n_loc), 0.0);
    std::size_t its = 0;

    {
        DD_TIMED_SCOPE_X("solve", reg,
                         /*bytes=*/0, /*iters=*/0, "note");

        // MPI-aware PCG (works on local vectors; matrix passed by base ref)
        PCGSolverMPI solver(A_loc, &M_loc, comm, n_global, ls, le);
        solver.setMaxIters(500000);
        solver.setTolerance(1e-16);

        its = solver.solve(b_loc, x_loc);
    }

    if (rank == 0) {
        std::cout << "Solver finished in " << its
                  << " iterations.\n";
        reg.print_table();
    }

    // -------------------------------------------------------------------------
    // Optionally gather x_loc to root to inspect the global solution
    // -------------------------------------------------------------------------
    std::vector<double> x_global;
    part.gatherVectorToRoot(x_loc, x_global, /*root=*/0);
    if (rank == 0) {
        std::cout << "Gathered solution size on root = "
                  << x_global.size() << "\n";

        std::cout << "First 10 entries of x_global:\n";
        for (int i = 0; i < std::min(10, n_global); ++i) {
            std::cout << "x[" << i << "] = "
                      << x_global[static_cast<std::size_t>(i)] << "\n";
        }
    }

    MPI_Finalize();
    return 0;
}
