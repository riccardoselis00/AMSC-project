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
        std::cout << "Create The Matrix! (n_global = "
                  << n_global << ", np = " << size << ")\n";
    }

    // -------------------------------------------------------------------------
    // Build global matrix and RHS (same on all ranks for now)
    //   A_global ~ 1D Poisson-like operator
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
    const int ls    = part.ls();
    const int le    = part.le();
    const int n_loc = part.nLocal();

    std::cout << "[rank " << rank << "] local range [ls, le) = ["
              << ls << ", " << le << "), n_loc = " << n_loc << "\n";

    // -------------------------------------------------------------------------
    // Build local system A_loc, b_loc
    //
    // Here we restrict A_global to rows/cols in [ls, le).
    // That gives a block-diagonal system (block-Jacobi style); for a *true*
    // distributed solve you would keep global column indices and implement
    // halo exchanges in gemv, but this is a clean first step to test:
    //   - partitioner
    //   - AdditiveSchwarz on local blocks
    //   - PCGSolverMPI structure
    // -------------------------------------------------------------------------
    MatrixCOO A_loc(static_cast<Index>(n_loc),
                    static_cast<Index>(n_loc));

    // 1D Poisson → at most 3 nonzeros per row
    A_loc.reserve(3u * static_cast<std::size_t>(n_loc));

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

    // -------------------------------------------------------------------------
    // Local Additive Schwarz preconditioner & MPI PCG
    // -------------------------------------------------------------------------
    
    Registry reg;   // timing registry

    if (rank == 0) {
        std::cout << "Setting up the Additive Schwarz preconditioner (local)...\n";
    }

    // You can tune nparts / overlap as you like.
    // [nparts] is number of subdomains *within this rank*,
    // [overlap] is the additive Schwarz overlap in DOFs.
    AdditiveSchwarz M_loc(8, 1);

    {
        DD_TIMED_SCOPE_X("setup AS preconditioner", reg,
                         /*bytes=*/0, /*iters=*/0, "note");
        M_loc.setup(A_loc);
    }

    std::vector<double> x_loc(static_cast<std::size_t>(n_loc), 0.0);
    std::size_t its = 0;

    {
        DD_TIMED_SCOPE_X("solve", reg,
                         /*bytes=*/0, /*iters=*/0, "note");

        PCGSolverMPI solver(A_loc, &M_loc, comm);
        solver.setMaxIters(500000);
        solver.setTolerance(1e-12);

        its = solver.solve(b_loc, x_loc);
    }

    if (rank == 0) {
        std::cout << "Solver finished in " << its
                  << " iterations (local block system).\n";
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
    }

    MPI_Finalize();
    return 0;
}
