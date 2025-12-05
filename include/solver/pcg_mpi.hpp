#pragma once

#include "solver.hpp"
#include <mpi.h>
#include <vector>
#include <algebra/matrixSparse.hpp>
#include <preconditioner/preconditioner.hpp>

// MPI-aware Preconditioned Conjugate Gradient solver.
//
//  - Works on *local* vectors b, x of length n_loc (rows owned by this rank).
//  - Uses the same MatrixSparse / Preconditioner / Solver base as PCGSolver.
//  - Uses MPI_Allreduce for global dot-products and norms.
//
// Typical usage (on each rank):
//
//   MatrixSparse        A_loc = ...;              // local rows [ls, le)
//   std::vector<double> b_loc(n_loc), x_loc(n_loc, 0.0);
//
//   AdditiveSchwarz M(n_global, ls, le,
//                     nparts, overlap, comm,
//                     AdditiveSchwarz::Level::TwoLevels);
//   M.setup(A_loc);
//
//   PCGSolverMPI solver(A_loc, &M, comm, n_global, ls, le);
//   solver.setTolerance(1e-8);
//   solver.setMaxIterations(5000);
//   solver.solve(b_loc, x_loc);
//
//   if (rank == 0) {
//       std::cout << "Iterations: " << solver.iterations() << "\n";
//   }

class PCGSolverMPI : public Solver {
public:
    using MatrixSparse = ::MatrixSparse;
    using Index        = MatrixSparse::Index;
    using Scalar       = MatrixSparse::Scalar;

    // Constructor
    //
    //  A_loc    : local matrix corresponding to global rows [ls, le)
    //  M        : (optional) preconditioner working on local vectors
    //  comm     : MPI communicator
    //  n_global : global number of DOFs
    //  ls, le   : global row range owned by this rank [ls, le)
    PCGSolverMPI(const MatrixSparse&   A_loc,
                 const Preconditioner* M,
                 MPI_Comm              comm,
                 int                   n_global,
                 int                   ls,
                 int                   le);

    // Solve A x = b (distributed).
    //
    //  - b_loc, x_loc are local vectors of length n_loc = le - ls
    //  - returns number of iterations performed
  std::size_t solve(const std::vector<double>& b_loc,
                  std::vector<double>&       x_loc) override;

    // Accessors for convergence info
    std::size_t iterations() const noexcept      { return its_; }
    Scalar      lastResidualNorm() const noexcept { return last_rnorm_; }
    Scalar      lastRelativeResidual() const noexcept { return last_rel_; }
    bool        converged() const noexcept       { return converged_; }

    // MPI info
    MPI_Comm    comm() const noexcept { return comm_; }
    int         rank() const noexcept { return rank_; }
    int         size() const noexcept { return size_; }

private:
    // Local matrix and preconditioner
    const MatrixSparse&   A_;
    const Preconditioner* M_{nullptr};

    // Global / local geometry
    int n_global_{0};   // total number of DOFs
    int ls_{0};         // first global row owned by this rank
    int le_{0};         // one-past-last global row owned by this rank
    int n_loc_{0};      // local number of rows = le_ - ls_

    // MPI
    MPI_Comm comm_{MPI_COMM_WORLD};
    int      rank_{0};
    int      size_{1};

    // Convergence info
    std::size_t its_{0};
    Scalar      last_rnorm_{0};
    Scalar      last_rel_{1};
    bool        converged_{false};
};
