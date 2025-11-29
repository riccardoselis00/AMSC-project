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
//   PCGSolverMPI solver(A_loc, &M_loc, comm);
//   solver.setMaxIters(5000);
//   solver.setTolerance(1e-8);
//   std::size_t it = solver.solve(b_loc, x_loc);
//
class PCGSolverMPI final : public Solver {
public:
    using Solver::Solver;   // inherit Solver(A, M) ctor
    using Scalar = Solver::Scalar;

    // Explicit constructor that also takes the MPI communicator.
    PCGSolverMPI(const MatrixSparse& A,
                 Preconditioner*     M,
                 MPI_Comm            comm);

    // Solve A x = b using (possibly) a local preconditioner M.
    // b and x are *local* vectors of size n_loc on each rank.
    std::size_t solve(const std::vector<Scalar>& b,
                      std::vector<Scalar>&       x) override;

    std::size_t iterations()       const noexcept { return its_; }
    Scalar      lastResidualNorm() const noexcept { return last_rnorm_; }
    Scalar      lastRelResidual()  const noexcept { return last_rel_; }
    bool        converged()        const noexcept { return converged_; }

    MPI_Comm    comm() const noexcept { return comm_; }
    int         rank() const noexcept { return rank_; }
    int         size() const noexcept { return size_; }

private:
    MPI_Comm    comm_{MPI_COMM_WORLD};
    int         rank_{0};
    int         size_{1};

    std::size_t its_{0};
    Scalar      last_rnorm_{0};
    Scalar      last_rel_{1};
    bool        converged_{false};
};
