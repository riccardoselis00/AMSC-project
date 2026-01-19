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
    // PCGSolverMPI(const MatrixSparse&    A_loc,
    //              const Preconditioner*  M,
    //              MPI_Comm               comm,
    //              Index                  n_global,
    //              Index                  ls,
    //              Index                  le);

    PCGSolverMPI(const MatrixSparse&  A_loc,
             Preconditioner*      M,
             MPI_Comm             comm,
             Index                n_global,
             Index                ls,
             Index                le);

    // Solve A x = b (distributed).
    //
    //  - b_loc, x_loc are local vectors of length n_loc = le - ls
    //  - returns number of iterations performed
    std::size_t solve(const std::vector<double>& b_loc,
                      std::vector<double>&       x_loc) override;

    // Accessors for convergence info
    std::size_t iterations() const noexcept           { return its_; }
    Scalar      lastResidualNorm() const noexcept     { return last_rnorm_; }
    Scalar      lastRelativeResidual() const noexcept { return last_rel_; }
    bool        converged() const noexcept            { return converged_; }

    // MPI info
    MPI_Comm comm() const noexcept { return comm_; }
    int      rank() const noexcept { return rank_; }
    int      size() const noexcept { return size_; }

private:
    // Local matrix and preconditioner
    const MatrixSparse&   A_;
    const Preconditioner* M_{nullptr};

    // MPI (declare BEFORE geometry to avoid -Wreorder depending on ctor init list)
    MPI_Comm comm_{MPI_COMM_WORLD};
    int      rank_{0};
    int      size_{1};

    // Global / local geometry (use Index to avoid -Wsign-compare with A_.rows())
    Index n_global_{0};   // total number of DOFs
    Index ls_{0};         // first global row owned by this rank
    Index le_{0};         // one-past-last global row owned by this rank
    Index n_loc_{0};      // local number of rows = le_ - ls_

    // Convergence info
    std::size_t its_{0};
    Scalar      last_rnorm_{0};
    Scalar      last_rel_{1};
    bool        converged_{false};
};
