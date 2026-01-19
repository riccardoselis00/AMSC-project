#include "solver/pcg_mpi.hpp"

#include <cmath>
#include <stdexcept>
#include <iostream>

// -----------------------------------------------------------------------------
// Constructor
// -----------------------------------------------------------------------------
PCGSolverMPI::PCGSolverMPI(const MatrixSparse& A_loc,
                           Preconditioner*     M,
                           MPI_Comm            comm,
                           Index               n_global,
                           Index               ls,
                           Index               le)
  : Solver(A_loc, M)          // <-- base class has no default ctor
  , A_(A_loc)
  , M_(M)
  , comm_(comm)
  , rank_(0)
  , size_(1)
  , n_global_(n_global)
  , ls_(ls)
  , le_(le)
  , n_loc_(le - ls)
{
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &size_);

    // Optional safety (also avoids sign warnings now that Index is used):
    if (A_.rows() != n_loc_ || A_.cols() != n_loc_) {
        throw std::runtime_error("PCGSolverMPI: A_loc dimensions mismatch with n_loc");
    }
}

std::size_t PCGSolverMPI::solve(const std::vector<double>& b_loc,
                                std::vector<double>&       x_loc)
{
    const std::size_t n = static_cast<std::size_t>(n_loc_);

    if (b_loc.size() != n) {
        throw std::runtime_error("PCGSolverMPI::solve: b_loc has wrong size");
    }

    if (x_loc.size() != n) {
        x_loc.assign(n, 0.0);
    } else {
        std::fill(x_loc.begin(), x_loc.end(), 0.0);
    }

    // Local working vectors (size = n_loc_)
    std::vector<double> r_loc(n);
    std::vector<double> z_loc(n);
    std::vector<double> p_loc(n);
    std::vector<double> q_loc(n);

    // Helper lambdas for global norms / dot products
    auto global_dot = [this, n](const std::vector<double>& x,
                               const std::vector<double>& y) -> double {
        if (x.size() != n || y.size() != n) {
            throw std::runtime_error("PCGSolverMPI::global_dot: size mismatch");
        }
        double local = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            local += x[i] * y[i];
        }
        double global = 0.0;
        MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, comm_);
        return global;
    };

    auto global_norm2 = [this, n](const std::vector<double>& x) -> double {
        if (x.size() != n) {
            throw std::runtime_error("PCGSolverMPI::global_norm2: size mismatch");
        }
        double local = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const double v = x[i];
            local += v * v;
        }
        double global = 0.0;
        MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, comm_);
        return std::sqrt(global);
    };

    // -------------------- Initial residual r0 = b - A x0 --------------------
    // x0 is initially zero, so r0 = b
    r_loc = b_loc;

    // Compute norm of b for relative stopping criterion
    const double normb = global_norm2(b_loc);
    const double eps   = tolerance();           // from Solver base
    const double tiny  = 1e-30;

    // Precondition: z0 = M^{-1} r0
    if (M_) M_->apply(r_loc, z_loc);
    else    z_loc = r_loc;

    p_loc = z_loc;

    double rho    = global_dot(r_loc, z_loc);
    double relRes = global_norm2(r_loc) / (normb > 0.0 ? normb : 1.0);

    its_      = 0;
    last_rel_ = relRes;

    if (rank_ == 0) {
        std::cout << "PCGSolverMPI: initial relative residual = "
                  << relRes << std::endl;
    }

    if (relRes < eps) {
        if (rank_ == 0) {
            std::cout << "PCGSolverMPI: already converged.\n";
        }
        converged_ = true;
        last_rnorm_ = global_norm2(r_loc);
        return its_;
    }

    const std::size_t maxIts = static_cast<std::size_t>(maxIters()); // from Solver base

    for (its_ = 0; its_ < maxIts; ++its_) {

        // q = A p
        A_.gemv(p_loc, q_loc);   // local matvec

        const double denom = global_dot(p_loc, q_loc);
        if (std::abs(denom) < tiny) {
            if (rank_ == 0) {
                std::cerr << "PCGSolverMPI: breakdown (p^T A p ~ 0)\n";
            }
            break;
        }

        const double alpha = rho / denom;

        // x_{k+1} = x_k + alpha p_k
        for (std::size_t i = 0; i < n; ++i) {
            x_loc[i] += alpha * p_loc[i];
        }

        // r_{k+1} = r_k - alpha q_k
        for (std::size_t i = 0; i < n; ++i) {
            r_loc[i] -= alpha * q_loc[i];
        }

        // Check convergence
        relRes     = global_norm2(r_loc) / (normb > 0.0 ? normb : 1.0);
        last_rel_  = relRes;
        last_rnorm_ = global_norm2(r_loc);

        if (relRes < eps) {
            converged_ = true;
            break;
        }

        // z_{k+1} = M^{-1} r_{k+1}
        if (M_) M_->apply(r_loc, z_loc);
        else    z_loc = r_loc;

        const double rho_next = global_dot(r_loc, z_loc);
        if (std::abs(rho) < tiny) {
            if (rank_ == 0) {
                std::cerr << "PCGSolverMPI: breakdown (rho ~ 0)\n";
            }
            break;
        }

        const double beta = rho_next / rho;

        // p_{k+1} = z_{k+1} + beta p_k
        for (std::size_t i = 0; i < n; ++i) {
            p_loc[i] = z_loc[i] + beta * p_loc[i];
        }

        rho = rho_next;
    }

    if (rank_ == 0) {
        std::cout << "PCGSolverMPI: iterations = " << its_
                  << ", final relative residual = " << last_rel_
                  << ", tolerance = " << eps
                  << ", converged = " << (converged_ ? "yes" : "no")
                  << "\n";
    }

    return its_;
}
