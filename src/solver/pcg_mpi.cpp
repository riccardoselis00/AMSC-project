#include "solver/pcg_mpi.hpp"

#include <cmath>
#include <stdexcept>

// -----------------------------------------------------------------------------
// Global dot-product and 2-norm using MPI_Allreduce
// -----------------------------------------------------------------------------
// static inline double dot_global(const std::vector<double>& a,
//                                 const std::vector<double>& b,
//                                 MPI_Comm comm)
// {
//     if (a.size() != b.size())
//         throw std::runtime_error("dot_global: size mismatch");

//     double local = 0.0;
//     for (std::size_t i = 0; i < a.size(); ++i)
//         local += a[i] * b[i];

//     double global = 0.0;
//     MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, comm);
//     return global;
// }

// static inline double nrm2_global(const std::vector<double>& a,
//                                  MPI_Comm comm)
// {
//     double local = 0.0;
//     for (double v : a) local += v * v;

//     double global = 0.0;
//     MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, comm);
//     return std::sqrt(global);
// }

// -----------------------------------------------------------------------------
// Constructor
// -----------------------------------------------------------------------------

PCGSolverMPI::PCGSolverMPI(const MatrixSparse&      A_loc,
                           const Preconditioner*    M,
                           MPI_Comm                 comm,
                           int                      n_global,
                           int                      ls,
                           int                      le)
    : Solver(A_loc, const_cast<Preconditioner*>(M)),  // <-- FIX HERE
    A_(A_loc),
    M_(M),
    comm_(comm),
    n_global_(n_global),
    ls_(ls),
    le_(le),
    n_loc_(le - ls)
{
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &size_);

    if (A_.rows() != n_loc_ || A_.cols() != n_loc_) {
        throw std::runtime_error("PCGSolverMPI: local matrix size mismatch");
    }
}

std::size_t PCGSolverMPI::solve(const std::vector<double>& b_loc,
                                std::vector<double>&       x_loc)
{
    if ((int)b_loc.size() != n_loc_) {
        throw std::runtime_error("PCGSolverMPI::solve: b_loc has wrong size");
    }

    x_loc.resize(static_cast<std::size_t>(n_loc_), 0.0);

    // Local working vectors (size = n_loc_)
    std::vector<double> r_loc(n_loc_);
    std::vector<double> z_loc(n_loc_);
    std::vector<double> p_loc(n_loc_);
    std::vector<double> q_loc(n_loc_);

    // Helper lambdas for global norms / dot products
    auto global_dot = [this](const std::vector<double>& x,
                             const std::vector<double>& y) -> double {
        double local = 0.0;
        for (int i = 0; i < n_loc_; ++i) {
            local += x[static_cast<std::size_t>(i)] *
                     y[static_cast<std::size_t>(i)];
        }
        double global = 0.0;
        MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, comm_);
        return global;
    };

    auto global_norm2 = [&](const std::vector<double>& x) -> double {
        double local = 0.0;
        for (int i = 0; i < n_loc_; ++i) {
            double v = x[static_cast<std::size_t>(i)];
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
        return its_;
    }

    const int maxIts = maxIters();         // from Solver base

    for (its_ = 0; its_ < maxIts; ++its_) {

        // q = A p
        A_.gemv(p_loc, q_loc);   // A_loc * p_loc

        double denom = global_dot(p_loc, q_loc);
        if (std::abs(denom) < tiny) {
            if (rank_ == 0) {
                std::cerr << "PCGSolverMPI: breakdown (p^T A p ~ 0)\n";
            }
            break;
        }

        double alpha = rho / denom;

        // x_{k+1} = x_k + alpha p_k
        for (int i = 0; i < n_loc_; ++i) {
            x_loc[static_cast<std::size_t>(i)] += alpha *
                                                   p_loc[static_cast<std::size_t>(i)];
        }

        // r_{k+1} = r_k - alpha q_k
        for (int i = 0; i < n_loc_; ++i) {
            r_loc[static_cast<std::size_t>(i)] -= alpha *
                                                   q_loc[static_cast<std::size_t>(i)];
        }

        // Check convergence
        relRes   = global_norm2(r_loc) / (normb > 0.0 ? normb : 1.0);
        last_rel_ = relRes;

        if (rank_ == 0) {
            // Optional: debug prints
            // std::cout << "Iter " << its_+1 << ", relRes = " << relRes << "\n";
        }

        if (relRes < eps) {
            break;
        }

        // z_{k+1} = M^{-1} r_{k+1}
        if (M_) M_->apply(r_loc, z_loc);
        else    z_loc = r_loc;

        double rho_next = global_dot(r_loc, z_loc);
        if (std::abs(rho) < tiny) {
            if (rank_ == 0) {
                std::cerr << "PCGSolverMPI: breakdown (rho ~ 0)\n";
            }
            break;
        }

        double beta = rho_next / rho;

        // p_{k+1} = z_{k+1} + beta p_k
        for (int i = 0; i < n_loc_; ++i) {
            p_loc[static_cast<std::size_t>(i)] =
                z_loc[static_cast<std::size_t>(i)] +
                beta * p_loc[static_cast<std::size_t>(i)];
        }

        rho = rho_next;
    }

    if (rank_ == 0) {
        std::cout << "Converged in " << its_
                  << " iterations, final relative residual = "
                  << last_rel_ << ", tolerance = " << eps << "\n";
    }
    return its_;
}