#include "solver/pcg_mpi.hpp"

#include <cmath>
#include <stdexcept>

// -----------------------------------------------------------------------------
// Global dot-product and 2-norm using MPI_Allreduce
// -----------------------------------------------------------------------------
static inline double dot_global(const std::vector<double>& a,
                                const std::vector<double>& b,
                                MPI_Comm comm)
{
    if (a.size() != b.size())
        throw std::runtime_error("dot_global: size mismatch");

    double local = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
        local += a[i] * b[i];

    double global = 0.0;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, comm);
    return global;
}

static inline double nrm2_global(const std::vector<double>& a,
                                 MPI_Comm comm)
{
    double local = 0.0;
    for (double v : a) local += v * v;

    double global = 0.0;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, comm);
    return std::sqrt(global);
}

// -----------------------------------------------------------------------------
// Constructor
// -----------------------------------------------------------------------------
PCGSolverMPI::PCGSolverMPI(const MatrixSparse& A,
                           Preconditioner*     M,
                           MPI_Comm            comm)
    : Solver(A, M), comm_(comm)
{
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &size_);
}

// -----------------------------------------------------------------------------
// PCG solve on distributed (row-partitioned) system.
//
//  - A_     : *local* matrix (A_loc) on each rank
//  - b, x   : *local* vectors of size n_loc
//  - M_     : local preconditioner (e.g. AdditiveSchwarz on A_loc)
//  - Global reductions only in dot_global / nrm2_global.
// -----------------------------------------------------------------------------
std::size_t PCGSolverMPI::solve(const std::vector<Scalar>& b,
                                std::vector<Scalar>&       x)
{
    const std::size_t n_loc = b.size();

    if (x.size() != n_loc)
        x.assign(n_loc, Scalar{0});

    std::vector<Scalar> r(n_loc), z(n_loc), p(n_loc), Ap(n_loc);

    // r_loc = b_loc - A_loc x_loc
    A_.gemv(x, r, Scalar{1}, Scalar{0});  // r = A x
    for (std::size_t i = 0; i < n_loc; ++i)
        r[i] = b[i] - r[i];

    const double r0_norm = nrm2_global(r, comm_);
    last_rnorm_ = r0_norm;
    its_        = 0;
    converged_  = (r0_norm == 0.0);
    last_rel_   = 0.0;

    if (converged_)
        return 0;

    // z = M^{-1} r  (local preconditioner)
    if (M_)
        M_->apply(r, z);
    else
        z = r;

    p = z;

    double rho = dot_global(r, z, comm_);
    if (std::abs(rho) == 0.0)
        throw std::runtime_error("PCG breakdown: rho == 0");

    const std::size_t maxit = maxIters();
    const double      tol   = tolerance();

    its_ = 0;
    for (std::size_t k = 0; k < maxit; ++k) {

        // Ap_loc = A_loc p_loc
        A_.gemv(p, Ap, Scalar{1}, Scalar{0});

        const double pAp = dot_global(p, Ap, comm_);
        if (pAp <= 0.0)
            throw std::runtime_error("PCG breakdown: p^T A p <= 0 (matrix not SPD?)");

        const double alpha = rho / pAp;

        // x_loc += alpha * p_loc
        for (std::size_t i = 0; i < n_loc; ++i)
            x[i] += static_cast<Scalar>(alpha) * p[i];

        // r_loc -= alpha * Ap_loc
        for (std::size_t i = 0; i < n_loc; ++i)
            r[i] -= static_cast<Scalar>(alpha) * Ap[i];

        last_rnorm_ = nrm2_global(r, comm_);
        last_rel_   = (r0_norm > 0.0) ? (last_rnorm_ / r0_norm) : 0.0;
        ++its_;

        if (last_rel_ <= tol) {
            converged_ = true;
            break;
        }

        // z = M^{-1} r  (local preconditioner)
        if (M_)
            M_->apply(r, z);
        else
            z = r;

        const double rho_next = dot_global(r, z, comm_);
        if (std::abs(rho_next) == 0.0)
            throw std::runtime_error("PCG breakdown: rho_next == 0");

        const double beta = rho_next / rho;

        for (std::size_t i = 0; i < n_loc; ++i)
            p[i] = z[i] + static_cast<Scalar>(beta) * p[i];

        rho = rho_next;
    }

    // Only rank 0 prints convergence info to avoid duplicates.
    int myrank = 0;
    MPI_Comm_rank(comm_, &myrank);
    if (myrank == 0) {
        printConvergenceInfo(its_, last_rel_, tol);
    }

    return its_;
}
