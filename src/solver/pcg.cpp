
#include "solver/pcg.hpp"
#include <cmath>
#include <stdexcept>

static inline double dot(const std::vector<double>& a,
                         const std::vector<double>& b)
{
    if (a.size() != b.size())
        throw std::runtime_error("dot: size mismatch");
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

static inline double nrm2(const std::vector<double>& a)
{
    double s = 0.0;
    for (double v : a) s += v * v;
    return std::sqrt(s);
}

std::size_t PCGSolver::solve(const std::vector<Scalar>& b,
                             std::vector<Scalar>& x)
{
    const std::size_t n = b.size();
    
    if (x.size() != n) x.assign(n, 0.0);

    std::vector<Scalar> r(n), z(n), p(n), Ap(n);

    A_.gemv(x, r, 1.0, 0.0); 
    for (std::size_t i = 0; i < n; ++i) r[i] = b[i] - r[i];

    const double r0_norm = nrm2(r);
    last_rnorm_ = r0_norm;
    its_ = 0;
    converged_ = (r0_norm == 0.0);
    last_rel_ = 0.0;

    if (converged_) return 0;

    if (M_) M_->apply(r, z); else z = r;
    p = z;
    double rho = dot(r, z);
    if (std::abs(rho) == 0.0)
        throw std::runtime_error("PCG breakdown: rho == 0");

    const std::size_t maxit = maxIters();
    const double tol = tolerance();
    its_ = 0;
    for (std::size_t k = 0; k < maxit; ++k) {

        A_.gemv(p, Ap, 1.0, 0.0);
        const double pAp = dot(p, Ap);
        if (pAp <= 0.0)
            throw std::runtime_error("PCG breakdown: p^T A p <= 0 (matrix not SPD?)");

        const double alpha = rho / pAp;

        for (std::size_t i = 0; i < n; ++i) x[i] += alpha * p[i];

        for (std::size_t i = 0; i < n; ++i) r[i] -= alpha * Ap[i];

        last_rnorm_ = nrm2(r);
        last_rel_   = (r0_norm > 0.0) ? (last_rnorm_ / r0_norm) : 0.0;
        ++its_;
        if (last_rel_ <= tol) { converged_ = true; break; }

        if (M_) M_->apply(r, z); else z = r;
        const double rho_next = dot(r, z);
        if (std::abs(rho_next) == 0.0)
            throw std::runtime_error("PCG breakdown: rho_next == 0");

        const double beta = rho_next / rho;

        for (std::size_t i = 0; i < n; ++i) p[i] = z[i] + beta * p[i];

        rho = rho_next;
        // const double tol = tolerance();
    }

    printConvergenceInfo(its_, last_rel_, tol);

    return its_;
}

