#include "pde/pde.hpp"

static inline void require(bool cond, const char* msg) {
    if (!cond) throw std::runtime_error(msg);
}

PDE::PDE(int dim,
         std::array<Index, 3> n,
         std::array<Scalar, 3> a,
         std::array<Scalar, 3> b,
         Scalar mu,
         Scalar c,
         FieldFn f,
         FieldFn g_dirichlet,
         std::array<Scalar, 3> h)
: dim_(dim)
, n_(n)
, a_(a)
, b_(b)
, h_(h)
, mu_const_(mu)
, c_const_(c)
, mu_fun_([mu](const Coord3&) { return mu; })
, c_fun_([c](const Coord3&) { return c; })
, f_(std::move(f))
, g_(std::move(g_dirichlet))
{
    validate_();

    for (int d = 0; d < dim_; ++d) {
        if (h_[d] <= Scalar(0)) {
            const Scalar L = b_[d] - a_[d];
            require(L > Scalar(0), "PDE: invalid domain length (b<=a).");
            require(n_[d] > 0,     "PDE: n[d] must be >0 for active dimensions.");
            h_[d] = L / Scalar(n_[d] + 1);
        }
    }

    check_coeffs_sanity_();
}

PDE::PDE(int dim,
         std::array<Index, 3> n,
         std::array<Scalar, 3> a,
         std::array<Scalar, 3> b,
         CoeffFn mu_fun,
         CoeffFn c_fun,
         FieldFn f,
         FieldFn g_dirichlet,
         std::array<Scalar, 3> h)
: dim_(dim)
, n_(n)
, a_(a)
, b_(b)
, h_(h)
, mu_fun_(std::move(mu_fun))
, c_fun_(std::move(c_fun))
, f_(std::move(f))
, g_(std::move(g_dirichlet))
{
    validate_();

    for (int d = 0; d < dim_; ++d) {
        if (h_[d] <= Scalar(0)) {
            const Scalar L = b_[d] - a_[d];
            require(L > Scalar(0), "PDE: invalid domain length (b<=a).");
            require(n_[d] > 0,     "PDE: n[d] must be >0 for active dimensions.");
            h_[d] = L / Scalar(n_[d] + 1);
        }
    }

    // best-effort metadata for logs (not used in assembly)
    mu_const_ = 1.0;
    c_const_  = 0.0;

    check_coeffs_sanity_();
}

void PDE::validate_() const {
    require(dim_ >= 1 && dim_ <= 3, "PDE: dim must be 1,2,or 3.");
    require(static_cast<bool>(mu_fun_), "PDE: mu function must be provided.");
    require(static_cast<bool>(c_fun_),  "PDE: c function must be provided.");
    require(static_cast<bool>(f_),      "PDE: rhs function f(x) must be provided.");

    if (dim_ == 1) {
        require(n_[0] > 0, "PDE: n[0] must be > 0 for 1D.");
        require(n_[1] == 1 && n_[2] == 1, "PDE: for 1D, set n={nx,1,1}.");
    } else if (dim_ == 2) {
        require(n_[0] > 0 && n_[1] > 0, "PDE: n[0],n[1] must be > 0 for 2D.");
        require(n_[2] == 1, "PDE: for 2D, set n={nx,ny,1}.");
    } else {
        require(n_[0] > 0 && n_[1] > 0 && n_[2] > 0, "PDE: n[0],n[1],n[2] must be > 0 for 3D.");
    }
}

void PDE::check_coeffs_sanity_() const {
    // Sample a few points to catch negative / NaNs early (cheap constant-time)
    const auto sample = [&](Scalar x0, Scalar x1, Scalar x2) {
        Coord3 x{ x0, x1, x2 };
        const Scalar mu = mu_fun_(x);
        const Scalar c  = c_fun_(x);
        require(std::isfinite(mu) && mu > Scalar(0), "PDE: mu(x) must be finite and > 0.");
        require(std::isfinite(c)  && c  >= Scalar(0), "PDE: c(x) must be finite and >= 0.");
    };

    // interior-ish points
    const Scalar xm = Scalar(0.5)*(a_[0]+b_[0]);
    const Scalar ym = Scalar(0.5)*(a_[1]+b_[1]);
    const Scalar zm = Scalar(0.5)*(a_[2]+b_[2]);

    sample(xm, ym, zm);
    sample(a_[0] + h_[0], ym, zm);
    sample(b_[0] - h_[0], ym, zm);
    if (dim_ >= 2) {
        sample(xm, a_[1] + h_[1], zm);
        sample(xm, b_[1] - h_[1], zm);
    }
    if (dim_ >= 3) {
        sample(xm, ym, a_[2] + h_[2]);
        sample(xm, ym, b_[2] - h_[2]);
    }
}

Scalar PDE::gOrZero_(const Coord3& x) const {
    return g_ ? g_(x) : Scalar(0);
}

Index PDE::numUnknowns() const noexcept {
    if (dim_ == 1) return n_[0];
    if (dim_ == 2) return n_[0] * n_[1];
    return n_[0] * n_[1] * n_[2];
}

Index PDE::idx1_(Index i) const { return i - 1; }

Index PDE::idx2_(Index i, Index j) const {
    return (j - 1) * n_[0] + (i - 1);
}

Index PDE::idx3_(Index i, Index j, Index k) const {
    return ((k - 1) * n_[1] + (j - 1)) * n_[0] + (i - 1);
}

Coord3 PDE::coord_(Index i, Index j, Index k) const {
    Coord3 x{Scalar(0), Scalar(0), Scalar(0)};
    x[0] = a_[0] + Scalar(i) * h_[0];
    if (dim_ >= 2) x[1] = a_[1] + Scalar(j) * h_[1];
    if (dim_ >= 3) x[2] = a_[2] + Scalar(k) * h_[2];
    return x;
}

std::pair<MatrixCOO, std::vector<Scalar>> PDE::assembleCOO() const {
    const Index N = numUnknowns();
    require(N > 0, "PDE::assembleCOO: number of unknowns must be > 0.");

    MatrixCOO A(N, N);
    std::vector<Scalar> rhs(static_cast<std::size_t>(N), Scalar(0));

    const std::size_t nnz_est = static_cast<std::size_t>(N) * static_cast<std::size_t>(1 + 2 * dim_);
    A.reserve(nnz_est);

    const Scalar invhx2 = Scalar(1) / (h_[0] * h_[0]);
    const Scalar invhy2 = (dim_ >= 2) ? Scalar(1) / (h_[1] * h_[1]) : Scalar(0);
    const Scalar invhz2 = (dim_ >= 3) ? Scalar(1) / (h_[2] * h_[2]) : Scalar(0);

    const Index nx = n_[0];
    const Index ny = n_[1];
    const Index nz = n_[2];

    for (Index k = 1; k <= nz; ++k) {
        for (Index j = 1; j <= ny; ++j) {
            for (Index i = 1; i <= nx; ++i) {

                const Index row = (dim_ == 1) ? idx1_(i)
                                  : (dim_ == 2) ? idx2_(i, j)
                                                : idx3_(i, j, k);

                const Coord3 x  = coord_(i, j, k);

                const Scalar mu_c = mu_fun_(x);
                const Scalar c_c  = c_fun_(x);

                Scalar diag = c_c;
                Scalar bval = f_(x);

                // ---- X direction (faces i-1/2 and i+1/2) ----
                {
                    // left face
                    Scalar muL = mu_c;
                    if (i > 1) {
                        Coord3 xn = x; xn[0] -= h_[0];
                        const Scalar mu_n = mu_fun_(xn);
                        muL = harmonic_(mu_c, mu_n);
                        const Index col = (dim_ == 1) ? idx1_(i - 1)
                                      : (dim_ == 2) ? idx2_(i - 1, j)
                                                    : idx3_(i - 1, j, k);
                        const Scalar aL = -muL * invhx2;
                        A.add(row, col, aL);
                        diag -= aL; // add +muL/h^2
                    } else {
                        // boundary x=a
                        const Scalar aL = -muL * invhx2;
                        diag -= aL;
                        Coord3 xb = x; xb[0] = a_[0];
                        bval -= aL * gOrZero_(xb);
                    }

                    // right face
                    Scalar muR = mu_c;
                    if (i < nx) {
                        Coord3 xn = x; xn[0] += h_[0];
                        const Scalar mu_n = mu_fun_(xn);
                        muR = harmonic_(mu_c, mu_n);
                        const Index col = (dim_ == 1) ? idx1_(i + 1)
                                      : (dim_ == 2) ? idx2_(i + 1, j)
                                                    : idx3_(i + 1, j, k);
                        const Scalar aR = -muR * invhx2;
                        A.add(row, col, aR);
                        diag -= aR;
                    } else {
                        const Scalar aR = -muR * invhx2;
                        diag -= aR;
                        Coord3 xb = x; xb[0] = b_[0];
                        bval -= aR * gOrZero_(xb);
                    }
                }

                // ---- Y direction ----
                if (dim_ >= 2) {
                    // down
                    Scalar muD = mu_c;
                    if (j > 1) {
                        Coord3 xn = x; xn[1] -= h_[1];
                        const Scalar mu_n = mu_fun_(xn);
                        muD = harmonic_(mu_c, mu_n);
                        const Index col = (dim_ == 2) ? idx2_(i, j - 1)
                                                      : idx3_(i, j - 1, k);
                        const Scalar aD = -muD * invhy2;
                        A.add(row, col, aD);
                        diag -= aD;
                    } else {
                        const Scalar aD = -muD * invhy2;
                        diag -= aD;
                        Coord3 xb = x; xb[1] = a_[1];
                        bval -= aD * gOrZero_(xb);
                    }

                    // up
                    Scalar muU = mu_c;
                    if (j < ny) {
                        Coord3 xn = x; xn[1] += h_[1];
                        const Scalar mu_n = mu_fun_(xn);
                        muU = harmonic_(mu_c, mu_n);
                        const Index col = (dim_ == 2) ? idx2_(i, j + 1)
                                                      : idx3_(i, j + 1, k);
                        const Scalar aU = -muU * invhy2;
                        A.add(row, col, aU);
                        diag -= aU;
                    } else {
                        const Scalar aU = -muU * invhy2;
                        diag -= aU;
                        Coord3 xb = x; xb[1] = b_[1];
                        bval -= aU * gOrZero_(xb);
                    }
                }

                // ---- Z direction ----
                if (dim_ >= 3) {
                    // back
                    Scalar muB = mu_c;
                    if (k > 1) {
                        Coord3 xn = x; xn[2] -= h_[2];
                        const Scalar mu_n = mu_fun_(xn);
                        muB = harmonic_(mu_c, mu_n);
                        const Index col = idx3_(i, j, k - 1);
                        const Scalar aB = -muB * invhz2;
                        A.add(row, col, aB);
                        diag -= aB;
                    } else {
                        const Scalar aB = -muB * invhz2;
                        diag -= aB;
                        Coord3 xb = x; xb[2] = a_[2];
                        bval -= aB * gOrZero_(xb);
                    }

                    // front
                    Scalar muF = mu_c;
                    if (k < nz) {
                        Coord3 xn = x; xn[2] += h_[2];
                        const Scalar mu_n = mu_fun_(xn);
                        muF = harmonic_(mu_c, mu_n);
                        const Index col = idx3_(i, j, k + 1);
                        const Scalar aF = -muF * invhz2;
                        A.add(row, col, aF);
                        diag -= aF;
                    } else {
                        const Scalar aF = -muF * invhz2;
                        diag -= aF;
                        Coord3 xb = x; xb[2] = b_[2];
                        bval -= aF * gOrZero_(xb);
                    }
                }

                A.add(row, row, diag);
                rhs[static_cast<std::size_t>(row)] = bval;
            }
        }
    }

    return {std::move(A), std::move(rhs)};
}
