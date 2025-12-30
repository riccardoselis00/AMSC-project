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
: dim_(dim), n_(n), a_(a), b_(b), h_(h), mu_(mu), c_(c), f_(std::move(f)), g_(std::move(g_dirichlet))
{
    validate_();

    // Fill missing h from domain length / (n+1)
    for (int d = 0; d < dim_; ++d) {
        if (h_[d] <= Scalar(0)) {
            const Scalar L = b_[d] - a_[d];
            require(L > Scalar(0), "PDE: invalid domain length (b<=a).");
            require(n_[d] > 0,     "PDE: n[d] must be >0 for active dimensions.");
            h_[d] = L / Scalar(n_[d] + 1);
        }
    }
}

void PDE::validate_() const {
    require(dim_ >= 1 && dim_ <= 3, "PDE: dim must be 1,2,or 3.");
    require(mu_ > Scalar(0),        "PDE: mu must be > 0.");
    require(c_  >= Scalar(0),       "PDE: c must be >= 0.");
    require(static_cast<bool>(f_),  "PDE: rhs function f(x) must be provided.");

    // For inactive dims, enforce n=1 by convention (so loops don’t break).
    // You can relax this if you prefer.
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

Scalar PDE::gOrZero_(const Coord3& x) const {
    if (g_) return g_(x);
    return Scalar(0);
}

Index PDE::numUnknowns() const noexcept {
    if (dim_ == 1) return n_[0];
    if (dim_ == 2) return n_[0] * n_[1];
    return n_[0] * n_[1] * n_[2];
}

Index PDE::idx1_(Index i) const {
    // i = 1..nx
    return i - 1;
}

Index PDE::idx2_(Index i, Index j) const {
    // i = 1..nx, j=1..ny
    return (j - 1) * n_[0] + (i - 1);
}

Index PDE::idx3_(Index i, Index j, Index k) const {
    // i = 1..nx, j=1..ny, k=1..nz
    return ((k - 1) * n_[1] + (j - 1)) * n_[0] + (i - 1);
}

Coord3 PDE::coord_(Index i, Index j, Index k) const {
    Coord3 x {Scalar(0), Scalar(0), Scalar(0)};
    // interior node coordinates: a + i*h (i starts at 1)
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

    // Reserve a safe upper bound (interior-only stencil: 1 + 2*dim nonzeros per row)
    const std::size_t nnz_est = static_cast<std::size_t>(N) * static_cast<std::size_t>(1 + 2 * dim_);
    A.reserve(nnz_est);

    const Scalar invhx2 = Scalar(1) / (h_[0] * h_[0]);
    const Scalar invhy2 = (dim_ >= 2) ? Scalar(1) / (h_[1] * h_[1]) : Scalar(0);
    const Scalar invhz2 = (dim_ >= 3) ? Scalar(1) / (h_[2] * h_[2]) : Scalar(0);

    // Diagonal coefficient (constant mu,c, uniform per direction)
    const Scalar diag = c_
                      + Scalar(2) * mu_ * invhx2
                      + (dim_ >= 2 ? Scalar(2) * mu_ * invhy2 : Scalar(0))
                      + (dim_ >= 3 ? Scalar(2) * mu_ * invhz2 : Scalar(0));

    // Off-diagonal coefficients per direction
    const Scalar ax = -mu_ * invhx2;
    const Scalar ay = (dim_ >= 2) ? -mu_ * invhy2 : Scalar(0);
    const Scalar az = (dim_ >= 3) ? -mu_ * invhz2 : Scalar(0);

    const Index nx = n_[0];
    const Index ny = n_[1];
    const Index nz = n_[2];

    // Loops over interior nodes
    for (Index k = 1; k <= nz; ++k) {
        for (Index j = 1; j <= ny; ++j) {
            for (Index i = 1; i <= nx; ++i) {

                const Index row = (dim_ == 1) ? idx1_(i)
                                  : (dim_ == 2) ? idx2_(i, j)
                                                : idx3_(i, j, k);

                const Coord3 x = coord_(i, j, k);
                Scalar bval = f_(x);

                // Diagonal
                A.add(row, row, diag);

                // --- X neighbors ---
                // left (i-1)
                if (i > 1) {
                    const Index col = (dim_ == 1) ? idx1_(i - 1)
                                      : (dim_ == 2) ? idx2_(i - 1, j)
                                                    : idx3_(i - 1, j, k);
                    A.add(row, col, ax);
                } else {
                    // boundary at x=a
                    Coord3 xb = x;
                    xb[0] = a_[0]; // x= a
                    bval -= ax * gOrZero_(xb); // move ax*u_boundary to rhs (note ax is negative)
                }

                // right (i+1)
                if (i < nx) {
                    const Index col = (dim_ == 1) ? idx1_(i + 1)
                                      : (dim_ == 2) ? idx2_(i + 1, j)
                                                    : idx3_(i + 1, j, k);
                    A.add(row, col, ax);
                } else {
                    // boundary at x=b
                    Coord3 xb = x;
                    xb[0] = b_[0];
                    bval -= ax * gOrZero_(xb);
                }

                // --- Y neighbors ---
                if (dim_ >= 2) {
                    // down (j-1)
                    if (j > 1) {
                        const Index col = (dim_ == 2) ? idx2_(i, j - 1)
                                                      : idx3_(i, j - 1, k);
                        A.add(row, col, ay);
                    } else {
                        Coord3 xb = x;
                        xb[1] = a_[1];
                        bval -= ay * gOrZero_(xb);
                    }

                    // up (j+1)
                    if (j < ny) {
                        const Index col = (dim_ == 2) ? idx2_(i, j + 1)
                                                      : idx3_(i, j + 1, k);
                        A.add(row, col, ay);
                    } else {
                        Coord3 xb = x;
                        xb[1] = b_[1];
                        bval -= ay * gOrZero_(xb);
                    }
                }

                // --- Z neighbors ---
                if (dim_ >= 3) {
                    // back (k-1)
                    if (k > 1) {
                        const Index col = idx3_(i, j, k - 1);
                        A.add(row, col, az);
                    } else {
                        Coord3 xb = x;
                        xb[2] = a_[2];
                        bval -= az * gOrZero_(xb);
                    }

                    // front (k+1)
                    if (k < nz) {
                        const Index col = idx3_(i, j, k + 1);
                        A.add(row, col, az);
                    } else {
                        Coord3 xb = x;
                        xb[2] = b_[2];
                        bval -= az * gOrZero_(xb);
                    }
                }

                rhs[static_cast<std::size_t>(row)] = bval;
            }
        }
    }

    return {std::move(A), std::move(rhs)};
}

