#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>
#include <cmath>

// Forward declare your COO type (include your real header instead)
#include "algebra/COO.hpp" // must provide MatrixCOO(rows, cols), reserve(nnz), add(i,j,val)


// Adapt these to your project typedefs if you already have them.
using Index  = std::int64_t;
using Scalar = double;

// 3D coordinate container; for dim<3 only first components are used.
using Coord3 = std::array<Scalar, 3>;

// f(x) and g(x) (Dirichlet data) signatures
using FieldFn = std::function<Scalar(const Coord3&)>;

// Diffusion–reaction PDE:
//   -div(mu grad u) + c u = f  in Omega
//   u = g on boundary (Dirichlet)
//
// Discretization: FD on a Cartesian grid, interior unknowns only.
class PDE {
public:
    // dim = 1,2,3
    // n = number of INTERIOR points in each direction (n[1]=1 for 1D, n[2]=1 for 2D).
    // a,b = domain bounds per direction. For 1D only a[0],b[0] used; etc.
    // h = grid spacing per direction. If h[d] <= 0, it is computed as (b[d]-a[d])/(n[d]+1).
    PDE(int dim,
        std::array<Index, 3> n,
        std::array<Scalar, 3> a,
        std::array<Scalar, 3> b,
        Scalar mu,
        Scalar c,
        FieldFn f,
        FieldFn g_dirichlet = FieldFn{},              // optional (defaults to 0)
        std::array<Scalar, 3> h = {Scalar(-1), Scalar(-1), Scalar(-1)});

    int dim() const noexcept { return dim_; }
    std::array<Index,3> n() const noexcept { return n_; }
    std::array<Scalar,3> a() const noexcept { return a_; }
    std::array<Scalar,3> b() const noexcept { return b_; }
    std::array<Scalar,3> h() const noexcept { return h_; }

    Index numUnknowns() const noexcept;

    // Assemble and return {A, rhs}
    std::pair<MatrixCOO, std::vector<Scalar>> assembleCOO() const;

private:
    int dim_;
    std::array<Index,3>  n_;
    std::array<Scalar,3> a_;
    std::array<Scalar,3> b_;
    std::array<Scalar,3> h_;
    Scalar mu_;
    Scalar c_;
    FieldFn f_;
    FieldFn g_; // Dirichlet; if empty => zero

    void validate_() const;
    Scalar gOrZero_(const Coord3& x) const;

    // Global 1D index from interior multi-index (i,j,k are 1..n[d])
    Index idx1_(Index i) const;
    Index idx2_(Index i, Index j) const;
    Index idx3_(Index i, Index j, Index k) const;

    // Coordinate of interior node
    Coord3 coord_(Index i, Index j, Index k) const;
};

