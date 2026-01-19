#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>
#include <cmath>
#include <algorithm>

#include "algebra/COO.hpp"

using Index  = std::int64_t;
using Scalar = double;

using Coord3  = std::array<Scalar, 3>;
using FieldFn = std::function<Scalar(const Coord3&)>;

// New: coefficient function type
using CoeffFn = std::function<Scalar(const Coord3&)>;

class PDE {
public:
    // OLD constructor (kept): constant coefficients
    PDE(int dim,
        std::array<Index, 3> n,
        std::array<Scalar, 3> a,
        std::array<Scalar, 3> b,
        Scalar mu,
        Scalar c,
        FieldFn f,
        FieldFn g_dirichlet = FieldFn{},
        std::array<Scalar, 3> h = {Scalar(-1), Scalar(-1), Scalar(-1)});

    // NEW constructor (optional): variable coefficients
    PDE(int dim,
        std::array<Index, 3> n,
        std::array<Scalar, 3> a,
        std::array<Scalar, 3> b,
        CoeffFn mu_fun,     // must return > 0
        CoeffFn c_fun,      // must return >= 0
        FieldFn f,
        FieldFn g_dirichlet = FieldFn{},
        std::array<Scalar, 3> h = {Scalar(-1), Scalar(-1), Scalar(-1)});

    int dim() const noexcept { return dim_; }
    std::array<Index,3> n() const noexcept { return n_; }
    std::array<Scalar,3> a() const noexcept { return a_; }
    std::array<Scalar,3> b() const noexcept { return b_; }
    std::array<Scalar,3> h() const noexcept { return h_; }

    Index numUnknowns() const noexcept;

    std::pair<MatrixCOO, std::vector<Scalar>> assembleCOO() const;

private:
    int dim_;
    std::array<Index,3>  n_;
    std::array<Scalar,3> a_;
    std::array<Scalar,3> b_;
    std::array<Scalar,3> h_;

    // Keep the constants (useful for debugging / metadata)
    Scalar mu_const_ = 1.0;
    Scalar c_const_  = 0.0;

    // Always used by assembly (even for constant case)
    CoeffFn mu_fun_;
    CoeffFn c_fun_;

    FieldFn f_;
    FieldFn g_; // Dirichlet; if empty => zero

    void validate_() const;
    void check_coeffs_sanity_() const;

    Scalar gOrZero_(const Coord3& x) const;

    Index idx1_(Index i) const;
    Index idx2_(Index i, Index j) const;
    Index idx3_(Index i, Index j, Index k) const;

    Coord3 coord_(Index i, Index j, Index k) const;

    // helper: harmonic average (stable)
    static Scalar harmonic_(Scalar a, Scalar b) {
        // assume a>0,b>0
        return (Scalar(2) * a * b) / (a + b);
    }
};
