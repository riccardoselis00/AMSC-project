
#pragma once

#include <vector>
#include <cstddef>
#include "dd/algebra/matrixSparse.hpp"

namespace dd { namespace algebra {

/**
 * @brief Abstract preconditioner base class.
 *
 * A preconditioner approximates A^{-1}.  Given a residual r, it
 * computes z ≈ A^{-1} r via @ref apply().  Some preconditioners
 * build internal data from A in @ref update().
 */
class Preconditioner {
public:
    using Scalar = double;
    using Index  = std::size_t;

    virtual ~Preconditioner();

    /// (Re)build internal structures for the given matrix A.
    /// Default implementation does nothing.
    virtual void update(const MatrixSparse& /*A*/) {}

    /// Compute z = M^{-1} r. Must be implemented by derived classes.
    virtual void apply(const std::vector<Scalar>& r,
                       std::vector<Scalar>& z) const = 0;
};

}} // namespace dd::algebra
