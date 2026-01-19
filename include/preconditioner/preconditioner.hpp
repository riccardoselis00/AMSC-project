#pragma once

#include <vector>
#include <cstddef>
#include "algebra/matrixSparse.hpp"

class Preconditioner {
public:
    using Scalar = double;
    using Index  = std::size_t;

    virtual ~Preconditioner();

virtual void setup(const MatrixSparse& /*A*/) { }
virtual void update(const MatrixSparse& /*A*/) { }

    virtual void apply(const std::vector<Scalar>& r,
                       std::vector<Scalar>& z) const = 0;
};