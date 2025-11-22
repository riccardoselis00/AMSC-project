#pragma once

#include "preconditioner.hpp"

class IdentityPreconditioner final : public Preconditioner {
public:
    using Preconditioner::Scalar;
    using Preconditioner::Index;
    ~IdentityPreconditioner() override = default;

    void update(const MatrixSparse&) override {}

    void apply(const std::vector<Scalar>& r,
               std::vector<Scalar>& z) const override
    {
        z = r;
    }
};
