#pragma once
#include "preconditioner.hpp"

namespace dd { namespace algebra {

/// Identity preconditioner: z = r.
class IdentityPreconditioner final : public Preconditioner {
public:
    using Preconditioner::Scalar;
    using Preconditioner::Index;
    ~IdentityPreconditioner() override = default;

    void update(const MatrixSparse& /*A*/) override { /* no-op */ }

    void apply(const std::vector<Scalar>& r,
               std::vector<Scalar>& z) const override
    {
        z = r;
    }
};

}} // namespace dd::algebra
