#pragma once

#include <cstddef>
#include <vector>
#include <memory>
#include <functional>
#include <string>
#include <stdexcept>

namespace dd { namespace algebra {

// Identify the concrete storage layout behind a MatrixSparse
enum class SparseFormat { COO, CSR };

/**
 * @brief Abstract base for sparse matrices.
 *
 * Child classes (e.g. MatrixCOO, MatrixCSR) must implement the core
 * virtual methods. The base provides convenience wrappers and
 * utilities that can be expressed generically (vector overloads,
 * norms, pretty‑print) without assuming a particular storage format.
 */
class MatrixSparse {
public:
    using Scalar = double;
    using Index  = std::size_t;
    using TripletVisitor = std::function<void(Index i, Index j, Scalar v)>;

    virtual ~MatrixSparse() = default;

    // ----- Shape & identity -----
    virtual Index rows()   const noexcept = 0;
    virtual Index cols()   const noexcept = 0;
    virtual Index nnz()    const noexcept = 0;
    virtual SparseFormat format() const noexcept = 0;

    // ----- Core operation -----
    // Compute y = alpha * A * x + beta * y, where x length = cols(), y length = rows().
    virtual void gemv(const Scalar* x, Scalar* y,
                      Scalar alpha = 1.0, Scalar beta = 0.0) const = 0;

    // Vector convenience overloads.
    void gemv(const std::vector<Scalar>& x, std::vector<Scalar>& y,
              Scalar alpha = 1.0, Scalar beta = 0.0) const;
    std::vector<Scalar> gemv(const std::vector<Scalar>& x) const;

    // ----- Structure access -----
    // Extract the diagonal (length = min(rows, cols)).
    virtual void extractDiagonal(std::vector<Scalar>& d) const = 0;

    // Visit all nonzero entries (i,j,val). Order is implementation‑defined.
    virtual void forEachNZ(const TripletVisitor& f) const = 0;

    // Polymorphic copy.
    virtual std::unique_ptr<MatrixSparse> clone() const = 0;

    // ----- Generic utilities -----
    // Frobenius norm: sqrt(sum |v|^2).
    Scalar frobeniusNorm() const;
    // Maximum absolute value of any stored element.
    Scalar maxAbs() const;
    // Produce a human‑readable preview of the matrix. Limits the number of lines.
    std::string toString(std::size_t max_lines = 40) const;
};

}} // namespace dd::algebra