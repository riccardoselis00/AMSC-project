#pragma once

#include <cstddef>
#include <vector>
#include <memory>
#include <functional>
#include <string>
#include <stdexcept>


enum class SparseFormat { COO, CSR };

class MatrixSparse {
public:
    using Scalar = double;
    using Index  = std::size_t;
    using TripletVisitor = std::function<void(Index i, Index j, Scalar v)>;

    virtual ~MatrixSparse() = default;

    virtual Index rows()   const noexcept = 0;
    virtual Index cols()   const noexcept = 0;
    virtual Index nnz()    const noexcept = 0;
    virtual SparseFormat format() const noexcept = 0;


    virtual void gemv(const Scalar* x, Scalar* y,
                      Scalar alpha = 1.0, Scalar beta = 0.0) const = 0;

    void gemv(const std::vector<Scalar>& x, std::vector<Scalar>& y,
              Scalar alpha = 1.0, Scalar beta = 0.0) const;
    std::vector<Scalar> gemv(const std::vector<Scalar>& x) const;

    virtual void extractDiagonal(std::vector<Scalar>& d) const = 0;

    virtual void forEachNZ(const TripletVisitor& f) const = 0;

    virtual std::unique_ptr<MatrixSparse> clone() const = 0;

    Scalar frobeniusNorm() const;

    Scalar maxAbs() const;

    std::string toString(std::size_t max_lines = 40) const;
};