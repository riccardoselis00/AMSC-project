#pragma once

#include <vector>
#include <cstddef>
#include <string>
#include <initializer_list>
#include <stdexcept>
#include <fstream>
#include <cctype>
#include <limits>

class MatrixDense {
public:
    using Scalar = double;
    using Index  = std::size_t;

    MatrixDense() : m_rows(0), m_cols(0) {}

    MatrixDense(Index rows, 
                Index cols, 
                Scalar init_value = Scalar{0});

    MatrixDense(std::initializer_list<std::initializer_list<Scalar>> rows);

    static MatrixDense Zero(Index rows, Index cols) { return MatrixDense(rows, cols, Scalar{0}); }

    static MatrixDense Identity(Index n);

    inline Index rows() const noexcept { return m_rows; }

    inline Index cols() const noexcept { return m_cols; }

    inline Index size() const noexcept { return m_data.size(); }

    inline bool empty() const noexcept { return m_data.empty(); }

    inline Scalar*       data()       noexcept { return m_data.data(); }
    inline const Scalar* data() const noexcept { return m_data.data(); }

    inline Scalar& operator()(Index i, Index j) noexcept { return m_data[i * m_cols + j]; }
    inline const Scalar& operator()(Index i, Index j) const noexcept { return m_data[i * m_cols + j]; }

    Scalar& at(Index i, Index j);
    const Scalar& at(Index i, Index j) const;

    void fill(Scalar v) noexcept;

    void setZero() noexcept { fill(Scalar{0}); }

    void setIdentity();

    void resize(Index rows, 
                Index cols, 
                bool keep = false, 
                Scalar pad_value = Scalar{0});

    void gemv(const Scalar* x, 
                Scalar* y, 
                Scalar alpha = 1.0, 
                Scalar beta = 0.0) const noexcept;

    void gemv(const std::vector<Scalar>& x, 
                std::vector<Scalar>& y, 
                Scalar alpha = 1.0, 
                Scalar beta = 0.0) const;

    std::vector<Scalar> gemv(const std::vector<Scalar>& x) const;

    static void gemm(const MatrixDense& A, 
                const MatrixDense& B, 
                MatrixDense& C, 
                Scalar alpha = 1.0, 
                Scalar beta = 0.0);

    void scale(Scalar alpha) noexcept;
   
    void axpy(Scalar alpha, const MatrixDense& B);

    MatrixDense transpose() const;
    
    void transposeInPlace();

    Scalar frobeniusNorm() const noexcept;
    
    Scalar maxAbs() const noexcept;

    void extractDiagonal(std::vector<Scalar>& d) const;

    std::string toString(Index max_rows = 6, Index max_cols = 6) const;

    MatrixDense block(Index i0, Index j0, Index r, Index c) const;

    static MatrixDense from_mm_array_file(const std::string& filename);

    static MatrixDense from_mm_array_stream(std::istream& is);

    bool to_mm_array_file(const std::string& filename, const std::string& symmetry = "general") const;
    
    bool to_mm_array_stream(std::ostream& os, const std::string& symmetry = "general") const;

private:
    Index m_rows{};
    Index m_cols{};
    std::vector<Scalar> m_data;
};