#include <cmath>
#include <limits>
#include <stdexcept>
#include <typeinfo>

#include "algebra/matrixSparse.hpp"
#include "algebra/CSR.hpp"
#include "algebra/COO.hpp"
#include "preconditioner/diagonal_jacobi.hpp"

// --------------------------------------
// Constructor
// --------------------------------------
DiagonalJacobi::DiagonalJacobi(int nparts)
    : m_nparts(nparts), m_n(0)
{
    if (m_nparts <= 0) {
        throw std::invalid_argument("DiagonalJacobi: nparts must be > 0");
    }
}

// --------------------------------------
// Internal helpers
// --------------------------------------

using Index  = MatrixSparse::Index;
using Scalar = MatrixSparse::Scalar;

double diag_from_csr_row(const MatrixCSR& A, int i)
{
    const auto& rowPtr = A.rowPtr();
    const auto& colIdx = A.colIndex();
    const auto& vals   = A.values();

    Index ii  = static_cast<Index>(i);
    Index beg = rowPtr[ii];
    Index end = rowPtr[ii + 1];
    for (Index k = beg; k < end; ++k) {
        if (colIdx[k] == ii) {
            return static_cast<double>(vals[k]);
        }
    }
    return 0.0;
}


// --------------------------------------
// Setup
// --------------------------------------
void DiagonalJacobi::setup(const MatrixSparse& A)
{
    using Index  = MatrixSparse::Index;
    using Scalar = MatrixSparse::Scalar;

    m_n = static_cast<int>(A.rows());
    if (A.cols() != static_cast<std::size_t>(m_n)) {
        throw std::runtime_error("DiagonalJacobi::setup requires a square matrix.");
    }

    // 1) Partition rows into nparts chunks (just like before)
    m_starts.resize(static_cast<std::size_t>(m_nparts) + 1);
    int base = m_n / m_nparts;
    int rem  = m_n % m_nparts;
    int pos  = 0;
    for (int p = 0; p < m_nparts; ++p) {
        m_starts[static_cast<std::size_t>(p)] = pos;
        int sz = base + (p < rem ? 1 : 0);
        pos += sz;
    }
    m_starts[static_cast<std::size_t>(m_nparts)] = m_n;

    // 2) Compute diagonal entries
    m_inv_diag.assign(static_cast<std::size_t>(m_n), 0.0);

    if (auto csr = dynamic_cast<const MatrixCSR*>(&A)) {
        // Fast path: CSR row iteration
        for (Index i = 0; i < static_cast<Index>(m_n); ++i) {
            double aii = diag_from_csr_row(*csr, static_cast<int>(i));
            if (aii == 0.0) {
                throw std::runtime_error(
                    "DiagonalJacobi::setup: zero diagonal at row " + std::to_string(i));
            }
            m_inv_diag[static_cast<std::size_t>(i)] = 1.0 / aii;
        }
    } else {
        // Generic path: one pass over all nonzeros
        std::vector<Scalar> diag(static_cast<std::size_t>(m_n), Scalar{0});

        A.forEachNZ([&](Index r, Index c, Scalar v) {
            if (r == c) {
                diag[static_cast<std::size_t>(r)] = v;
            }
        });

        for (Index i = 0; i < static_cast<Index>(m_n); ++i) {
            Scalar aii = diag[static_cast<std::size_t>(i)];
            if (aii == Scalar{0}) {
                throw std::runtime_error(
                    "DiagonalJacobi::setup: zero diagonal at row " + std::to_string(i));
            }
            m_inv_diag[static_cast<std::size_t>(i)] =
                1.0 / static_cast<double>(aii);
        }
    }
}

// --------------------------------------
// Apply
// --------------------------------------
void DiagonalJacobi::apply(const std::vector<double>& r,
                           std::vector<double>&       z) const
{
    if (r.size() != static_cast<std::size_t>(m_n)) {
        throw std::runtime_error("DiagonalJacobi::apply: r has wrong size");
    }

    z.assign(r.size(), 0.0);

    // If you care about "chunks" (nparts) for cache behavior / future MPI,
    // keep this partitioned loop. Otherwise, you could just do a single loop.
    for (int p = 0; p < m_nparts; ++p) {
        int s = m_starts[static_cast<std::size_t>(p)];
        int e = m_starts[static_cast<std::size_t>(p + 1)];
        for (int i = s; i < e; ++i) {
            std::size_t idx = static_cast<std::size_t>(i);
            z[idx] = m_inv_diag[idx] * r[idx];
        }
    }
}
