// #include <cmath>
// #include <limits>
// #include <algorithm>   // std::max, std::min

// #include "algebra/matrixSparse.hpp"
// #include "algebra/CSR.hpp"
// #include "algebra/COO.hpp"
// #include "preconditioner/additive_schwarz.hpp"

// #include <typeinfo>
// #include <stdexcept>

// // -----------------------------------------------------------------------------
// // Small dense LU factorization + solve, EXACTLY like in BlockJacobi,
// // but we keep them local (static) to this .cpp.
// // -----------------------------------------------------------------------------
// static void lu_factor_block(std::vector<double>& M, int n)
// {
//   for (int k = 0; k < n; ++k) {
//     double pivot = M[k*n + k];
//     if (std::abs(pivot) < 1e-15) {
//       throw std::runtime_error(
//         "AdditiveSchwarz::lu_factor_block: zero/near-zero pivot");
//     }

//     for (int i = k + 1; i < n; ++i) {
//       M[i*n + k] /= pivot;
//       double lik = M[i*n + k];
//       for (int j = k + 1; j < n; ++j) {
//         M[i*n + j] -= lik * M[k*n + j];
//       }
//     }
//   }
// }

// static void lu_solve_block(const std::vector<double>& LU,
//                            int n,
//                            const double* b,
//                            double* x)
// {
//   // Forward substitution: L y = b (L has unit diagonal)
//   std::vector<double> y(static_cast<std::size_t>(n));
//   for (int i = 0; i < n; ++i) {
//     double sum = b[i];
//     for (int j = 0; j < i; ++j) {
//       sum -= LU[i*n + j] * y[j];
//     }
//     y[i] = sum;
//   }

//   // Backward substitution: U x = y
//   for (int i = n - 1; i >= 0; --i) {
//     double sum = y[static_cast<std::size_t>(i)];
//     for (int j = i + 1; j < n; ++j) {
//       sum -= LU[i*n + j] * x[j];
//     }
//     double pivot = LU[i*n + i];
//     if (std::abs(pivot) < 1e-15) {
//       throw std::runtime_error(
//         "AdditiveSchwarz::lu_solve_block: zero/near-zero pivot");
//     }
//     x[i] = sum / pivot;
//   }
// }

// static void lu_SSOR(const std::vector<double>& A,
//                            int n,
//                            const double* b,
//                            double* x)
// {
//   // SSOR parameters (can be tuned or made configurable)
//   const int    sweeps = 2;     // 10 SSOR sweeps (forward+backward)
//   const double omega  = 1.95;   // ω = 1 -> symmetric Gauss–Seidel

//   // Start from zero: ensures linearity in b (important for a preconditioner)
//   std::fill(x, x + n, 0.0);

//   std::vector<double> x_old(static_cast<std::size_t>(n));

//   for (int k = 0; k < sweeps; ++k) {

//     // ----- Forward SOR sweep -----
//     std::copy(x, x + n, x_old.data());

//     for (int i = 0; i < n; ++i) {
//       double sum_lower = 0.0;
//       for (int j = 0; j < i; ++j) {
//         sum_lower += A[i*n + j] * x[j];        // new values
//       }

//       double sum_upper = 0.0;
//       for (int j = i + 1; j < n; ++j) {
//         sum_upper += A[i*n + j] * x_old[j];    // old values
//       }

//       double aii = A[i*n + i];
//       if (std::abs(aii) < 1e-15) {
//         throw std::runtime_error(
//           "AdditiveSchwarz::lu_solve_block (forward SSOR): zero/near-zero pivot");
//       }

//       double x_gs = (b[i] - sum_lower - sum_upper) / aii; // Gauss–Seidel update
//       x[i] = (1.0 - omega) * x_old[i] + omega * x_gs;     // SOR relaxation
//     }

//     // ----- Backward SOR sweep (symmetric) -----
//     std::copy(x, x + n, x_old.data());

//     for (int i = n - 1; i >= 0; --i) {
//       double sum_upper = 0.0;
//       for (int j = i + 1; j < n; ++j) {
//         sum_upper += A[i*n + j] * x[j];        // new values (backwards)
//       }

//       double sum_lower = 0.0;
//       for (int j = 0; j < i; ++j) {
//         sum_lower += A[i*n + j] * x_old[j];    // old values
//       }

//       double aii = A[i*n + i];
//       if (std::abs(aii) < 1e-15) {
//         throw std::runtime_error(
//           "AdditiveSchwarz::lu_solve_block (backward SSOR): zero/near-zero pivot");
//       }

//       double x_gs = (b[i] - sum_lower - sum_upper) / aii;
//       x[i] = (1.0 - omega) * x_old[i] + omega * x_gs;
//     }
//   }
// }


// // -----------------------------------------------------------------------------
// // Constructor
// // -----------------------------------------------------------------------------
// AdditiveSchwarz::AdditiveSchwarz(int nparts, int overlap)
//   : m_nparts(nparts),
//     m_overlap(overlap),
//     m_n(0)
// {
//   if (m_nparts <= 0) {
//     throw std::invalid_argument("AdditiveSchwarz: nparts must be > 0");
//   }
//   if (m_overlap < 0) {
//     throw std::invalid_argument("AdditiveSchwarz: overlap must be >= 0");
//   }
// }

// // -----------------------------------------------------------------------------
// // Setup: build non-overlapping partition, then extend each block by overlap,
// // then extract local matrices and factor them.
// // -----------------------------------------------------------------------------
// void AdditiveSchwarz::setup(const MatrixSparse& A)
// {
//   using Index  = MatrixSparse::Index;
//   using Scalar = MatrixSparse::Scalar;

//   // Basic checks
//   m_n = A.rows();
//   if (A.cols() != m_n) {
//     throw std::runtime_error(
//       "AdditiveSchwarz::setup requires a square matrix.");
//   }

//   // 1) Non-overlapping partition: same strategy as BlockJacobi
//   m_starts.resize(static_cast<std::size_t>(m_nparts) + 1);

//   const Index base = m_n / static_cast<Index>(m_nparts);
//   Index rem = m_n % static_cast<Index>(m_nparts);
//   Index pos = 0;
//   for (int p = 0; p < m_nparts; ++p) {
//     m_starts[static_cast<std::size_t>(p)] = static_cast<int>(pos);
//     Index sz = base + (static_cast<Index>(p) < rem ? 1 : 0);
//     pos += sz;
//   }
//   m_starts[static_cast<std::size_t>(m_nparts)] = static_cast<int>(m_n);

//   // 2) Build overlapped ranges [ls_p, le_p)
//   m_localStarts.resize(static_cast<std::size_t>(m_nparts));
//   m_localSizes.resize(static_cast<std::size_t>(m_nparts));

//   const int n_int = static_cast<int>(m_n);

//   for (int p = 0; p < m_nparts; ++p) {
//     const int s = m_starts[static_cast<std::size_t>(p)];
//     const int e = m_starts[static_cast<std::size_t>(p + 1)];

//     // Extend to left/right by m_overlap DOFs (clamped to [0, n))
//     int ls = std::max(0, s - m_overlap);
//     int le = std::min(n_int, e + m_overlap);

//     if (le < ls) {
//       throw std::runtime_error("AdditiveSchwarz::setup: invalid local range");
//     }

//     m_localStarts[static_cast<std::size_t>(p)] = ls;
//     m_localSizes[static_cast<std::size_t>(p)]  = le - ls;
//   }

//   // 3) Allocate and build local dense matrices, then factor them
//   m_LUblocks.clear();
//   m_LUblocks.resize(static_cast<std::size_t>(m_nparts));

//   for (int p = 0; p < m_nparts; ++p) {
//     const int ls = m_localStarts[static_cast<std::size_t>(p)];
//     const int bs = m_localSizes[static_cast<std::size_t>(p)];
//     if (bs <= 0) {
//       m_LUblocks[static_cast<std::size_t>(p)].clear();
//       continue;
//     }

//     auto& blockLU = m_LUblocks[static_cast<std::size_t>(p)];
//     blockLU.assign(static_cast<std::size_t>(bs * bs), 0.0);

//     // Extract submatrix A(Ω_p, Ω_p) where Ω_p = [ls, ls+bs)
//     A.forEachNZ([&](Index r, Index c, Scalar v) {
//       int ri = static_cast<int>(r);
//       int ci = static_cast<int>(c);

//       if (ri >= ls && ri < ls + bs &&
//           ci >= ls && ci < ls + bs)
//       {
//         int lr = ri - ls;  // local row index
//         int lc = ci - ls;  // local col index
//         blockLU[lr * bs + lc] = static_cast<double>(v);
//       }
//     });

//     // Factor local block
//     //lu_factor_block(blockLU, bs);
//   }
// }

// // -----------------------------------------------------------------------------
// // Apply: z = sum_p R_p^T A_p^{-1} R_p r
// //
// // - R_p is restriction to overlapped range Ω_p = [ls_p, ls_p+bs_p)
// // - No weighting: plain additive Schwarz (basic version).
// // -----------------------------------------------------------------------------
// void AdditiveSchwarz::apply(const std::vector<double>& r,
//                             std::vector<double>& z) const
// {
//   if (r.size() != static_cast<std::size_t>(m_n)) {
//     throw std::runtime_error("AdditiveSchwarz::apply: r has wrong size");
//   }

//   // Initialize with zeros; we will ACCUMULATE contributions (because overlap)
//   z.assign(r.size(), 0.0);

//   const int n_int = static_cast<int>(m_n);

//   for (int p = 0; p < m_nparts; ++p) {
//     const int ls = m_localStarts[static_cast<std::size_t>(p)];
//     const int bs = m_localSizes[static_cast<std::size_t>(p)];

//     if (bs <= 0) continue;

//     const auto& LU = m_LUblocks[static_cast<std::size_t>(p)];
//     if (LU.empty()) continue;

//     std::vector<double> rhs(static_cast<std::size_t>(bs));
//     std::vector<double> sol(static_cast<std::size_t>(bs));

//     // Restrict r to subdomain p: rhs_i = r[ls + i]
//     for (int i = 0; i < bs; ++i) {
//       int gi = ls + i;
//       if (gi < 0 || gi >= n_int) {
//         throw std::runtime_error("AdditiveSchwarz::apply: index out of range");
//       }
//       rhs[static_cast<std::size_t>(i)] = r[static_cast<std::size_t>(gi)];
//     }

//     // Solve A_p * sol = rhs using local LU
//     lu_SSOR(LU, bs, rhs.data(), sol.data());

//     // Prolongate/add contribution: z[gi] += sol_i
//     for (int i = 0; i < bs; ++i) {
//       int gi = ls + i;
//       z[static_cast<std::size_t>(gi)] += sol[static_cast<std::size_t>(i)];
//     }
//   }
// }

#include <cmath>
#include <limits>
#include <algorithm>   // std::max, std::min

#include "algebra/matrixSparse.hpp"
#include "preconditioner/additive_schwarz.hpp"

#include <stdexcept>

// -----------------------------------------------------------------------------
// Helper: SSOR on a single sparse local block
// -----------------------------------------------------------------------------
static void ssor_local_block(const AdditiveSchwarz::LocalBlock& blk,
                             const double* b,
                             double*       x,
                             int           sweeps,
                             double        omega)
{
    const int n = blk.bs;
    const auto& rows = blk.rows;

    if (n <= 0) return;

    std::fill(x, x + n, 0.0);
    std::vector<double> x_old(static_cast<std::size_t>(n));

    for (int k = 0; k < sweeps; ++k) {
        // -------------------- Forward sweep --------------------
        std::copy(x, x + n, x_old.data());

        for (int i = 0; i < n; ++i) {
            double sum  = b[i];
            double diag = 0.0;

            // A(i,j) * x_j contributions
            for (const auto& entry : rows[static_cast<std::size_t>(i)]) {
                int    j   = entry.first;
                double aij = entry.second;

                if (j < i) {
                    // lower part: use NEW x_j
                    sum -= aij * x[j];
                } else if (j > i) {
                    // upper part: use OLD x_j
                    sum -= aij * x_old[j];
                } else {
                    // diagonal
                    diag = aij;
                }
            }

            if (std::abs(diag) < 1e-15) {
                throw std::runtime_error(
                    "ssor_local_block (forward): zero/near-zero diagonal");
            }

            double x_gs = sum / diag;                   // Gauss–Seidel update
            x[i] = (1.0 - omega) * x_old[i] + omega * x_gs; // SOR relaxation
        }

        // -------------------- Backward sweep -------------------
        std::copy(x, x + n, x_old.data());

        for (int ii = n - 1; ii >= 0; --ii) {
            int i = ii;
            double sum  = b[i];
            double diag = 0.0;

            for (const auto& entry : rows[static_cast<std::size_t>(i)]) {
                int    j   = entry.first;
                double aij = entry.second;

                if (j > i) {
                    // upper part: use NEW x_j
                    sum -= aij * x[j];
                } else if (j < i) {
                    // lower part: use OLD x_j
                    sum -= aij * x_old[j];
                } else {
                    diag = aij;
                }
            }

            if (std::abs(diag) < 1e-15) {
                throw std::runtime_error(
                    "ssor_local_block (backward): zero/near-zero diagonal");
            }

            double x_gs = sum / diag;
            x[i] = (1.0 - omega) * x_old[i] + omega * x_gs;
        }
    }
}

// -----------------------------------------------------------------------------
// Constructor
// -----------------------------------------------------------------------------
AdditiveSchwarz::AdditiveSchwarz(int nparts, int overlap)
  : m_nparts(nparts),
    m_overlap(overlap),
    m_n(0)
{
    if (m_nparts <= 0) {
        throw std::invalid_argument("AdditiveSchwarz: nparts must be > 0");
    }
    if (m_overlap < 0) {
        throw std::invalid_argument("AdditiveSchwarz: overlap must be >= 0");
    }
}

// -----------------------------------------------------------------------------
// Setup: build non-overlapping partition, then extend by overlap,
// then extract local sparse matrices (blocks) as adjacency lists.
// -----------------------------------------------------------------------------
void AdditiveSchwarz::setup(const MatrixSparse& A)
{
    using Index  = MatrixSparse::Index;
    using Scalar = MatrixSparse::Scalar;

    // Basic checks
    m_n = A.rows();
    if (A.cols() != m_n) {
        throw std::runtime_error(
            "AdditiveSchwarz::setup requires a square matrix.");
    }

    // 1) Non-overlapping partition
    m_starts.resize(static_cast<std::size_t>(m_nparts) + 1);

    const Index base = m_n / static_cast<Index>(m_nparts);
    Index rem        = m_n % static_cast<Index>(m_nparts);
    Index pos        = 0;
    for (int p = 0; p < m_nparts; ++p) {
        m_starts[static_cast<std::size_t>(p)] = static_cast<int>(pos);
        Index sz = base + (static_cast<Index>(p) < rem ? 1 : 0);
        pos += sz;
    }
    m_starts[static_cast<std::size_t>(m_nparts)] = static_cast<int>(m_n);

    // 2) Build overlapped ranges [ls_p, le_p)
    m_localStarts.resize(static_cast<std::size_t>(m_nparts));
    m_localSizes.resize(static_cast<std::size_t>(m_nparts));

    const int n_int = static_cast<int>(m_n);

    for (int p = 0; p < m_nparts; ++p) {
        const int s = m_starts[static_cast<std::size_t>(p)];
        const int e = m_starts[static_cast<std::size_t>(p + 1)];

        int ls = std::max(0,     s - m_overlap);
        int le = std::min(n_int, e + m_overlap);

        if (le < ls) {
            throw std::runtime_error("AdditiveSchwarz::setup: invalid local range");
        }

        m_localStarts[static_cast<std::size_t>(p)] = ls;
        m_localSizes [static_cast<std::size_t>(p)] = le - ls;
    }

    // 3) Allocate and build local sparse adjacency lists
    m_blocks.clear();
    m_blocks.resize(static_cast<std::size_t>(m_nparts));

    for (int p = 0; p < m_nparts; ++p) {
        const int ls = m_localStarts[static_cast<std::size_t>(p)];
        const int bs = m_localSizes [static_cast<std::size_t>(p)];
        auto& blk    = m_blocks      [static_cast<std::size_t>(p)];

        blk.ls = ls;
        blk.bs = bs;

        if (bs <= 0) {
            blk.rows.clear();
            continue;
        }

        blk.rows.assign(static_cast<std::size_t>(bs),
                        std::vector<std::pair<int,double>>{});

        // Extract submatrix A(Ω_p, Ω_p) where Ω_p = [ls, ls+bs)
        A.forEachNZ([&](Index r, Index c, Scalar v) {
            int ri = static_cast<int>(r);
            int ci = static_cast<int>(c);

            if (ri >= ls && ri < ls + bs &&
                ci >= ls && ci < ls + bs)
            {
                int lr = ri - ls;  // local row
                int lc = ci - ls;  // local col
                blk.rows[static_cast<std::size_t>(lr)]
                    .emplace_back(lc, static_cast<double>(v));
            }
        });
    }
}

// -----------------------------------------------------------------------------
// Apply: z = sum_p R_p^T M_p^{-1} R_p r
//
// where M_p^{-1} is approximated by a few SSOR sweeps on each local block.
// -----------------------------------------------------------------------------
void AdditiveSchwarz::apply(const std::vector<double>& r,
                            std::vector<double>&       z) const
{
    if (r.size() != static_cast<std::size_t>(m_n)) {
        throw std::runtime_error("AdditiveSchwarz::apply: r has wrong size");
    }

    // We accumulate contributions in overlapped DOFs
    z.assign(r.size(), 0.0);

    const int n_int = static_cast<int>(m_n);

    std::vector<double> rhs;
    std::vector<double> sol;

    for (int p = 0; p < m_nparts; ++p) {
        const auto& blk = m_blocks[static_cast<std::size_t>(p)];
        const int   ls  = blk.ls;
        const int   bs  = blk.bs;

        if (bs <= 0) continue;

        rhs.resize(static_cast<std::size_t>(bs));
        sol.resize(static_cast<std::size_t>(bs));

        // Restrict r to local RHS
        for (int i = 0; i < bs; ++i) {
            int gi = ls + i;
            if (gi < 0 || gi >= n_int) {
                throw std::runtime_error("AdditiveSchwarz::apply: index out of range");
            }
            rhs[static_cast<std::size_t>(i)] =
                r[static_cast<std::size_t>(gi)];
        }

        // Local approximate solve with SSOR
        ssor_local_block(blk,
                         rhs.data(),
                         sol.data(),
                         m_ssor_sweeps,
                         m_omega);

        // Prolongate/add contribution: z[gi] += sol_i
        for (int i = 0; i < bs; ++i) {
            int gi = ls + i;
            z[static_cast<std::size_t>(gi)] +=
                sol[static_cast<std::size_t>(i)];
        }
    }
}
