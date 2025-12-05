#include <cmath>
#include <limits>
#include <algorithm>   // std::max, std::min

#include "algebra/matrixSparse.hpp"
#include "preconditioner/additive_schwarz.hpp"

#include <mpi.h>  

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
// -----------------------------------------------------------------------------
// Main constructor: MPI-aware (works also for serial when n_global==0)
// -----------------------------------------------------------------------------
AdditiveSchwarz::AdditiveSchwarz(int      n_global,
                                 int      ls,
                                 int      le,
                                 int      nparts,
                                 int      overlap,
                                 MPI_Comm comm,
                                 Level    level)
  : m_nparts(nparts),
    m_overlap(overlap),
    m_n(0),              // we'll set this to local size in setup()
    m_level(level),
    m_comm(comm),
    m_rank(0),
    m_size(1),
    m_n_global(n_global),
    m_ls(ls),
    m_le(le),
    m_n_loc(le - ls)
{
    if (m_nparts <= 0)
        throw std::invalid_argument("AdditiveSchwarz: nparts must be > 0");
    if (m_overlap < 0)
        throw std::invalid_argument("AdditiveSchwarz: overlap must be >= 0");

    MPI_Comm_rank(m_comm, &m_rank);
    MPI_Comm_size(m_comm, &m_size);
}

// -----------------------------------------------------------------------------
// Convenience ctor for *serial* tests: same API you had before
// -----------------------------------------------------------------------------
AdditiveSchwarz::AdditiveSchwarz(int nparts, int overlap, Level level)
  : AdditiveSchwarz(/*n_global=*/0,
                    /*ls=*/0,
                    /*le=*/0,
                    nparts,
                    overlap,
                    MPI_COMM_SELF,   // single-process communicator
                    level)
{}



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

    // m_rowToPart.assign(static_cast<std::size_t>(m_n), -1);
    // for (int p = 0; p < m_nparts; ++p) {
    //     int s = m_starts[static_cast<std::size_t>(p)];
    //     int e = m_starts[static_cast<std::size_t>(p + 1)];
    //     for (int i = s; i < e; ++i) {
    //         m_rowToPart[static_cast<std::size_t>(i)] = p;
    //     }
    // }

    // Build *global* row -> part mapping for coarse space
    m_rowToPart.assign(static_cast<std::size_t>(m_n_global), -1);

    const int nG   = m_n_global;
    const int baseG = nG / m_nparts;
    int       remG  = nG % m_nparts;
    int       posG  = 0;

    for (int p = 0; p < m_nparts; ++p) {
        int s = posG;
        int sz = baseG + (p < remG ? 1 : 0);
        int e = s + sz;
        for (int i = s; i < e; ++i) {
            m_rowToPart[static_cast<std::size_t>(i)] = p;
        }
        posG = e;
    }

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

    // // NEW: build coarse operator if two-level AS is requested
    // if (m_level == Level::TwoLevels) {
    //     // ncoarse = number of parts
    //     m_ncoarse = m_nparts;

    //     // Dense ncoarse x ncoarse matrix, row-major, initialized to 0
    //     m_A0.assign(static_cast<std::size_t>(m_ncoarse * m_ncoarse), 0.0);

    //     // Assemble A0(i,j) = sum_{r in part i, c in part j} A(r,c)
    //     A.forEachNZ([&](Index r, Index c, Scalar v) {
    //         int ri = static_cast<int>(r);
    //         int ci = static_cast<int>(c);

    //         int pi = m_rowToPart[static_cast<std::size_t>(ri)];
    //         int pj = m_rowToPart[static_cast<std::size_t>(ci)];

    //         // accumulate into dense coarse matrix
    //         m_A0[static_cast<std::size_t>(pi * m_ncoarse + pj)]
    //             += static_cast<double>(v);
    //     });

    //     // Allocate work vectors for coarse residual and solution
    //     m_r0.assign(static_cast<std::size_t>(m_ncoarse), 0.0);
    //     m_y0.assign(static_cast<std::size_t>(m_ncoarse), 0.0);



        // (Optional but recommended)
        // You can factorize m_A0 in-place once here with a simple Cholesky
        // and then use it in apply(). For now we can keep a simple dense solve
        // inside solveCoarse().


            // 4) Build coarse operator A0 if two-level AS is requested
    if (m_level == Level::TwoLevels) {
        // One coarse DOF per (global) part
        m_ncoarse = m_nparts;

        // Local contribution to A0
        std::vector<double> A0_local(
            static_cast<std::size_t>(m_ncoarse * m_ncoarse), 0.0);

        // A is local; global row/col = m_ls + local row/col
        A.forEachNZ([&](Index r, Index c, Scalar v) {
            int i_loc = static_cast<int>(r);
            int j_loc = static_cast<int>(c);

            int gi = m_ls + i_loc;  // global row index
            int gj = m_ls + j_loc;  // global col index (assuming simple dist)

            int pi = m_rowToPart[static_cast<std::size_t>(gi)];
            int pj = m_rowToPart[static_cast<std::size_t>(gj)];

            A0_local[static_cast<std::size_t>(pi * m_ncoarse + pj)]
                += static_cast<double>(v);
        });

        // Global coarse matrix A0 = sum over ranks of A0_local
        m_A0.assign(static_cast<std::size_t>(m_ncoarse * m_ncoarse), 0.0);
        MPI_Allreduce(A0_local.data(), m_A0.data(),
                      m_ncoarse * m_ncoarse,
                      MPI_DOUBLE, MPI_SUM, m_comm);

        // Allocate coarse residual and solution
        m_r0_local.assign(static_cast<std::size_t>(m_ncoarse), 0.0);
        m_r0      .assign(static_cast<std::size_t>(m_ncoarse), 0.0);
        m_y0      .assign(static_cast<std::size_t>(m_ncoarse), 0.0);

        // You may optionally factorize m_A0 here once for speed.
    }
}




// -----------------------------------------------------------------------------
// Helper: naive dense solve A0 * x = b for coarse system
// (small ncoarse, so we can afford a simple Gaussian elimination)
// -----------------------------------------------------------------------------
static void solve_dense_sym_pos(const std::vector<double>& A,
                                int                        n,
                                const double*              b,
                                double*                    x)
{
    // Very simple (and not super robust) Cholesky-like approach for SPD-ish A.
    // For a real project, replace with a proper LAPACK binding or your own
    // factorization. For now, it's OK as a toy coarse solver.

    // Copy A into a working matrix
    std::vector<double> M(A);

    // Forward factorization: Cholesky (lower triangle)
    for (int k = 0; k < n; ++k) {
        double diag = M[k*n + k];
        for (int s = 0; s < k; ++s) {
            double Lks = M[k*n + s];
            diag -= Lks * Lks;
        }
        if (diag <= 1e-14) diag = 1e-14; // crude regularization
        double Lkk = std::sqrt(diag);
        M[k*n + k] = Lkk;

        for (int i = k + 1; i < n; ++i) {
            double Lik = M[i*n + k];
            for (int s = 0; s < k; ++s) {
                Lik -= M[i*n + s] * M[k*n + s];
            }
            M[i*n + k] = Lik / Lkk;
        }
    }

    // Forward substitution: solve L y = b
    std::vector<double> y(n, 0.0);
    for (int i = 0; i < n; ++i) {
        double sum = b[i];
        for (int j = 0; j < i; ++j) {
            sum -= M[i*n + j] * y[j];
        }
        y[i] = sum / M[i*n + i];
    }

    // Backward substitution: solve L^T x = y
    for (int i = n - 1; i >= 0; --i) {
        double sum = y[i];
        double Lii = M[i*n + i];
        for (int j = i + 1; j < n; ++j) {
            sum -= M[j*n + i] * x[j];
        }
        x[i] = sum / Lii;
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
    // r is local residual of size n_loc
    if (r.size() != static_cast<std::size_t>(m_n_loc)) {
        throw std::runtime_error("AdditiveSchwarz::apply: local r has wrong size");
    }

    z.assign(r.size(), 0.0);

    const int n_int = m_n_loc; // local size

    std::vector<double> rhs;
    std::vector<double> sol;

    for (int p = 0; p < m_nparts; ++p) {
        const auto& blk = m_blocks[static_cast<std::size_t>(p)];
        const int   ls  = blk.ls;   // local start
        const int   bs  = blk.bs;

        if (bs <= 0) continue;

        rhs.resize(static_cast<std::size_t>(bs));
        sol.resize(static_cast<std::size_t>(bs));

        // Restrict local r to local block
        for (int i = 0; i < bs; ++i) {
            int li = ls + i;
            if (li < 0 || li >= n_int) {
                throw std::runtime_error("AdditiveSchwarz::apply: index out of range");
            }
            rhs[static_cast<std::size_t>(i)] =
                r[static_cast<std::size_t>(li)];
        }

        // Local approximate solve with SSOR
        ssor_local_block(blk,
                         rhs.data(),
                         sol.data(),
                         m_ssor_sweeps,
                         m_omega);

        // Prolongate/add local contribution
        for (int i = 0; i < bs; ++i) {
            int li = ls + i;
            z[static_cast<std::size_t>(li)] +=
                sol[static_cast<std::size_t>(i)];
        }
    }

    // -------------------- Coarse correction (2-level AS) --------------------
    if (m_level == Level::TwoLevels) {
        if (m_ncoarse != m_nparts) {
            throw std::runtime_error("AdditiveSchwarz::apply: invalid coarse size");
        }

        // 1) Local coarse residual r0_local[p] = sum_{gi in part p} r_loc[gi]
        std::fill(m_r0_local.begin(), m_r0_local.end(), 0.0);

        for (int i_loc = 0; i_loc < m_n_loc; ++i_loc) {
            int gi = m_ls + i_loc;   // global index of this local DOF
            int p  = m_rowToPart[static_cast<std::size_t>(gi)];
            m_r0_local[static_cast<std::size_t>(p)] +=
                r[static_cast<std::size_t>(i_loc)];
        }

        // 2) Global coarse residual r0 = sum over ranks of r0_local
        MPI_Allreduce(m_r0_local.data(), m_r0.data(),
                      m_ncoarse, MPI_DOUBLE, MPI_SUM, m_comm);

        // 3) Solve coarse system A0 y0 = r0
        //    (solve on each rank; A0 is small, this is cheap)
        solve_dense_sym_pos(m_A0, m_ncoarse,
                            m_r0.data(), m_y0.data());

        // 4) Prolong coarse solution back to local fine grid
        //    z[i_loc] += y0[ part(global_i) ]
        for (int i_loc = 0; i_loc < m_n_loc; ++i_loc) {
            int gi = m_ls + i_loc;
            int p  = m_rowToPart[static_cast<std::size_t>(gi)];
            z[static_cast<std::size_t>(i_loc)] +=
                m_y0[static_cast<std::size_t>(p)];
        }
    }
}

