// preconditioner/additive_schwarz.hpp
#pragma once

#include <vector>
#include <utility>

#include <mpi.h>                     // NEW: MPI-aware AS

#include "algebra/matrixSparse.hpp"
#include "preconditioner.hpp"
#include "algebra/CSR.hpp"
#include "algebra/COO.hpp"

// -----------------------------------------------------------------------------
// Additive Schwarz preconditioner (1-level / 2-level, serial + MPI-aware)
//
// - Works on *local* vectors in MPI: size = n_loc = le - ls
// - Knows global layout: n_global, ls, le
// - 1-level: classical overlapping AS with SSOR local solves
// - 2-level: AS + algebraic coarse operator (one coarse DOF per part)
// -----------------------------------------------------------------------------
class AdditiveSchwarz final : public Preconditioner {
public:
    enum class Level { OneLevel, TwoLevels };

    using MatrixSparse = ::MatrixSparse;
    using Index        = MatrixSparse::Index;
    using Scalar       = MatrixSparse::Scalar;

    // Local sparse block corresponding to one (possibly overlapped)
    // subdomain in *local* index space.
    struct LocalBlock {
        int ls = 0;   // local start index in [0, n_loc)
        int bs = 0;   // block size
        // adjacency list: rows[i] = list of (j, a_ij) in local coordinates
        std::vector<std::vector<std::pair<int, double>>> rows;
    };

    // -------------------------------------------------------------------------
    // Constructors
    // -------------------------------------------------------------------------

    // MPI-aware constructor:
    //
    //  - n_global : total number of DOFs in the global problem
    //  - ls, le   : global index range owned by this rank [ls, le)
    //  - nparts   : number of subdomains (parts) per rank (in *local* index space)
    //  - overlap  : integer overlap (in local indices) around each non-overlap block
    //  - comm     : MPI communicator
    //  - level    : OneLevel or TwoLevels (with coarse operator)
    AdditiveSchwarz(int      n_global,
                    int      ls,
                    int      le,
                    int      nparts,
                    int      overlap,
                    MPI_Comm comm,
                    Level    level = Level::OneLevel);

    // Convenience ctor for serial tests: behaves like your original
    // AdditiveSchwarz(nparts, overlap) API. We set up a "fake" MPI world
    // (MPI_COMM_SELF) and let setup() infer n_global from A.rows().
    AdditiveSchwarz(int nparts,
                    int overlap,
                    Level level = Level::OneLevel);

    // -------------------------------------------------------------------------
    // Setup & Apply
    // -------------------------------------------------------------------------

    // Setup on (local) matrix A.
    //
    // SERIAL:
    //   - A is the full global matrix
    //   - this will infer n_global, ls=0, le=n_global, n_loc = n_global
    //
    // MPI:
    //   - A is the local matrix for rows [ls, le)
    //   - local partitions & blocks are built in local index space [0, n_loc)
    //   - global coarse operator A0 is assembled via local contributions + MPI
    void setup(const MatrixSparse& A);

    // Apply preconditioner: z = M^{-1} r
    //
    // SERIAL:
    //   - r, z are global-size vectors
    //
    // MPI:
    //   - r, z are *local* vectors of size n_loc = le - ls
    //   - PCGSolverMPI passes local residual and receives local correction
    void apply(const std::vector<double>& r,
               std::vector<double>&       z) const override;

    // Optional: tweak SSOR parameters
    void setSSORSweeps(int sweeps) { m_ssor_sweeps = sweeps; }
    void setOmega(double omega)    { m_omega = omega; }

private:
    // -------------------------------------------------------------------------
    // Basic AS data
    // -------------------------------------------------------------------------
    int   m_nparts;    // number of subdomains per rank
    int   m_overlap;   // overlap size in local indices

    Index m_n = 0;     // local matrix size (A.rows()) used inside setup

    // Non-overlapping partition of local index space [0, n_loc)
    // m_starts[p]..m_starts[p+1] defines the non-overlap block for part p
    std::vector<int> m_starts;       // size m_nparts + 1

    // Overlapped ranges [m_localStarts[p], m_localStarts[p]+m_localSizes[p])
    // for each local block in local index space
    std::vector<int> m_localStarts;  // size m_nparts
    std::vector<int> m_localSizes;   // size m_nparts

    // Local sparse blocks (one per part)
    std::vector<LocalBlock> m_blocks;

    // SSOR parameters for local block solves
    int    m_ssor_sweeps = 1;
    double m_omega       = 1.95;

    // -------------------------------------------------------------------------
    // MPI + geometry info
    // -------------------------------------------------------------------------
    MPI_Comm m_comm    = MPI_COMM_SELF;
    int      m_rank    = 0;
    int      m_size    = 1;

    int   m_n_global = 0;  // total number of DOFs
    int   m_ls       = 0;  // first global index owned by this rank
    int   m_le       = 0;  // one-past-last global index owned by this rank
    int   m_n_loc    = 0;  // local number of DOFs (m_le - m_ls)

    Level m_level    = Level::OneLevel;

    // -------------------------------------------------------------------------
    // Coarse space (2-level AS)
    // -------------------------------------------------------------------------
    int m_ncoarse = 0;                 // number of coarse DOFs (usually == m_nparts)

    // Global mapping: for each global row i -> which coarse DOF (part) it belongs to
    // size = m_n_global
    std::vector<int> m_rowToPart;

    // Coarse operator A0 (dense, row-major, size m_ncoarse x m_ncoarse)
    std::vector<double> m_A0;

    // Coarse residual and solution:
    //
    //  - m_r0_local: local contribution on this rank
    //  - m_r0      : global coarse residual after MPI_Allreduce
    //  - m_y0      : coarse solution A0^{-1} r0 (replicated on all ranks)
    mutable std::vector<double> m_r0_local;
    mutable std::vector<double> m_r0;
    mutable std::vector<double> m_y0;
};
