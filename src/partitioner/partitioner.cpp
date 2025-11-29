#include "partitioner/partitioner.hpp"
#include <vector>
#include <mpi.h>
#include "algebra/matrixSparse.hpp"
#include "algebra/COO.hpp"

#include <stdexcept>

// -----------------------------------------------------------------------------
// Constructor: simple block-row partition
// -----------------------------------------------------------------------------
BlockRowPartitioner::BlockRowPartitioner(int n_global, MPI_Comm comm)
  : n_global_(n_global),
    comm_(comm)
{
    if (n_global_ <= 0) {
        throw std::invalid_argument("BlockRowPartitioner: n_global must be > 0");
    }
    if (comm_ == MPI_COMM_NULL) {
        throw std::invalid_argument("BlockRowPartitioner: MPI_Comm is MPI_COMM_NULL");
    }

    MPI_Comm_size(comm_, &size_);
    MPI_Comm_rank(comm_, &rank_);

    if (size_ <= 0) {
        throw std::runtime_error("BlockRowPartitioner: MPI_Comm_size returned <= 0");
    }

    rowStarts_.resize(size_ + 1);

    const int base = n_global_ / size_;
    const int rem  = n_global_ % size_;

    int pos = 0;
    for (int p = 0; p < size_; ++p) {
        rowStarts_[p] = pos;
        const int n_p = base + (p < rem ? 1 : 0);
        pos += n_p;
    }
    rowStarts_[size_] = n_global_;

    ls_ = rowStarts_[rank_];
    le_ = rowStarts_[rank_ + 1];
}

// -----------------------------------------------------------------------------
// buildLocalSystem: from A_global, b_global -> A_loc, b_loc for THIS rank
// -----------------------------------------------------------------------------
// void BlockRowPartitioner::buildLocalSystem(const MatrixSparse&         A_global,
//                                            const std::vector<double>&  b_global,
//                                            MatrixCOO&               A_loc,
//                                            std::vector<double>&        b_loc) const
// {
//     if (A_global.rows() != n_global_ || static_cast<int>(b_global.size()) != n_global_) {
//         throw std::runtime_error("BlockRowPartitioner::buildLocalSystem: "
//                                  "global sizes do not match n_global");
//     }

//     const int n_loc = nLocal();

//     // Construct local matrix with n_loc rows and same number of cols as global.
//     A_loc = MatrixSparse(n_loc, A_global.cols());

//     using Index  = MatrixSparse::Index;
//     using Scalar = MatrixSparse::Scalar;

//     // Copy only rows in [ls_, le_)
//     A_global.forEachNZ([&](Index i, Index j, Scalar v) {
//         const int gi = static_cast<int>(i);
//         if (gi >= ls_ && gi < le_) {
//             const int i_loc = gi - ls_;
//             A_loc.addValue(i_loc,
//                            static_cast<int>(j),
//                            static_cast<double>(v));
//         }
//     });

//     A_loc.finalize();

//     // Extract local RHS
//     extractLocalVector(b_global, b_loc);
// }

void BlockRowPartitioner::buildLocalSystem(const MatrixSparse&        A_global,
                                           const std::vector<double>& b_global,
                                           MatrixCOO&                 A_loc,
                                           std::vector<double>&       b_loc) const
{
    using Index  = MatrixSparse::Index;
    using Scalar = MatrixSparse::Scalar;

    // Basic size checks (careful with signed/unsigned)
    if (A_global.rows() != static_cast<Index>(n_global_) ||
        b_global.size() != static_cast<std::size_t>(n_global_)) {
        throw std::runtime_error(
            "BlockRowPartitioner::buildLocalSystem: "
            "global sizes do not match n_global");
    }

    const int n_loc = nLocal();
    const int ls    = ls_;
    const int le    = le_;

    // Rebuild A_loc as a fresh COO of shape (n_loc x n_cols)
    // (this assumes MatrixCOO has this constructor and a working operator=)
    A_loc = MatrixCOO(static_cast<Index>(n_loc), A_global.cols());

    // Optional: pre-reserve some nnz to avoid many reallocs
    // Crude estimate: assume uniform distribution of nnz per row
    Index nnz_est = A_global.nnz() / std::max(size_, 1);
    if (nnz_est == 0) nnz_est = 1;
    A_loc.reserve(static_cast<std::size_t>(nnz_est));

    // Copy only rows in [ls, le)
    A_global.forEachNZ([&](Index i, Index j, Scalar v) {
        const int gi = static_cast<int>(i);
        if (gi >= ls && gi < le) {
            const int i_loc = gi - ls;
            A_loc.add(static_cast<Index>(i_loc), j, v);
        }
    });

    // Local RHS: just slice b_global
    extractLocalVector(b_global, b_loc);
}


// -----------------------------------------------------------------------------
// extractLocalVector: v_global -> v_loc for rows [ls_, le_)
// -----------------------------------------------------------------------------
void BlockRowPartitioner::extractLocalVector(const std::vector<double>& v_global,
                                             std::vector<double>&       v_loc) const
{
    if (static_cast<int>(v_global.size()) != n_global_) {
        throw std::runtime_error("BlockRowPartitioner::extractLocalVector: "
                                 "global vector size does not match n_global");
    }

    const int n_loc = nLocal();
    v_loc.resize(n_loc);

    for (int i = 0; i < n_loc; ++i) {
        int gi = ls_ + i;
        v_loc[static_cast<std::size_t>(i)] =
            v_global[static_cast<std::size_t>(gi)];
    }
}

// -----------------------------------------------------------------------------
// gatherVectorToRoot: gather distributed vector x_loc back to root
// -----------------------------------------------------------------------------
void BlockRowPartitioner::gatherVectorToRoot(const std::vector<double>& x_loc,
                                             std::vector<double>&       x_global,
                                             int                        root) const
{
    const int n_loc = nLocal();
    if (static_cast<int>(x_loc.size()) != n_loc) {
        throw std::runtime_error("BlockRowPartitioner::gatherVectorToRoot: "
                                 "local vector size does not match nLocal()");
    }

    // Counts and displacements for Gatherv
    std::vector<int> counts(size_), displs(size_);
    for (int p = 0; p < size_; ++p) {
        counts[p] = rowStarts_[p + 1] - rowStarts_[p];
        displs[p] = rowStarts_[p];
    }

    if (rank_ == root) {
        x_global.resize(n_global_);
    }

    MPI_Gatherv(x_loc.data(), n_loc, MPI_DOUBLE,
                rank_ == root ? x_global.data() : nullptr,
                counts.data(), displs.data(), MPI_DOUBLE,
                root, comm_);
}
