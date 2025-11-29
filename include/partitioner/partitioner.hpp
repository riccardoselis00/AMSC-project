#pragma once

#include <vector>
#include <mpi.h>

#include "algebra/matrixSparse.hpp"  // adjust to your actual path/name
#include "algebra/COO.hpp"          // adjust to your actual path/name

// BlockRowPartitioner
//
// Central utility for domain decomposition in the row-wise (block-row) sense.
// It knows:
//  - n_global         : total number of unknowns
//  - rank, size       : MPI info
//  - ls, le           : local range [ls, le) of global row indices
//  - rowStarts        : full partition [0..n_global] split among ranks
//
// It also provides helpers to:
//  - build the local matrix A_loc and RHS b_loc from a global A,b
//    (in the first version, assuming A,b exist on each rank)
//  - extract local slices of any global vector
//  - gather a distributed vector back to root.
class BlockRowPartitioner
{
public:
    // Constructor: compute partition of [0, n_global) among ranks in 'comm'
    BlockRowPartitioner(int n_global, MPI_Comm comm);

    // Basic info
    int nGlobal()   const { return n_global_; }
    int nLocal()    const { return le_ - ls_; }
    int size()      const { return size_; }
    int rank()      const { return rank_; }
    int ls()        const { return ls_; }     // first global row index [ls, le)
    int le()        const { return le_; }     // one past last
    MPI_Comm comm() const { return comm_; }

    // Full partition: rowStarts[p..p+1) is the range of rows for rank p
    const std::vector<int>& rowStarts() const { return rowStarts_; }

    // ------------------------------------------------------------------
    // Build local system (A_loc, b_loc) from global (A_global, b_global)
    //
    // For now this assumes A_global and b_global are available on *each*
    // rank. This keeps the logic simple. Later you can change only this
    // function to do a proper distribution from rank 0.
    //
    // A_loc will be resized/constructed inside this method.
    // b_loc will be resized accordingly.
    // ------------------------------------------------------------------
    void buildLocalSystem(const MatrixSparse&         A_global,
                          const std::vector<double>&  b_global,
                          MatrixCOO&               A_loc,
                          std::vector<double>&        b_loc) const;

    // Extract just the local slice of a global vector v_global -> v_loc
    void extractLocalVector(const std::vector<double>& v_global,
                            std::vector<double>&       v_loc) const;

    // Gather a distributed vector x_loc (one chunk per rank) to root.
    // On non-root ranks, x_global is left unchanged (you can ignore it).
    void gatherVectorToRoot(const std::vector<double>& x_loc,
                            std::vector<double>&       x_global,
                            int                        root = 0) const;

private:
    int n_global_   = 0;
    int size_       = 0;
    int rank_       = 0;
    int ls_         = 0;
    int le_         = 0;
    std::vector<int> rowStarts_;
    MPI_Comm comm_  = MPI_COMM_NULL;
};
