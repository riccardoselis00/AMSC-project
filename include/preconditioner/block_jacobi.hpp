#pragma once

#include <vector>
#include "preconditioner/preconditioner.hpp"
#include "algebra/matrixSparse.hpp"

class BlockJacobi final : public Preconditioner {
public:
    // block_size = size (in DOFs) of each diagonal block
    explicit BlockJacobi(int block_size = 4);

    void setup(const MatrixSparse& A) override;

    void apply(const std::vector<double>& r,
               std::vector<double>& z) const override;

    int block_size() const { return m_blockSize; }
    int num_blocks() const { return m_nparts; }

private:
    int m_blockSize;                    // target block size (e.g. 4, 8, 16)
    int m_n;                            // global size
    int m_nparts;                       // number of blocks

    std::vector<int> m_starts;         // size m_nparts+1, start row of each block
    std::vector<int> m_blockSizes;     // size m_nparts, actual block sizes
    std::vector<std::vector<double>> m_LUblocks; // each block's dense LU
    std::vector<int> m_rowToBlock;     // map row i -> block index p

    // workspaces reused in apply()
    mutable std::vector<double> m_rhs;
    mutable std::vector<double> m_sol;
};
