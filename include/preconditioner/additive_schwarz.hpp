// preconditioner/additive_schwarz.hpp
#pragma once

#include <vector>
#include <utility>   // std::pair

#include "algebra/matrixSparse.hpp"
#include "preconditioner.hpp"    
#include "algebra/CSR.hpp"          
#include "algebra/COO.hpp"         

// If you have a base class Preconditioner, keep the inheritance as it was.
class AdditiveSchwarz final  : public Preconditioner {
public:
    using MatrixSparse = ::MatrixSparse;    // or your actual alias
    using Index        = MatrixSparse::Index;
    using Scalar       = MatrixSparse::Scalar;
    
    struct LocalBlock {
        int ls = 0;  // global start index of this block
        int bs = 0;  // local size

        // rows[i] = list of (local column j, value a_ij)
        std::vector<std::vector<std::pair<int,double>>> rows;
    };

    AdditiveSchwarz(int nparts, int overlap);

    // Build the preconditioner
    void setup(const MatrixSparse& A);

    // Apply z = M^{-1} r
    void apply(const std::vector<double>& r,
               std::vector<double>&       z) const;

    // Optional: configure local SSOR
    void setSSOR(int sweeps, double omega) {
        m_ssor_sweeps = sweeps;
        m_omega       = omega;
    }

private:


    int         m_nparts   = 0;
    int         m_overlap  = 0;
    std::size_t m_n        = 0; // global size

    // non-overlapping partition (global indices)
    std::vector<int> m_starts;

    // overlapped partition
    std::vector<int> m_localStarts;
    std::vector<int> m_localSizes;

    // sparse local blocks
    std::vector<LocalBlock> m_blocks;

    // SSOR parameters
    int    m_ssor_sweeps = 1;
    double m_omega       = 1.95;
};

