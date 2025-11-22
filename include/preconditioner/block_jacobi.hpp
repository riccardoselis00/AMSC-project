#pragma once

#include <vector>
#include <stdexcept>
#include <cstddef>

#include "preconditioner.hpp"    
#include "algebra/matrixSparse.hpp" 
#include "algebra/CSR.hpp"          
#include "algebra/COO.hpp"      

class BlockJacobi final : public Preconditioner {
public:

  explicit BlockJacobi(int nparts);

  void setup(const MatrixSparse& A);
  void apply(const std::vector<double>& r, std::vector<double>& z) const;

private:
  int m_nparts;              
  Index m_n; 

  std::vector<int> m_starts;

  std::vector<int>                 m_blockSizes;
  std::vector<std::vector<double>> m_LUblocks; 

};