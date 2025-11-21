#include <vector>
#include <stdexcept>
#include <cstddef>

#include "preconditioner.hpp"    
#include "algebra/matrixSparse.hpp" 
#include "algebra/CSR.hpp"          
#include "algebra/COO.hpp"          


namespace algebra {
class MatrixSparse;
class CSR;
class COO;
} 


namespace preconditioner {

class BlockJacobi final : public Preconditioner {
public:

  explicit BlockJacobi(int nparts = 1);

  void setup(const algebra::MatrixSparse& A);

  void apply(const std::vector<double>& r, std::vector<double>& z) const;

  int parts() const { return m_nparts; }

  const std::vector<int>& block_starts() const { return m_starts; }

private:
  static double diag_at(const algebra::MatrixSparse& A, int i);

  int m_nparts;                 
  int m_n;                      
  std::vector<int> m_starts;    
  std::vector<double> m_inv_diag; 


}} 