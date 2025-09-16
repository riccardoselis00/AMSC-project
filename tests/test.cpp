#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <stdexcept>
#include <sstream>   // for std::stringstream
#include <cstdio>    // for std::remove
#include <cstdlib>   // for std::rand
#include <stdlib.h>

#include "dd/algebra/COO.hpp"
#include "dd/algebra/CSR.hpp"
#include "dd/solver/pcg.hpp"
#include "dd/preconditioner/preconditioner.hpp"
#include "dd/preconditioner/identity.hpp"
#include "../tests/test_util.hpp"

/* The lines `using dd::algebra::MatrixCOO;`, `using dd::algebra::MatrixCSR;`, `using
dd::algebra::PCGSolver;`, and `using dd::algebra::IdentityPreconditioner;` are using declarations in
C++. */
using dd::algebra::MatrixCOO;
using dd::algebra::MatrixCSR;
using dd::algebra::PCGSolver;
using dd::algebra::IdentityPreconditioner;


int main()
{

    std::cout << "Read The Matrix!" << std::endl;   
    
    MatrixCOO A = MatrixCOO::read_COO("../tests/input/out.mtx");

    //std::vector<double> b(A.rows(), 1.0);

    printf("Matrix read: %zu x %zu, nnz=%zu\n", A.rows(), A.cols(), A.nnz());

    std::cout << "Write the Matrix!" << std::endl;

    if (!A.write_COO("out.mtx")) {
        std::fprintf(stderr, "Failed to write out.mtx\n");
        return 1;
    }

    std::cout << "Solve The Linear System!" << std::endl;

    std::vector<double> b(A.rows(), 1.0);
    std::vector<double> x(A.cols(), 0.0);
    IdentityPreconditioner M;

    PCGSolver solver(A, &M);
    solver.setMaxIters(1000);
    solver.setTolerance(1e-12);
    auto its = solver.solve(b, x);

    std::cout << "Solver finished in " << its << " iterations." << std::endl;

    std::cout << "Solution x entries" << std::endl;
    for (size_t i = 0; i < std::min((size_t)10, x.size()); ++i) {
        std::cout << "x[" << i << "] = " << x[i] << std::endl;
    }

    return 0;
}