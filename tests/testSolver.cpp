#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <string>

#include "algebra/COO.hpp"
#include "algebra/CSR.hpp"
#include "solver/pcg.hpp"
#include "preconditioner/preconditioner.hpp"
#include "preconditioner/identity.hpp"
#include "../tests/test_util.hpp"

using expect_true;
using expect_eq;
using expect_throw;
using expect_near;

using algebra::MatrixCOO;
using algebra::MatrixCSR;
using solver::PCGSolver;
using preconditioner::IdentityPreconditioner;

static void testIdentityPreconditionerApply()
{
    IdentityPreconditioner M;
    std::vector<double> r = {1.0, -2.0, 3.0}, z(3, 0.0);
    M.apply(r, z);
    expect_eq(z[0], 1.0, __func__, "z[0]", "1.0");
    expect_eq(z[1], -2.0, __func__, "z[1]", "-2.0");
    expect_eq(z[2], 3.0, __func__, "z[2]", "3.0");
}

static void testIdentityPreconditionerUpdate()
{
    MatrixCOO coo(2,2);
    coo.add(0,0,1.0);
    coo.add(1,1,1.0);
    MatrixCSR A(coo);
    IdentityPreconditioner M;
    // no-throw
    M.update(A);
    expect_true(true, __func__, "update no-throw");
}

static void testPCGSolverSolve2x2()
{
    // A = [[4,1],[1,3]] (SPD), b = [1,2]
    MatrixCOO coo(2,2);
    coo.add(0,0,4); coo.add(0,1,1);
    coo.add(1,0,1); coo.add(1,1,3);
    MatrixCSR A(coo);
    std::vector<double> b = {1.0, 2.0};
    std::vector<double> x(2, 0.0);

    IdentityPreconditioner M;
    PCGSolver solver(A, &M);
    solver.setMaxIters(100);
    solver.setTolerance(1e-12);
    auto its = solver.solve(b, x);
    (void)its;

    // exact solution
    const double x0_exp = 1.0/11.0;
    const double x1_exp = 7.0/11.0;
    expect_near(x[0], x0_exp, 1e-10, __func__, "x[0]", "1/11");
    expect_near(x[1], x1_exp, 1e-10, __func__, "x[1]", "7/11");
}

static void testPCGSolverZeroRHS()
{
    MatrixCOO coo(2,2);
    coo.add(0,0,4); coo.add(0,1,1);
    coo.add(1,0,1); coo.add(1,1,3);
    MatrixCSR A(coo);
    std::vector<double> b = {0.0, 0.0};
    std::vector<double> x(2, 0.0);

    IdentityPreconditioner M;
    PCGSolver solver(A, &M);
    solver.setMaxIters(10);
    solver.setTolerance(1e-14);
    auto its = solver.solve(b, x);
    expect_eq(its, (std::size_t)0, __func__, "its", "0");
    expect_near(x[0], 0.0, 1e-14, __func__, "x[0]", "0.0");
    expect_near(x[1], 0.0, 1e-14, __func__, "x[1]", "0.0");
}

static void testPCGSolverDimensionMismatch()
{
    MatrixCOO coo(2,2);
    coo.add(0,0,2); coo.add(1,1,2);
    MatrixCSR A(coo);
    std::vector<double> b = {1.0, 2.0, 3.0}; // wrong size
    std::vector<double> x(2, 0.0);
    IdentityPreconditioner M;
    PCGSolver solver(A, &M);
    expect_throw([&]{ (void)solver.solve(b, x); }, __func__, "dimension mismatch should throw");
}

static void testPCGSolverNonSPD()
{
    // Skew-symmetric => guaranteed breakdown (p^T A p == 0)
    MatrixCOO coo(2,2);
    coo.add(0,1, 1.0);
    coo.add(1,0,-1.0);
    MatrixCSR A(coo);
    std::vector<double> b = {1.0, 2.0};
    std::vector<double> x(2, 0.0);
    IdentityPreconditioner M;
    PCGSolver solver(A, &M);
    solver.setMaxIters(5);
    expect_throw([&]{ (void)solver.solve(b, x); }, __func__, "PCG on non-SPD should throw");
}

static void testPCGSolverMaxIterations()
{
    // SPD but set maxIters=1 to force early stop
    MatrixCOO coo(2,2);
    coo.add(0,0,4); coo.add(0,1,1);
    coo.add(1,0,1); coo.add(1,1,3);
    MatrixCSR A(coo);
    std::vector<double> b = {1.0, 2.0};
    std::vector<double> x(2, 0.0);
    IdentityPreconditioner M;
    PCGSolver solver(A, &M);
    solver.setMaxIters(1);
    solver.setTolerance(1e-30);
    auto its = solver.solve(b, x);
    expect_eq(its, (std::size_t)1, __func__, "its", "1");
}

int main()
{
    testIdentityPreconditionerApply();
    testIdentityPreconditionerUpdate();
    testPCGSolverSolve2x2();
    testPCGSolverZeroRHS();
    testPCGSolverDimensionMismatch();
    testPCGSolverNonSPD();
    testPCGSolverMaxIterations();
    return ddtest::summarize_and_exit();
}