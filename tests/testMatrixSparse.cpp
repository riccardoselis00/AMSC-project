#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <stdexcept>
#include <cstdio>

#include "dd/algebra/matrixSparse.hpp"
#include "dd/algebra/COO.hpp"
#include "dd/algebra/CSR.hpp"
#include "../tests/test_util.hpp"

using ddtest::expect_true;
using ddtest::expect_eq;
using ddtest::expect_near;

using dd::algebra::MatrixCOO;
using dd::algebra::MatrixCSR;

static MatrixCOO build_coo_example()
{
    MatrixCOO A(2,3);
    A.add(0,0,1.0);
    A.add(1,0,3.0);
    A.add(0,1,5.0);
    A.add(1,1,2.0);
    A.add(0,2,4.0);
    A.add(1,2,6.0);
    return A;
}

static MatrixCSR build_csr_example()
{
    return MatrixCSR(build_coo_example());
}

// ------------------- COO tests -------------------

static void test_coo_gemv_vectors()
{
    MatrixCOO A = build_coo_example();
    std::vector<double> x = {1.0, 2.0, 3.0};
    std::vector<double> y(2, 0.0);
    A.gemv(x, y, 1.0, 0.0);
    expect_near(y[0], 1.0*1 + 2.0*5 + 3.0*4, 1e-12, __func__, "y[0]", "dot row0");
    expect_near(y[1], 1.0*3 + 2.0*2 + 3.0*6, 1e-12, __func__, "y[1]", "dot row1");
}

static void test_coo_extractDiagonal()
{
    MatrixCOO A = build_coo_example();
    std::vector<double> d;
    A.extractDiagonal(d);
    expect_eq(d.size(), (std::size_t)2, __func__, "diag size", "2");
    expect_eq(d[0], 1.0, __func__, "d[0]", "1");
    expect_eq(d[1], 2.0, __func__, "d[1]", "2");
}

static void test_coo_norms()
{
    MatrixCOO A = build_coo_example();
    expect_near(A.frobeniusNorm(), std::sqrt(1+9+25+4+16+36), 1e-12, __func__, "||A||_F", "sqrt(91)");
    //expect_near(A.normInf(),  std::max(1.0+5.0+4.0, 3.0+2.0+6.0), 1e-12, __func__, "||A||_inf", "row max sum");
}

static void test_coo_axpy()
{
    MatrixCOO A = build_coo_example();
    MatrixCOO B = build_coo_example();
    A.axpy(2.0, B); // A := A + 2B
    std::vector<double> x = {1,2,3}, yA(2,0), yB(2,0);
    A.gemv(x, yA, 1.0, 0.0);
    B.gemv(x, yB, 3.0, 0.0); // 1*A + 2*B vs 3*B
    expect_near(yA[0], yB[0], 1e-12, __func__, "axpy row0", "match");
    expect_near(yA[1], yB[1], 1e-12, __func__, "axpy row1", "match");
}

static void test_coo_to_csr_and_back()
{
    MatrixCOO A = build_coo_example();
    MatrixCSR C(A);
    MatrixCOO B(C);
    std::vector<double> x = {1,2,3}, yA(2,0), yB(2,0);
    A.gemv(x, yA, 1.0, 0.0);
    B.gemv(x, yB, 1.0, 0.0);
    expect_near(yA[0], yB[0], 1e-12, __func__, "row0", "match");
    expect_near(yA[1], yB[1], 1e-12, __func__, "row1", "match");
}

// ------------------- CSR tests -------------------

static void test_csr_gemv_vectors()
{
    MatrixCSR A = build_csr_example();
    std::vector<double> x = {1.0, 2.0, 3.0};
    std::vector<double> y(2, 0.0);
    A.gemv(x, y, 1.0, 0.0);
    expect_near(y[0], 1.0*1 + 2.0*5 + 3.0*4, 1e-12, __func__, "y[0]", "dot row0");
    expect_near(y[1], 1.0*3 + 2.0*2 + 3.0*6, 1e-12, __func__, "y[1]", "dot row1");
}

static void test_csr_extractDiagonal()
{
    MatrixCSR A = build_csr_example();
    std::vector<double> d;
    A.extractDiagonal(d);
    expect_eq(d.size(), (std::size_t)2, __func__, "diag size", "2");
    expect_eq(d[0], 1.0, __func__, "d[0]", "1");
    expect_eq(d[1], 2.0, __func__, "d[1]", "2");
}

static void test_csr_transpose_roundtrip()
{
    MatrixCSR A = build_csr_example();
    MatrixCSR AT = A.transpose();
    MatrixCSR ATT = AT.transpose();
    std::vector<double> x = {1,2,3}, yA(2,0), yR(2,0);
    A.gemv(x, yA, 1.0, 0.0);
    ATT.gemv(x, yR, 1.0, 0.0);
    expect_near(yA[0], yR[0], 1e-12, __func__, "row0", "match");
    expect_near(yA[1], yR[1], 1e-12, __func__, "row1", "match");
}

// static void test_csr_scale_and_axpySamePattern()
// {
//     MatrixCSR A = build_csr_example();
//     MatrixCSR B = build_csr_example();
//     A.scale(0.5);              // A := 0.5 A
//     B.axpySamePattern(0.5, A); // B := B + 0.5 A = 1.5 * original
//     std::vector<double> x = {1,2,3}, yB(2,0), yRef(2,0);
//     B.gemv(x, yB, 1.0, 0.0);
//     MatrixCSR Ref = build_csr_example();
//     Ref.scale(1.5);
//     Ref.gemv(x, yRef, 1.0, 0.0);
//     expect_near(yB[0], yRef[0], 1e-12, __func__, "row0", "match");
//     expect_near(yB[1], yRef[1], 1e-12, __func__, "row1", "match");
// }

static void test_csr_scale_and_axpySamePattern()
{
    MatrixCSR A = build_csr_example();
    MatrixCSR B = build_csr_example();

    A.scale(0.5);              // A = 0.5 * original
    //B.axpySamePattern(0.5, A); // B = 1.25 * original  (WRONG for the test)
    B.axpySamePattern(1.0, A); // B = 1.0*orig + 1.0*(0.5*orig) = 1.5 * original

    std::vector<double> x = {1,2,3}, yB(2,0), yRef(2,0);
    B.gemv(x, yB, 1.0, 0.0);

    MatrixCSR Ref = build_csr_example();
    Ref.scale(1.5);
    Ref.gemv(x, yRef, 1.0, 0.0);

    expect_near(yB[0], yRef[0], 1e-12, __func__, "row0", "match");
    expect_near(yB[1], yRef[1], 1e-12, __func__, "row1", "match");
}

static void test_csr_zero_and_identity()
{
    using namespace dd::algebra;

    // Zero: just scale values to 0.0
    MatrixCSR A = build_csr_example();
    A.scale(0.0);
    std::vector<double> x = {1,2,3}, y(2,0);
    A.gemv(x, y, 1.0, 0.0);
    expect_eq(y[0], 0.0, __func__, "zero row0", "0");
    expect_eq(y[1], 0.0, __func__, "zero row1", "0");

    // Identity: build with COO and convert to CSR
    MatrixCOO Icoo(3,3);
    Icoo.add(0,0,1.0);
    Icoo.add(1,1,1.0);
    Icoo.add(2,2,1.0);
    MatrixCSR I(Icoo);

    std::vector<double> xi = {1,2,3}, yi(3,0);
    I.gemv(xi, yi, 1.0, 0.0);
    expect_eq(yi[0], 1.0, __func__, "I*x[0]", "1");
    expect_eq(yi[1], 2.0, __func__, "I*x[1]", "2");
    expect_eq(yi[2], 3.0, __func__, "I*x[2]", "3");
}


int main()
{
    // COO suite
    test_coo_gemv_vectors();
    test_coo_extractDiagonal();
    test_coo_norms();
    test_coo_axpy();
    test_coo_to_csr_and_back();

    // CSR suite
    test_csr_gemv_vectors();
    test_csr_extractDiagonal();
    test_csr_transpose_roundtrip();
    test_csr_scale_and_axpySamePattern();
    test_csr_zero_and_identity();

    return ddtest::summarize_and_exit();
    
}