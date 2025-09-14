#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <stdexcept>

#include "dd/algebra/matrixSparse.hpp"
#include "dd/algebra/COO.hpp"
#include "dd/algebra/CSR.hpp"

using dd::algebra::MatrixSparse;
using dd::algebra::MatrixCOO;
using dd::algebra::MatrixCSR;

namespace testutil {
    static int total_tests = 0;
    static int total_fail  = 0;
    void expect_true(bool cond, const std::string& msg)
    {
        ++total_tests;
        if (!cond) {
            ++total_fail;
            std::cerr << "EXPECT_TRUE failed: " << msg << "\n";
        }
    }
    template<class T, class U>
    void expect_eq(const T& a, const U& b, const std::string& msg)
    {
        ++total_tests;
        if (!(a == b)) {
            ++total_fail;
            std::cerr << "EXPECT_EQ failed: " << msg << " got=" << a << " expected=" << b << "\n";
        }
    }
    void expect_near(double a, double b, double eps, const std::string& msg)
    {
        ++total_tests;
        if (std::fabs(a - b) > eps) {
            ++total_fail;
            std::cerr << "EXPECT_NEAR failed: " << msg << " got=" << a << " expected=" << b << " diff=" << std::fabs(a - b) << "\n";
        }
    }
    template<class Ex, class F>
    void expect_throw(F&& f, const std::string& msg)
    {
        ++total_tests;
        bool thrown = false;
        try {
            f();
        } catch (const Ex&) {
            thrown = true;
        } catch (...) {
        }
        if (!thrown) {
            ++total_fail;
            std::cerr << "EXPECT_THROW failed: " << msg << "\n";
        }
    }
}

// Test basic COO operations: gemv, gemv(vector), diagonal, norms, scale, axpy, transpose, zero/identity.
static void test_coo_basic()
{
    using namespace testutil;
    // Build a 2x3 dense pattern in COO
    MatrixCOO A(2, 3);
    A.add(0,0,1); A.add(0,1,2); A.add(0,2,3);
    A.add(1,0,4); A.add(1,1,5); A.add(1,2,6);
    expect_eq(A.rows(), (std::size_t)2, "COO rows");
    expect_eq(A.cols(), (std::size_t)3, "COO cols");
    expect_eq(A.nnz(), 6UL, "COO nnz");

    // MatrixCOO::MatrixCOO(const MatrixCSR& A)
    // ADD TEST
    MatrixCSR Acsr = MatrixCSR::Identity(3);
    MatrixCOO A2(Acsr);
    expect_eq(A2.rows(), (std::size_t)3, "COO from CSR rows");
    expect_eq(A2.cols(), (std::size_t)3, "COO from CSR cols");
    expect_eq(A2.nnz(), 3UL, "COO from CSR nnz");

    // gemv pointer
    double x_arr[] = {1.0, 1.0, 1.0};
    double y_arr[] = {0.0, 0.0};
    A.gemv(x_arr, y_arr); // y = A*x
    expect_near(y_arr[0], 6.0, 1e-12, "COO gemv y[0]");
    expect_near(y_arr[1], 15.0, 1e-12, "COO gemv y[1]");
    
    // gemv vector wrapper (beta and alpha)
    std::vector<double> x_vec{1.0, 0.0, -1.0};
    std::vector<double> y_vec(2, 1.0);
    A.gemv(x_vec, y_vec, 2.0, 0.5);
    // A*[1,0,-1] = [-2,-2]; y = 2*[-2,-2] + 0.5*[1,1] = [-4+0.5, -4+0.5] = [-3.5, -3.5]
    expect_near(y_vec[0], -3.5, 1e-12, "COO gemv vector[0]");
    expect_near(y_vec[1], -3.5, 1e-12, "COO gemv vector[1]");
    
    // gemv returning vector
    auto y_vec2 = A.gemv(std::vector<double>{1.0,1.0,1.0});
    expect_eq(y_vec2.size(), (std::size_t)2, "COO gemv return size");
    expect_near(y_vec2[0], 6.0, 1e-12, "COO gemv return[0]");
    expect_near(y_vec2[1], 15.0, 1e-12, "COO gemv return[1]");
    
    // extract diagonal
    std::vector<double> diag;
    A.extractDiagonal(diag);
    expect_eq(diag.size(), (std::size_t)2, "COO diag size");
    expect_near(diag[0], 1.0, 1e-12, "COO diag[0]");
    expect_near(diag[1], 5.0, 1e-12, "COO diag[1]");
    
    // norms
    expect_near(A.frobeniusNorm(), std::sqrt(91.0), 1e-15, "COO frobenius");
    expect_near(A.maxAbs(), 6.0, 1e-15, "COO maxAbs");
    
    // scale
    MatrixCOO B = A;
    B.scale(0.5);
    auto y_scale = B.gemv(std::vector<double>{1.0,1.0,1.0});
    expect_near(y_scale[0], 3.0, 1e-12, "COO scale y[0]");
    expect_near(y_scale[1], 7.5, 1e-12, "COO scale y[1]");
    // axpy same pattern
    MatrixCOO C = A;
    C.axpySamePattern(-1.0, A);
    auto y_cancel = C.gemv(std::vector<double>{1.0,1.0,1.0});
    expect_near(y_cancel[0] + y_cancel[1], 0.0, 1e-12, "COO axpySamePattern cancel");
    // axpy general (duplicate pattern)
    MatrixCOO D = A;
    D.axpy(-1.0, A);
    auto y_cancel2 = D.gemv(std::vector<double>{1.0,1.0,1.0});
    expect_near(y_cancel2[0] + y_cancel2[1], 0.0, 1e-12, "COO axpy cancel");
    // transpose (check shape and one value)
    MatrixCOO AT = A.transpose();
    expect_eq(AT.rows(), (std::size_t)3, "COO transpose rows");
    expect_eq(AT.cols(), (std::size_t)2, "COO transpose cols");
    // find entry (1,0) from AT should correspond to (0,1) in A (value 2)
    bool found = false;
    AT.forEachNZ([&](MatrixSparse::Index i, MatrixSparse::Index j, MatrixSparse::Scalar v){
        if (i == 1 && j == 0) { expect_near(v, 2.0, 1e-12, "COO transpose val"); found = true; }
    });
    expect_true(found, "COO transpose entry found");
    // static Zero and Identity
    MatrixCOO Z = MatrixCOO::Zero(2, 3);
    expect_eq(Z.nnz(), 0UL, "COO Zero nnz");
    MatrixCOO I = MatrixCOO::Identity(3);
    expect_eq(I.rows(), (std::size_t)3, "COO Identity rows");
    expect_near(I.frobeniusNorm(), std::sqrt(3.0), 1e-15, "COO Identity frobenius");
}

// Test basic CSR operations: conversion from COO, gemv, diagonal, norms, scale, axpy, transpose, Zero/Identity.
static void test_csr_basic()
{
    using namespace testutil;
    // Build same dense pattern as COO
    MatrixCOO Acoo(2, 3);
    Acoo.add(0,0,1); Acoo.add(0,1,2); Acoo.add(0,2,3);
    Acoo.add(1,0,4); Acoo.add(1,1,5); Acoo.add(1,2,6);
    // Convert to CSR
    MatrixCSR A(Acoo);
    expect_eq(A.rows(), (std::size_t)2, "CSR rows");
    expect_eq(A.cols(), (std::size_t)3, "CSR cols");
    expect_eq(A.nnz(), 6UL, "CSR nnz");

    // MatrixCSR::MatrixCSR(const MatrixCOO& A)
    
    // gemv pointer
    double x_arr[] = {1.0, 1.0, 1.0};
    double y_arr[] = {0.0, 0.0};
    A.gemv(x_arr, y_arr);
    expect_near(y_arr[0], 6.0, 1e-12, "CSR gemv y[0]");
    expect_near(y_arr[1], 15.0, 1e-12, "CSR gemv y[1]");
    // gemv vector wrapper
    std::vector<double> y_vec(2, 1.0);
    std::vector<double> x_vec{1.0, 0.0, -1.0};
    A.gemv(x_vec, y_vec, 2.0, 0.5);
    // A*[1,0,-1] = [-2, -2]; y = 2*[-2] + 0.5*[1] = [-3.5,-3.5]
    expect_near(y_vec[0], -3.5, 1e-12, "CSR gemv vector[0]");
    expect_near(y_vec[1], -3.5, 1e-12, "CSR gemv vector[1]");
    // extract diagonal
    std::vector<double> diag;
    A.extractDiagonal(diag);
    expect_eq(diag.size(), (std::size_t)2, "CSR diag size");
    expect_near(diag[0], 1.0, 1e-12, "CSR diag[0]");
    expect_near(diag[1], 5.0, 1e-12, "CSR diag[1]");
    // norms
    expect_near(A.frobeniusNorm(), std::sqrt(91.0), 1e-15, "CSR frobenius");
    expect_near(A.maxAbs(), 6.0, 1e-15, "CSR maxAbs");
    // scale and axpy same pattern
    MatrixCSR B = A;
    B.scale(0.5);
    MatrixCSR C = A;
    C.axpySamePattern(-1.0, A);
    std::vector<double> z(2);
    C.gemv(std::vector<double>{1.0,1.0,1.0}, z);
    expect_near(z[0] + z[1], 0.0, 1e-12, "CSR axpySamePattern cancel");
    // transpose (check shape and a value)
    MatrixCSR AT = A.transpose();
    expect_eq(AT.rows(), (std::size_t)3, "CSR transpose rows");
    expect_eq(AT.cols(), (std::size_t)2, "CSR transpose cols");
    bool found = false;
    AT.forEachNZ([&](MatrixSparse::Index i, MatrixSparse::Index j, MatrixSparse::Scalar v){
        if (i == 1 && j == 0) { expect_near(v, 2.0, 1e-12, "CSR transpose val"); found = true; }
    });
    expect_true(found, "CSR transpose entry found");
    // static Zero and Identity
    MatrixCSR Z = MatrixCSR::Zero(2,3);
    expect_eq(Z.nnz(), 0UL, "CSR Zero nnz");
    MatrixCSR I = MatrixCSR::Identity(3);
    expect_eq(I.nnz(), 3UL, "CSR Identity nnz");
    expect_near(I.frobeniusNorm(), std::sqrt(3.0), 1e-15, "CSR Identity frob");
}

int main()
{
    test_coo_basic();
    test_csr_basic();
    std::cout << "Total tests: " << testutil::total_tests << ", failures: " << testutil::total_fail << "\n";
    return testutil::total_fail == 0 ? 0 : 1;
}