#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <stdexcept>
#include <sstream>   // for std::stringstream
#include <cstdio>    // for std::remove

#include "algebra/matrixDense.hpp"

#include "algebra/matrixSparse.hpp"
#include "algebra/COO.hpp"
#include "algebra/CSR.hpp"
#include "../tests/test_util.hpp"

static void testConstructors()
{
    // default
    MatrixDense A;
    expect_true(A.rows() == 0 && A.cols() == 0, __func__, "default ctor yields 0x0");
    // sized with init value
    MatrixDense B(2, 3, 1.5);
    expect_eq(B.rows(), static_cast<MatrixDense::Index>(2), __func__, "B.rows()", "2");
    expect_eq(B.cols(), static_cast<MatrixDense::Index>(3), __func__, "B.cols()", "3");
    expect_eq(B(1, 2), 1.5, __func__, "B(1,2)", "1.5");
    // initializer list
    MatrixDense C{{1.0, 2.0}, {3.0, 4.0}};
    expect_eq(C.rows(), static_cast<MatrixDense::Index>(2), __func__, "C.rows()", "2");
    expect_eq(C.cols(), static_cast<MatrixDense::Index>(2), __func__, "C.cols()", "2");
    expect_eq(C(0, 1), 2.0, __func__, "C(0,1)", "2.0");
    expect_eq(C(1, 0), 3.0, __func__, "C(1,0)", "3.0");
    // ragged initializer throws
    expect_throw<std::runtime_error>([] {
        MatrixDense X{{1.0}, {2.0, 3.0}};
    }, __func__, "ragged initializer should throw");
}

static void testFillIdentityAndResize()
{
    MatrixDense M(2, 2, 0.0);
    // fill
    M.fill(3.0);
    expect_eq(M(0, 1), 3.0, __func__, "M(0,1)", "3.0");
    // zero
    M.setZero();
    expect_eq(M(1, 0), 0.0, __func__, "M(1,0)", "0.0");
    // identity
    M.setIdentity();
    expect_eq(M(0, 0), 1.0, __func__, "M(0,0)", "1.0");
    expect_eq(M(0, 1), 0.0, __func__, "M(0,1)", "0.0");
    expect_eq(M(1, 1), 1.0, __func__, "M(1,1)", "1.0");
    // identity on non‑square throws
    MatrixDense N(2, 3, 0.0);
    expect_throw<std::runtime_error>([&] { N.setIdentity(); }, __func__, "setIdentity on non‑square should throw");
    // resize without keeping (pad)
    MatrixDense R(2, 2, 1.0);
    R.resize(3, 4, false, 2.0);
    expect_eq(R.rows(), static_cast<MatrixDense::Index>(3), __func__, "R.rows()", "3");
    expect_eq(R.cols(), static_cast<MatrixDense::Index>(4), __func__, "R.cols()", "4");
    expect_eq(R(2, 3), 2.0, __func__, "R(2,3)", "2.0");
    // resize with keep
    R.resize(4, 5, true, 9.0);
    // original block [0..2)x[0..4) values should persist (value 2.0), new cells should be 9.0
    expect_eq(R(2, 3), 2.0, __func__, "R(2,3)", "2.0 (kept)");
    expect_eq(R(3, 4), 9.0, __func__, "R(3,4)", "9.0 (pad)");
}

static void testGemv()
{
    // Basic gemv with raw pointers
    MatrixDense M{{1, 2, 3}, {4, 5, 6}};
    double x[] = {1.0, 1.0, 1.0};
    double y[] = {0.0, 0.0};
    M.gemv(x, y);
    expect_eq(y[0], 6.0, __func__, "y[0]", "6.0");
    expect_eq(y[1], 15.0, __func__, "y[1]", "15.0");
    // gemv with alpha and beta
    double y2[] = {1.0, 2.0};
    M.gemv(x, y2, 2.0, 3.0); // y2 = 2*M*x + 3*y2
    expect_eq(y2[0], 15.0, __func__, "y2[0]", "15.0"); // 2*6 + 3*1
    expect_eq(y2[1], 36.0, __func__, "y2[1]", "36.0"); // 2*15 + 3*2
    // gemv with std::vector and scaling
    std::vector<double> vx = {1.0, 0.0, -1.0};
    std::vector<double> vy(2, 5.0);
    M.gemv(vx, vy, 1.0, 0.5);
    // M*[1,0,-1] = [-2, -2]; y = 1*[-2] +0.5*[5] = [-2+2.5]=0.5
    expect_near(vy[0], 0.5, 1e-12, __func__, "vy[0]", "0.5");
    expect_near(vy[1], 0.5, 1e-12, __func__, "vy[1]", "0.5");
    // gemv returning a vector
    auto w = M.gemv(std::vector<double>{1.0, 1.0, 1.0});
    expect_eq(w.size(), static_cast<std::size_t>(2), __func__, "w.size()", "2");
    expect_eq(w[0], 6.0, __func__, "w[0]", "6.0");
    expect_eq(w[1], 15.0, __func__, "w[1]", "15.0");
    // wrong size x should throw
    expect_throw<std::runtime_error>([&] {
        std::vector<double> badx = {1.0, 1.0};
        M.gemv(badx);
    }, __func__, "gemv with bad x size should throw");
}

static void testGemm()
{
    MatrixDense A{{1, 2}, {3, 4}};
    MatrixDense B{{5, 6}, {7, 8}};
    MatrixDense C(2, 2, 0.0);
    MatrixDense::gemm(A, B, C);
    
    // A*B = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
    expect_eq(C(0, 0), 19.0, __func__, "C(0,0)", "19.0");
    expect_eq(C(0, 1), 22.0, __func__, "C(0,1)", "22.0");
    expect_eq(C(1, 0), 43.0, __func__, "C(1,0)", "43.0");
    expect_eq(C(1, 1), 50.0, __func__, "C(1,1)", "50.0");
    
    // gemm with alpha and beta
    MatrixDense D{{1, 0}, {0, 1}};
    MatrixDense E = MatrixDense::Zero(2, 2);
    
    // compute E = 0.5*A*B + 0.1*I
    E(0,0) = 1.0; E(1,1) = 1.0;
    MatrixDense::gemm(A, B, E, 0.5, 0.1);
    
    // expected: 0.5*[[19,22],[43,50]] + 0.1*[[1,0],[0,1]] = [[9.5+0.1,11+0],[21.5,25+0.1]]
    expect_near(E(0,0), 9.6, 1e-12, __func__, "E(0,0)", "9.6");
    expect_near(E(0,1), 11.0, 1e-12, __func__, "E(0,1)", "11.0");
    expect_near(E(1,0), 21.5, 1e-12, __func__, "E(1,0)", "21.5");
    expect_near(E(1,1), 25.1, 1e-12, __func__, "E(1,1)", "25.1");
    
    // mismatched shapes should throw
    MatrixDense F(2, 3);
    MatrixDense G(4, 2);
    MatrixDense H(2, 2);
    expect_throw<std::runtime_error>([&] {
        MatrixDense::gemm(F, G, H);
    }, __func__, "gemm with mismatched dims should throw");
    
    // wrong C size should throw
    MatrixDense I(2, 2);
    MatrixDense J(2, 2);
    MatrixDense K(3, 3);
    expect_throw<std::runtime_error>([&] {
        MatrixDense::gemm(I, J, K);
    }, __func__, "gemm with wrong C shape should throw");
}

static void testScaleAxpy()
{
    MatrixDense M{{1, 2}, {3, 4}};
    M.scale(2.0);
    expect_eq(M(0, 0), 2.0, __func__, "M(0,0)", "2.0");
    expect_eq(M(1, 1), 8.0, __func__, "M(1,1)", "8.0");
    MatrixDense N{{1, 1}, {1, 1}};
    M.axpy(-1.0, N); // M = M - N
    expect_eq(M(0, 0), 1.0, __func__, "M(0,0)", "1.0");
    expect_eq(M(1, 1), 7.0, __func__, "M(1,1)", "7.0");
    // shape mismatch should throw
    MatrixDense X(2, 3);
    expect_throw<std::runtime_error>([&] {
        M.axpy(1.0, X);
    }, __func__, "axpy with mismatched shape should throw");
}

static void testTranspose()
{
    MatrixDense M{{1, 2, 3}, {4, 5, 6}};
    auto T = M.transpose();
    expect_eq(T.rows(), static_cast<MatrixDense::Index>(3), __func__, "T.rows()", "3");
    expect_eq(T.cols(), static_cast<MatrixDense::Index>(2), __func__, "T.cols()", "2");
    expect_eq(T(0, 0), 1.0, __func__, "T(0,0)", "1.0");
    expect_eq(T(2, 1), 6.0, __func__, "T(2,1)", "6.0");
    // in‑place transpose
    MatrixDense S{{1, 2}, {3, 4}};
    S.transposeInPlace();
    expect_eq(S(0, 1), 3.0, __func__, "S(0,1)", "3.0");
    expect_eq(S(1, 0), 2.0, __func__, "S(1,0)", "2.0");
    // transposeInPlace on non‑square throws
    MatrixDense R(2, 3);
    expect_throw<std::runtime_error>([&] {
        R.transposeInPlace();
    }, __func__, "transposeInPlace on non‑square should throw");
}

static void testNorms()
{
    MatrixDense M{{1, -2}, {3, -4}};
    double f = M.frobeniusNorm();
    expect_near(f, std::sqrt(30.0), 1e-12, __func__, "f", "sqrt(30)");
    double m = M.maxAbs();
    expect_eq(m, 4.0, __func__, "maxAbs", "4.0");
}

static void testDiagonalAndBlock()
{
    MatrixDense M{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    std::vector<double> d;
    M.extractDiagonal(d);
    expect_eq(d.size(), static_cast<std::size_t>(3), __func__, "d.size()", "3");
    expect_eq(d[0], 1.0, __func__, "d[0]", "1.0");
    expect_eq(d[2], 9.0, __func__, "d[2]", "9.0");
    // block extraction
    auto B = M.block(1, 1, 2, 2);
    expect_eq(B.rows(), static_cast<MatrixDense::Index>(2), __func__, "B.rows()", "2");
    expect_eq(B.cols(), static_cast<MatrixDense::Index>(2), __func__, "B.cols()", "2");
    expect_eq(B(0, 0), 5.0, __func__, "B(0,0)", "5.0");
    expect_eq(B(1, 1), 9.0, __func__, "B(1,1)", "9.0");
    // block out of range throws
    expect_throw<std::runtime_error>([&] {
        M.block(2, 2, 2, 2);
    }, __func__, "block out of range should throw");
}

static void testToString()
{
    MatrixDense M{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    std::string s = M.toString();
    // Should contain matrix dimensions and some entries
    bool ok = s.find("3x3") != std::string::npos && s.find("1, 2") != std::string::npos;
    expect_true(ok, __func__, "toString contains dimensions and entries");
}

static void testAtAndBounds()
{
    MatrixDense M{{1, 2}, {3, 4}};
    // valid
    expect_eq(M.at(1, 0), 3.0, __func__, "M.at(1,0)", "3.0");
    // out of range
    expect_throw<std::out_of_range>([&] {
        (void)M.at(2, 0);
    }, __func__, "at row out of range should throw");
    expect_throw<std::out_of_range>([&] {
        (void)M.at(0, 2);
    }, __func__, "at col out of range should throw");
}

static MatrixDense make_tridiag(int n, double d=2.0, double off=-1.0)
{
    MatrixDense A(n, n, 0.0);
    for (int i = 0; i < n; ++i) {
        A(i,i) = d;
        if (i+1 < n) { A(i, i+1) = off; A(i+1, i) = off; }
    }
    return A;
}

static MatrixDense make_skew(int n)
{
    MatrixDense A(n, n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = i+1; j < n; ++j) {
            double v = double(i - j);
            A(i,j) =  v;
            A(j,i) = -v;
        }
    }
    return A;
}

static bool allclose(const MatrixDense& A, const MatrixDense& B, double tol=1e-12)
{
    if (A.rows() != B.rows() || A.cols() != B.cols()) return false;
    for (int i = 0; i < A.rows(); ++i)
        for (int j = 0; j < A.cols(); ++j)
            if (std::fabs(A(i,j) - B(i,j)) > tol) return false;
    return true;
}

static void test_mm_array_general_roundtrip()
{
    MatrixDense A = make_tridiag(100);
    const char* fname = "A_100x100_general.mtx";
    expect_true(A.to_mm_array_file(fname, "general"), __func__, "write general mm array");
    MatrixDense B = MatrixDense::from_mm_array_file(fname);
    expect_true(allclose(A,B,1e-15), __func__, "general roundtrip A == B");
    std::remove(fname);
}

static void test_mm_array_symmetric_roundtrip()
{
    MatrixDense A = make_tridiag(32);
    const char* fname = "S_32x32_sym.mtx";
    expect_true(A.to_mm_array_file(fname, "symmetric"), __func__, "write symmetric mm array");
    MatrixDense B = MatrixDense::from_mm_array_file(fname);
    expect_true(allclose(A,B,1e-15), __func__, "symmetric roundtrip A == B");
    // spot checks on symmetry
    expect_eq(B(0,1), B(1,0), __func__, "B(0,1)", "B(1,0)");
    expect_eq(B(5,5), 2.0, __func__, "B(5,5)", "2.0");
    std::remove(fname);
}

static void test_mm_array_skew_roundtrip()
{
    MatrixDense A = make_skew(21);
    const char* fname = "K_21x21_skew.mtx";
    expect_true(A.to_mm_array_file(fname, "skew-symmetric"), __func__, "write skew-symmetric mm array");
    MatrixDense B = MatrixDense::from_mm_array_file(fname);
    expect_true(allclose(A,B,1e-15), __func__, "skew roundtrip A == B");
    // skew: diagonal should be zero and B(j,i) == -B(i,j)
    for (int i = 0; i < B.rows(); ++i) {
        expect_eq(B(i,i), 0.0, __func__, "B(i,i)", "0.0");
    }
    expect_eq(B(2,7), -B(7,2), __func__, "B(2,7)", "-B(7,2)");
    std::remove(fname);
}

static void test_mm_array_integer_field()
{
    // 2x3 integer general; column-major order on disk
    std::stringstream ss;
    ss << "%%MatrixMarket matrix array integer general\n"
       << "2 3\n"
       << "1\n3\n"   // col 1
       << "5\n"      // col 2, row 1
       << "2\n4\n"   // col 2, row 2; col 3, row 1
       << "6\n";     // col 3, row 2
    MatrixDense A = MatrixDense::from_mm_array_stream(ss);
    expect_eq(A.rows(), 2, __func__, "A.rows()", "2");
    expect_eq(A.cols(), 3, __func__, "A.cols()", "3");
    expect_eq(A(0,0), 1.0, __func__, "A(0,0)", "1.0");
    expect_eq(A(1,0), 3.0, __func__, "A(1,0)", "3.0");
    expect_eq(A(0,1), 5.0, __func__, "A(0,1)", "5.0");
    expect_eq(A(1,1), 2.0, __func__, "A(1,1)", "2.0");
    expect_eq(A(0,2), 4.0, __func__, "A(0,2)", "4.0");
    expect_eq(A(1,2), 6.0, __func__, "A(1,2)", "6.0");

}

static void test_mm_array_errors()
{
    // Hermitian not supported for double-only
    expect_throw<std::runtime_error>([](){
        std::stringstream ss;
        ss << "%%MatrixMarket matrix array real hermitian\n" << "3 3\n1\n";
        (void)MatrixDense::from_mm_array_stream(ss);
    }, __func__, "hermitian read should throw");

    // Bad size line
    expect_throw<std::runtime_error>([](){
        std::stringstream ss;
        ss << "%%MatrixMarket matrix array real general\n" << "0 3\n";
        (void)MatrixDense::from_mm_array_stream(ss);
    }, __func__, "invalid size should throw");

    // Writing symmetric when non-square should throw
    expect_throw<std::runtime_error>([](){
        MatrixDense A(2,3,0.0);
        (void)A.to_mm_array_file("x.mtx","symmetric");
    }, __func__, "symmetric write on non-square");
    std::remove("x.mtx");
}

int main()
{
    testConstructors();
    testFillIdentityAndResize();
    testGemv();
    testGemm();
    testScaleAxpy();
    testTranspose();
    testNorms();
    testDiagonalAndBlock();
    testToString();
    testAtAndBounds();
    
    test_mm_array_general_roundtrip();
    test_mm_array_symmetric_roundtrip();
    test_mm_array_skew_roundtrip();
    test_mm_array_integer_field();
    test_mm_array_errors();

    summarize_and_exit();

}