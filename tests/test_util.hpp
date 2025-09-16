#pragma once
#include <iostream>
#include <string>
#include <stdexcept>
#include <functional>
#include <type_traits>
#include <cmath>

namespace ddtest {

inline int& g_tests() { static int t = 0; return t; }
inline int& g_fail()  { static int f = 0; return f; }

inline void expect_true(bool cond,
                        const char* func,
                        const std::string& msg)
{
    ++g_tests();
    if (!cond) {
        ++g_fail();
        std::cerr << func << ": expect_true failed: " << msg << "\n";
    }
}

template<class T, class U>
inline void expect_eq(const T& a, const U& b,
                      const char* func,
                      const char* what,
                      const char* expected_str)
{
    ++g_tests();
    if (!(a == b)) {
        ++g_fail();
        std::cerr << func << ": expect_eq failed: " << what
                  << " got=" << a << " expected=" << expected_str << "\n";
    }
}

inline void expect_near(double a, double b, double tol,
                        const char* func,
                        const char* what,
                        const char* expected_str)
{
    ++g_tests();
    if (std::fabs(a - b) > tol) {
        ++g_fail();
        std::cerr << func << ": expect_near failed: " << what
                  << " got=" << a << " expected≈" << expected_str
                  << " (tol=" << tol << ")\n";
    }
}

// template<class Fn>
// inline void expect_throw(Fn&& fn,
//                          const char* func,
//                          const std::string& msg)
// {
//     ++g_tests();
//     try {
//         fn();
//         ++g_fail();
//         std::cerr << func << ": expect_throw failed: " << msg << "\n";
//     } catch (const std::exception&) {
//         // OK
//     }
// }

// expect_throw<T>(fn, ...) — asserts fn throws exactly T (or derived from T)
template<class E = std::exception, class Fn>
inline void expect_throw(Fn&& fn,
                         const char* func,
                         const std::string& msg)
{
    ++g_tests();
    try {
        fn();
        ++g_fail();
        std::cerr << func << ": expect_throw failed: " << msg
                  << " (no exception)\n";
    } catch (const E&) {
        // OK: expected exception type
    } catch (const std::exception& ex) {
        ++g_fail();
        std::cerr << func << ": expect_throw failed: " << msg
                  << " (caught different exception: " << ex.what() << ")\n";
    } catch (...) {
        ++g_fail();
        std::cerr << func << ": expect_throw failed: " << msg
                  << " (caught non-std exception)\n";
    }
}



inline int summarize_and_exit()
{
    std::cout << g_tests() << " tests, " << g_fail() << " failures\n";
    return g_fail() == 0 ? 0 : 1;
}

} // namespace ddtest