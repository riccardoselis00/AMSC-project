cmake_minimum_required(VERSION 3.20)
project(DomainDecomposition LANGUAGES CXX)

# ================================================================
# Options
# ================================================================
# Whether to build the unit tests. When off only the library is built.
option(DD_BUILD_TESTS       "Build unit tests"          ON)
# Enable address/undefined sanitizers (Debug mode only, non‑MSVC).
option(DD_ENABLE_SANITIZERS "Enable ASan/UBSan (Debug)" OFF)
# Build shared (ON) or static (OFF) libraries.
option(BUILD_SHARED_LIBS    "Build shared libs"         OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ================================================================
# Library: Algebra (Dense)
# ================================================================

add_library(dd_algebra
    src/algebra/matrixDense.cpp
    src/algebra/matrixSparse.cpp
    src/algebra/COO.cpp
    src/algebra/CSR.cpp
    src/solver/solver.cpp
    src/solver/pcg.cpp
    src/preconditioner/preconditioner.cpp
    src/preconditioner/identity.cpp
)
add_library(dd::algebra ALIAS dd_algebra)

target_include_directories(dd_algebra
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include        # header location for clients
        ${PROJECT_SOURCE_DIR}/src/algebra    # keep internal header reachable for now
)

if(MSVC)
  target_compile_options(dd_algebra PRIVATE /W4 /permissive-)
else()
  target_compile_options(dd_algebra PRIVATE -Wall -Wextra -Wpedantic)
endif()

set_target_properties(dd_algebra PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(DD_ENABLE_SANITIZERS AND CMAKE_BUILD_TYPE STREQUAL "Debug" AND NOT MSVC)
  target_compile_options(dd_algebra PRIVATE -fsanitize=address,undefined)
  target_link_libraries(dd_algebra PRIVATE -fsanitize=address,undefined)
endif()

# ================================================================
# Tests
# ================================================================
if(DD_BUILD_TESTS)
  enable_testing()
  add_executable(testMatrixDense tests/testMatrixDense.cpp)
  add_executable(testMatrixSparse tests/testMatrixSparse.cpp)
  add_executable(testSolver tests/testSolver.cpp)
  target_link_libraries(testMatrixDense PRIVATE dd_algebra)
  target_link_libraries(testMatrixSparse PRIVATE dd_algebra)
  target_link_libraries(testSolver PRIVATE dd_algebra)
  add_test(NAME MatrixDense COMMAND testMatrixDense)
  add_test(NAME MatrixSparse COMMAND testMatrixSparse)
  add_test(NAME Solver COMMAND testSolver)

endif()

# ================================================================
# Placeholders for future components (commented out)
# ================================================================
# add_library(dd_sparse ...)
# add_library(dd_solvers ...)
# add_library(dd_preconditioners ...)
# add_library(dd_eigen ...)
# add_executable(dd_usecase_poisson ...)
# ...

message(STATUS "\n--- Configuration Summary ---")
message(STATUS "CMAKE_BUILD_TYPE      = ${CMAKE_BUILD_TYPE}")
message(STATUS "DD_BUILD_TESTS        = ${DD_BUILD_TESTS}")
message(STATUS "DD_ENABLE_SANITIZERS  = ${DD_ENABLE_SANITIZERS}")
message(STATUS "BUILD_SHARED_LIBS     = ${BUILD_SHARED_LIBS}")