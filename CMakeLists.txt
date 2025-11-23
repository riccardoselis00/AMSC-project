cmake_minimum_required(VERSION 3.20)
project(DomainDecomposition LANGUAGES CXX)

# ================================================================
# Options
# ================================================================
option(DD_BUILD_TESTS       "Build unit tests"          ON)
option(DD_ENABLE_SANITIZERS "Enable ASan/UBSan (Debug)" OFF)
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

# Ensure all your source files are listed here
add_library(dd_algebra
    src/algebra/matrixDense.cpp
    src/algebra/matrixSparse.cpp
    src/algebra/COO.cpp
    src/algebra/CSR.cpp
    src/solver/solver.cpp
    src/solver/pcg.cpp
    src/preconditioner/preconditioner.cpp
    src/preconditioner/identity.cpp
    src/preconditioner/block_jacobi.cpp
    src/preconditioner/diagonal_jacobi.cpp
    src/preconditioner/additive_schwarz.cpp
)

# It is best practice to keep a namespace for CMake targets (e.g., dd::) 
# even if you removed the C++ namespace. This prevents name collisions.
add_library(dd::algebra ALIAS dd_algebra)

target_include_directories(dd_algebra
    PUBLIC
        # This is the CRITICAL directory. 
        # Since your folders are include/algebra, include/solver, etc.
        # this allows you to do: #include "algebra/matrix.hpp"
        ${PROJECT_SOURCE_DIR}/include
        
        # If you have utility headers in a separate utils folder:
        ${CMAKE_SOURCE_DIR}/utils
)

# Standard warnings
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
  
  # Update these if your test file names or locations changed
  add_executable(testMatrixDense tests/testMatrixDense.cpp)
  add_executable(testMatrixSparse tests/testMatrixSparse.cpp)
  add_executable(testSolver tests/testSolver.cpp)
  add_executable(testPreconditioner tests/testPreconditioner.cpp)

  # Link against the main library
  target_link_libraries(testMatrixDense PRIVATE dd_algebra)
  target_link_libraries(testMatrixSparse PRIVATE dd_algebra)
  target_link_libraries(testSolver PRIVATE dd_algebra)
  target_link_libraries(testPreconditioner PRIVATE dd_algebra)

  add_test(NAME MatrixDense COMMAND testMatrixDense)
  add_test(NAME MatrixSparse COMMAND testMatrixSparse)
  add_test(NAME Solver COMMAND testSolver)
  add_test(NAME Preconditioner COMMAND testPreconditioner)
endif()

# ================================================================
# Summary
# ================================================================
message(STATUS "\n--- Configuration Summary ---")
message(STATUS "CMAKE_BUILD_TYPE      = ${CMAKE_BUILD_TYPE}")
message(STATUS "DD_BUILD_TESTS        = ${DD_BUILD_TESTS}")
message(STATUS "DD_ENABLE_SANITIZERS  = ${DD_ENABLE_SANITIZERS}")
message(STATUS "BUILD_SHARED_LIBS     = ${BUILD_SHARED_LIBS}")