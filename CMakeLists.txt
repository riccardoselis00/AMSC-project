cmake_minimum_required(VERSION 3.20)
project(DomainDecomposition LANGUAGES CXX)

# ================================================================
# Options
# ================================================================
option(DD_BUILD_TESTS       "Build unit tests"          ON)
option(DD_ENABLE_SANITIZERS "Enable ASan/UBSan (Debug)" OFF)
option(BUILD_SHARED_LIBS    "Build shared libs"         OFF)
option(DD_BUILD_BENCHES     "Build bench / application executables" ON)
option(DD_ENABLE_MPI        "Enable MPI support"        ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ================================================================
# FIND MPI FIRST (before benches/tests)
# ================================================================
if(DD_ENABLE_MPI)
  find_package(MPI REQUIRED)
endif()

# ================================================================
# Core library
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
    src/preconditioner/block_jacobi.cpp
    src/preconditioner/diagonal_jacobi.cpp
    src/preconditioner/additive_schwarz.cpp
    src/solver/pcg_mpi.cpp
    src/partitioner/partitioner.cpp
    src/pde/pde.cpp
)

add_library(dd::algebra ALIAS dd_algebra)

target_include_directories(dd_algebra
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/utils
)

if(DD_ENABLE_MPI)
  target_link_libraries(dd_algebra PUBLIC MPI::MPI_CXX)
endif()

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
# BENCHES — must come AFTER find_package(MPI) and after the library
# ================================================================
if(DD_BUILD_BENCHES)
  add_subdirectory(benches)
endif()

# ================================================================
# TESTS — must also come after library and MPI
# ================================================================
if(DD_BUILD_TESTS)
  enable_testing()

  add_executable(testMatrixDense tests/testMatrixDense.cpp)
  add_executable(testMatrixSparse tests/testMatrixSparse.cpp)
  add_executable(testSolver tests/testSolver.cpp)
  add_executable(testPreconditioner tests/testPreconditioner.cpp)
  add_executable(testPartitioner tests/testPartitioner.cpp)
  add_executable(testMPI tests/testMPI.cpp)
  add_executable(testPDE tests/testPDE.cpp)

  target_link_libraries(testMatrixDense PRIVATE dd_algebra)
  target_link_libraries(testMatrixSparse PRIVATE dd_algebra)
  target_link_libraries(testSolver PRIVATE dd_algebra)
  target_link_libraries(testPreconditioner PRIVATE dd_algebra)
  target_link_libraries(testPartitioner PRIVATE dd_algebra)
  target_link_libraries(testMPI PRIVATE dd_algebra)
  target_link_libraries(testPDE PRIVATE dd_algebra)

  if(DD_ENABLE_MPI)
    target_link_libraries(testPartitioner PRIVATE MPI::MPI_CXX)
    target_link_libraries(testMPI PRIVATE MPI::MPI_CXX)
  endif()

  add_test(NAME MatrixDense COMMAND testMatrixDense)
  add_test(NAME MatrixSparse COMMAND testMatrixSparse)
  add_test(NAME Solver COMMAND testSolver)
  add_test(NAME Preconditioner COMMAND testPreconditioner)
  add_test(NAME Partitioner COMMAND testPartitioner)
  add_test(NAME MPI COMMAND testMPI)
  add_test(NAME PDE COMMAND testPDE)
endif()

# ================================================================
# Summary
# ================================================================
message(STATUS "\n--- Configuration Summary ---")
message(STATUS "CMAKE_BUILD_TYPE     = ${CMAKE_BUILD_TYPE}")
message(STATUS "DD_BUILD_TESTS       = ${DD_BUILD_TESTS}")
message(STATUS "DD_BUILD_BENCHES     = ${DD_BUILD_BENCHES}")
message(STATUS "DD_ENABLE_MPI        = ${DD_ENABLE_MPI}")
message(STATUS "DD_ENABLE_SANITIZERS = ${DD_ENABLE_SANITIZERS}")
message(STATUS "BUILD_SHARED_LIBS    = ${BUILD_SHARED_LIBS}")
