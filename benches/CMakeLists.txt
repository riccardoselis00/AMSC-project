cmake_minimum_required(VERSION 3.20)

# This file defines "applications" (benchmarks / experiments)
# that link against dd_algebra (and MPI if enabled).

# Helper function to reduce boilerplate
function(dd_add_bench name)
  add_executable(${name} ${ARGN})

  # Link with the core library
  target_link_libraries(${name} PRIVATE dd_algebra)

  # Propagate warnings etc. (optional but useful)
  if(MSVC)
    target_compile_options(${name} PRIVATE /W4 /permissive-)
  else()
    target_compile_options(${name} PRIVATE -Wall -Wextra -Wpedantic)
  endif()

  # MPI if available
  if(DD_ENABLE_MPI)
    target_link_libraries(${name} PRIVATE MPI::MPI_CXX)
  endif()

  # Include the same public headers
  target_include_directories(${name}
    PRIVATE
      ${PROJECT_SOURCE_DIR}/include
      ${CMAKE_SOURCE_DIR}/utils
  )
endfunction()

# ================================================================
# List your benches here
# ================================================================
# Example 1: pure sequential PCG + preconditioners
dd_add_bench(bench_pcg_sequential
  bench_seq_preconditioner.cpp
)

# Example 2: MPI additive Schwarz + coarse operator
dd_add_bench(bench_seq_as_blocks
  bench_seq_as_blocks.cpp
)

# Example 2: MPI additive Schwarz + coarse operator
dd_add_bench(bench_mpi_as_coarse
  bench_mpi_as_coarse.cpp
)

dd_add_bench(physical_problem
  physical_problem.cpp
)

dd_add_bench(bench_baseline_identity_solver
  bench_baseline_identity_solver.cpp
)

dd_add_bench(bench_final_identity_vs_AS2
  bench_final_identity_vs_AS2.cpp
)