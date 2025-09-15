# Domain Decompostion Project

## Introduction
This repo is the folder that contain the domain decompostions project valuable for AMSC course, year 2025-26. Project delve into how to split the solving linear system computation into multiple processors system through different algebraic domain decompositon techniques. 

## Command to compile and Run the project 

## Structure of the Projects

├── include
│   └── dd
│       ├── algebra
│       │   ├── COO.hpp
│       │   ├── CSR.hpp
│       │   ├── matrixDense.hpp
│       │   └── matrixSparse.hpp
│       ├── preconditioner
│       │   ├── identity.hpp
│       │   └── preconditioner.hpp
│       └── solver
│           ├── pcg.hpp
│           └── solver.hpp
├── README.md
├── src
│   ├── algebra
│   │   ├── COO.cpp
│   │   ├── CSR.cpp
│   │   ├── matrixDense.cpp
│   │   └── matrixSparse.cpp
│   ├── preconditioner
│   │   ├── identity.cpp
│   │   └── preconditioner.cpp
│   └── solver
│       ├── pcg.cpp
│       └── solver.cpp
└── tests
    ├── input
    │   ├── A_100x100.mtx
    │   └── COO_laplacian_5x5.mtx
    ├── testMatrixDense.cpp
    ├── testMatrixSparse.cpp
    ├── testSolver.cpp
    └── test_util.hpp
# AMSC-project
Project valuable for AMSC exam
