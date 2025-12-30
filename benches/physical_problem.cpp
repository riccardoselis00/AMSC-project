#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <array>
#include <filesystem>

#include "pde/pde.hpp"
#include "algebra/COO.hpp"
#include "algebra/CSR.hpp"
#include "solver/pcg.hpp"
#include "preconditioner/identity.hpp"
#include "utils/timing.hpp"
#include "utils/test_util.hpp"

namespace fs = std::filesystem;

// -------------------- helpers to dump txt files --------------------

static void write_vector_txt(const std::string& path, const std::vector<double>& v) {
    std::ofstream os(path);
    os.setf(std::ios::scientific);
    os << std::setprecision(17);
    for (double x : v) os << x << "\n";
}

// MatrixMarket COO dump (scipy.io.mmread)
static void write_matrix_market_mtx(const std::string& path, const MatrixCOO& A) {
    std::ofstream os(path);
    os.setf(std::ios::scientific);
    os << std::setprecision(17);

    os << "%%MatrixMarket matrix coordinate real general\n";
    os << A.rows() << " " << A.cols() << " " << A.nnz() << "\n";

    const auto& I = A.rowIndex();
    const auto& J = A.colIndex();
    const auto& V = A.values();

    for (size_t k = 0; k < (size_t)A.nnz(); ++k) {
        // MatrixMarket uses 1-based indices
        os << (static_cast<size_t>(I[k]) + 1) << " "
           << (static_cast<size_t>(J[k]) + 1) << " "
           << static_cast<double>(V[k]) << "\n";
    }
}

// grid dump: d=1 -> x u ; d=2 -> x y u ; d=3 -> x y z u  (interior only)
static void write_solution_grid_txt(const std::string& path,
                                   int dim,
                                   int Nx,
                                   const std::array<double,3>& a,
                                   const std::array<double,3>& h,
                                   const std::vector<double>& x)
{
    std::ofstream os(path);
    os.setf(std::ios::scientific);
    os << std::setprecision(17);

    auto idx2 = [Nx](int i, int j) { return (j-1)*Nx + (i-1); };
    auto idx3 = [Nx](int i, int j, int k) { return ((k-1)*Nx + (j-1))*Nx + (i-1); };

    if (dim == 1) {
        for (int i = 1; i <= Nx; ++i) {
            const double X = a[0] + i*h[0];
            const double U = x[static_cast<size_t>(i-1)];
            os << X << " " << U << "\n";
        }
    } else if (dim == 2) {
        for (int j = 1; j <= Nx; ++j) {
            for (int i = 1; i <= Nx; ++i) {
                const double X = a[0] + i*h[0];
                const double Y = a[1] + j*h[1];
                const double U = x[static_cast<size_t>(idx2(i,j))];
                os << X << " " << Y << " " << U << "\n";
            }
        }
    } else { // dim == 3
        for (int k = 1; k <= Nx; ++k) {
            for (int j = 1; j <= Nx; ++j) {
                for (int i = 1; i <= Nx; ++i) {
                    const double X = a[0] + i*h[0];
                    const double Y = a[1] + j*h[1];
                    const double Z = a[2] + k*h[2];
                    const double U = x[static_cast<size_t>(idx3(i,j,k))];
                    os << X << " " << Y << " " << Z << " " << U << "\n";
                }
            }
        }
    }
}

// Create output folder: ../data/output/plot-solutions (when running from build/)
static fs::path ensure_plot_dir() {
    fs::path out = fs::path("..") / "data" / "output" / "plot-solutions";
    fs::create_directories(out);
    return out;
}

static void run_case(int dim, int Nx) {
    std::cout << "\n==== CASE dim=" << dim << " Nx=" << Nx << " ====\n";

    int its = 0;

    std::array<double,3> a = {0.0, 0.0, 0.0};
    std::array<double,3> b = {1.0, 1.0, 1.0};

    const double mu = 1.0;
    const double c  = 0.0;

    auto f = [](const Coord3&) { return 1.0; };
    auto g = [](const Coord3&) { return 0.0; };

    std::array<Index,3> n = {static_cast<Index>(Nx), 1, 1};
    if (dim >= 2) n = {static_cast<Index>(Nx), static_cast<Index>(Nx), 1};
    if (dim == 3) n = {static_cast<Index>(Nx), static_cast<Index>(Nx), static_cast<Index>(Nx)};

    PDE p(dim, n, a, b, mu, c, f, g);

    std::cout << "Assembling A and b...\n";
    auto [A, rhs] = p.assembleCOO();

    printf("Matrix created: %zu x %zu, nnz=%zu\n", A.rows(), A.cols(), A.nnz());

    std::vector<double> x(A.cols(), 0.0);

    IdentityPreconditioner M;
    PCGSolver solver(A, &M);
    solver.setMaxIters(500000);
    solver.setTolerance(1e-12);

    its = solver.solve(rhs, x);

    std::cout << "Solver finished in " << its << " iterations.\n";

    const fs::path outdir = ensure_plot_dir();
    const std::string sd = "d" + std::to_string(dim);
    const std::string base = (outdir / (sd + "_Nx" + std::to_string(Nx))).string();

    std::cout << "Writing files to: " << outdir.string() << "\n";

    write_vector_txt(base + "_b.txt", rhs);
    write_vector_txt(base + "_x.txt", x);
    write_solution_grid_txt(base + "_grid.txt", dim, Nx, p.a(), p.h(), x);

    // WARNING: for dim=3 and large Nx this file can be huge.
    write_matrix_market_mtx(base + "_A.mtx", A);

    std::cout << "Done for " << sd << ".\n";
}

int main() {
    const int Nx = 20;
    run_case(1, Nx);
    run_case(2, Nx);
    run_case(3, Nx);
    return 0;
}
