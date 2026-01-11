#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <iomanip>
#include <array>
#include <chrono>
#include <filesystem>
#include <cstdlib>

#include "pde/pde.hpp"
#include "algebra/COO.hpp"
#include "solver/pcg.hpp"
#include "preconditioner/identity.hpp"

namespace fs = std::filesystem;

// -------------------- tiny CLI helpers --------------------

static bool starts_with(const std::string& s, const std::string& p) {
    return s.size() >= p.size() && s.compare(0, p.size(), p) == 0;
}

static std::string get_opt(int argc, char** argv, const std::string& key, const std::string& def = "") {
    // supports: --key value  OR  --key=value
    const std::string k1 = "--" + key;
    const std::string k2 = "--" + key + "=";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == k1) {
            if (i + 1 < argc) return argv[i + 1];
            return def;
        }
        if (starts_with(a, k2)) return a.substr(k2.size());
    }
    return def;
}

static bool has_flag(int argc, char** argv, const std::string& key) {
    const std::string k = "--" + key;
    for (int i = 1; i < argc; ++i) if (argv[i] == k) return true;
    return false;
}

static std::array<Index,3> parse_n(const std::string& s, int dim, int Nx_fallback) {
    // accepts: "Nx" or "Nx,Ny" or "Nx,Ny,Nz"
    std::array<Index,3> n = {static_cast<Index>(Nx_fallback), 1, 1};
    if (s.empty()) {
        // default: isotropic Nx in active dims
        if (dim >= 2) n[1] = n[0];
        if (dim == 3) n[2] = n[0];
        return n;
    }

    std::array<long long,3> tmp = {0,0,0};
    int count = 0;
    std::string cur;
    for (char c : s) {
        if (c == ',') {
            if (!cur.empty() && count < 3) tmp[count++] = std::atoll(cur.c_str());
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    if (!cur.empty() && count < 3) tmp[count++] = std::atoll(cur.c_str());

    auto positive_or = [&](long long v, long long fallback) {
        return (v > 0) ? v : fallback;
    };

    long long Nx = positive_or((count >= 1 ? tmp[0] : 0), Nx_fallback);
    long long Ny = positive_or((count >= 2 ? tmp[1] : 0), Nx);
    long long Nz = positive_or((count >= 3 ? tmp[2] : 0), Nx);

    n[0] = static_cast<Index>(Nx);
    n[1] = (dim >= 2) ? static_cast<Index>(Ny) : static_cast<Index>(1);
    n[2] = (dim == 3) ? static_cast<Index>(Nz) : static_cast<Index>(1);
    return n;
}

static fs::path ensure_parent_dir(const fs::path& p) {
    if (p.has_parent_path()) fs::create_directories(p.parent_path());
    return p;
}

static bool file_exists_nonempty(const fs::path& p) {
    std::error_code ec;
    return fs::exists(p, ec) && fs::is_regular_file(p, ec) && (fs::file_size(p, ec) > 0);
}

// -------------------- benchmark case --------------------

struct Result {
    int dim = 0;
    Index nx = 0, ny = 1, nz = 1;
    std::size_t N = 0;
    std::size_t nnz = 0;
    int iters = -1;
    double t_assemble = 0.0;
    double t_solve_best = 0.0;
    double t_solve_avg = 0.0;
    double t_total = 0.0;
    double tol = 0.0;
    int maxit = 0;
    int repeat = 1;
};

static Result run_one(int dim,
                      const std::array<Index,3>& n,
                      double mu,
                      double c,
                      double tol,
                      int maxit,
                      int repeat)
{
    Result r;
    r.dim = dim;
    r.nx = n[0]; r.ny = n[1]; r.nz = n[2];
    r.tol = tol;
    r.maxit = maxit;
    r.repeat = repeat;

    std::array<double,3> a = {0.0, 0.0, 0.0};
    std::array<double,3> b = {1.0, 1.0, 1.0};

    // constant forcing + homogeneous Dirichlet, same as your plotting snippet
    auto f = [](const Coord3&) { return 1.0; };
    auto g = [](const Coord3&) { return 0.0; };

    auto t0 = std::chrono::steady_clock::now();

    PDE p(dim, n, a, b, mu, c, f, g);

    auto ta0 = std::chrono::steady_clock::now();
    auto [A, rhs] = p.assembleCOO();
    auto ta1 = std::chrono::steady_clock::now();

    r.t_assemble = std::chrono::duration<double>(ta1 - ta0).count();

    r.N = static_cast<std::size_t>(A.cols());
    r.nnz = static_cast<std::size_t>(A.nnz());

    IdentityPreconditioner M;
    PCGSolver solver(A, &M);
    solver.setTolerance(tol);
    solver.setMaxIters(maxit);

    double best = 1e300;
    double sum  = 0.0;
    int iters_last = -1;

    std::vector<double> x(A.cols(), 0.0);

    for (int rep = 0; rep < repeat; ++rep) {
        std::fill(x.begin(), x.end(), 0.0);

        auto ts0 = std::chrono::steady_clock::now();
        iters_last = solver.solve(rhs, x);
        auto ts1 = std::chrono::steady_clock::now();

        double tsolve = std::chrono::duration<double>(ts1 - ts0).count();
        sum += tsolve;
        if (tsolve < best) best = tsolve;
    }

    auto t1 = std::chrono::steady_clock::now();

    r.iters = iters_last;
    r.t_solve_best = best;
    r.t_solve_avg  = sum / std::max(1, repeat);
    r.t_total = std::chrono::duration<double>(t1 - t0).count();
    return r;
}

static void write_csv_header(std::ofstream& os) {
    os << "dim,nx,ny,nz,unknowns,nnz,"
          "iters,tol,maxit,repeat,"
          "time_assemble_s,time_solve_best_s,time_solve_avg_s,total_time_s\n";
}

static void write_csv_row(std::ofstream& os, const Result& r) {
    os.setf(std::ios::scientific);
    os << std::setprecision(17);

    os << r.dim << ","
       << r.nx  << "," << r.ny << "," << r.nz << ","
       << r.N   << "," << r.nnz << ","
       << r.iters << ","
       << r.tol << ","
       << r.maxit << ","
       << r.repeat << ","
       << r.t_assemble << ","
       << r.t_solve_best << ","
       << r.t_solve_avg << ","
       << r.t_total
       << "\n";
}

static void usage(const char* prog) {
    std::cerr <<
      "Usage:\n"
      "  " << prog << " --dim=1|2|3  --Nx=NN   [options]\n"
      "  " << prog << " --dim=1|2|3  --n=Nx,Ny,Nz (Ny/Nz optional)\n\n"
      "Options:\n"
      "  --dim       Dimension (1,2,3)\n"
      "  --Nx        Grid points per dim (interior, same meaning as your snippet)\n"
      "  --n         Explicit n as 'Nx' or 'Nx,Ny' or 'Nx,Ny,Nz'\n"
      "  --tol       PCG tolerance (default 1e-12)\n"
      "  --maxit     PCG max iterations (default 500000)\n"
      "  --repeat    Repeat solve multiple times (best + avg reported) (default 1)\n"
      "  --csv       Output CSV path (default ../data/output/csv/baseline_identity.csv)\n"
      "  --append    Append to CSV (otherwise overwrite)\n"
      "  --mu        Diffusion coefficient mu (default 1.0)\n"
      "  --c         Reaction coefficient c (default 0.0)\n";
}

// -------------------- main --------------------

int main(int argc, char** argv) {
    if (has_flag(argc, argv, "help") || has_flag(argc, argv, "h")) {
        usage(argv[0]);
        return 0;
    }

    const int dim = std::atoi(get_opt(argc, argv, "dim", "0").c_str());
    const int Nx  = std::atoi(get_opt(argc, argv, "Nx",  "0").c_str());
    const std::string n_str = get_opt(argc, argv, "n", "");

    if (dim < 1 || dim > 3) {
        std::cerr << "ERROR: --dim must be 1, 2, or 3.\n";
        usage(argv[0]);
        return 1;
    }
    if (Nx <= 0 && n_str.empty()) {
        std::cerr << "ERROR: provide --Nx or --n.\n";
        usage(argv[0]);
        return 1;
    }

    const double tol = std::atof(get_opt(argc, argv, "tol", "1e-12").c_str());
    const int maxit  = std::atoi(get_opt(argc, argv, "maxit", "500000").c_str());
    const int repeat = std::max(1, std::atoi(get_opt(argc, argv, "repeat", "1").c_str()));

    const double mu = std::atof(get_opt(argc, argv, "mu", "1.0").c_str());
    const double c  = std::atof(get_opt(argc, argv, "c",  "0.0").c_str());

    std::array<Index,3> n = parse_n(n_str, dim, Nx);

    // CSV handling
    fs::path csv_path = get_opt(argc, argv, "csv", "../data/output/csv/baseline_identity.csv");
    ensure_parent_dir(csv_path);

    const bool append = has_flag(argc, argv, "append");
    const bool need_header = !(append && file_exists_nonempty(csv_path));

    std::ofstream os;
    if (append) os.open(csv_path, std::ios::out | std::ios::app);
    else        os.open(csv_path, std::ios::out | std::ios::trunc);

    if (!os) {
        std::cerr << "ERROR: cannot open CSV file: " << csv_path.string() << "\n";
        return 2;
    }
    if (need_header) write_csv_header(os);

    // Run
    std::cout << "Running baseline PCG+Identity: dim=" << dim
              << " n=(" << n[0] << "," << n[1] << "," << n[2] << ")"
              << " tol=" << tol << " maxit=" << maxit
              << " repeat=" << repeat
              << " mu=" << mu << " c=" << c
              << "\n";

    Result r = run_one(dim, n, mu, c, tol, maxit, repeat);

    std::cout << "  unknowns=" << r.N << " nnz=" << r.nnz
              << " iters=" << r.iters
              << " assemble=" << std::setprecision(6) << std::fixed << r.t_assemble << " s"
              << " solve(best)=" << r.t_solve_best << " s"
              << " solve(avg)=" << r.t_solve_avg << " s"
              << " total=" << r.t_total << " s\n";

    write_csv_row(os, r);
    std::cout << "CSV row written to: " << csv_path.string() << "\n";

    return 0;
}
