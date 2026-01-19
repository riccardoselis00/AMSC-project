// bench_final_identity_vs_AS2.cpp
#include <mpi.h>

#include <iostream>
#include <vector>
#include <string>
#include <array>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <algorithm>
#include <cmath>

#include "pde/pde.hpp"
#include "algebra/COO.hpp"
#include "solver/pcg_mpi.hpp"
#include "preconditioner/identity.hpp"
#include "preconditioner/additive_schwarz.hpp"
#include "partitioner/partitioner.hpp"

namespace fs = std::filesystem;

// -------------------- tiny CLI helpers (same style as baseline) --------------------

static bool starts_with(const std::string& s, const std::string& p) {
    return s.size() >= p.size() && s.compare(0, p.size(), p) == 0;
}

static std::string get_opt(int argc, char** argv, const std::string& key, const std::string& def = "") {
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
    std::array<Index,3> n = {static_cast<Index>(Nx_fallback), 1, 1};
    if (s.empty()) {
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

// -------------------- CSV helpers --------------------

static void write_csv_header(std::ofstream& os) {
    os << "dim,nx,ny,nz,unknowns,nnz,"
          "prec,nprocs,nparts,overlap,"
          "iters,tol,maxit,repeat,"
          "time_assemble_s,time_setup_s,time_solve_best_s,time_solve_avg_s,"
          "total_time_best_s,total_time_avg_s\n";
}

static void usage(const char* prog) {
    std::cerr <<
      "Usage:\n"
      "  mpirun -np P " << prog << " --dim=1|2|3  --Nx=NN  [options]\n"
      "  mpirun -np P " << prog << " --dim=1|2|3  --n=Nx,Ny,Nz (Ny/Nz optional)\n\n"
      "Options:\n"
      "  --prec       identity | as | as2     (default as2)\n"
      "  --nparts     subdomains per rank     (default 32)\n"
      "  --overlap    overlap size            (default 1)\n"
      "  --tol        PCG tolerance           (default 1e-12)\n"
      "  --maxit      PCG max iterations      (default 500000)\n"
      "  --repeat     repeat solve (best+avg) (default 1)\n"
      "  --mu         diffusion coefficient (const mode) (default 1.0)\n"
      "  --c          reaction coefficient (const)       (default 0.0)\n"
      "  --csv        output CSV path         (default ../data/output/csv/final_mpi_physical.csv)\n"
      "  --append     append to CSV (else overwrite)\n"
      "\n"
      "Difficulty knobs (variable diffusion / forcing):\n"
      "  --mu-type    const|layer|checker|inclusion (default const)\n"
      "  --mu-min     minimum diffusion            (default 1e-4)\n"
      "  --mu-max     maximum diffusion            (default 1.0)\n"
      "  --mu-pos     layer interface position x   (default 0.5)\n"
      "  --mu-cells   checker cells per axis       (default 8)\n"
      "  --mu-radius  inclusion radius             (default 0.25)\n"
      "  --f-type     const|sine|checker           (default const)\n"
      "  --f-amp      amplitude for f (non-const)  (default 1.0)\n";
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    int rank = 0, nprocs = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    if (has_flag(argc, argv, "help") || has_flag(argc, argv, "h")) {
        if (rank == 0) usage(argv[0]);
        MPI_Finalize();
        return 0;
    }

    // ---- parse physical problem ----
    const int dim = std::atoi(get_opt(argc, argv, "dim", "0").c_str());
    const int Nx  = std::atoi(get_opt(argc, argv, "Nx",  "0").c_str());
    const std::string n_str = get_opt(argc, argv, "n", "");

    if (dim < 1 || dim > 3) {
        if (rank == 0) { std::cerr << "ERROR: --dim must be 1,2,3\n"; usage(argv[0]); }
        MPI_Finalize();
        return 1;
    }
    if (Nx <= 0 && n_str.empty()) {
        if (rank == 0) { std::cerr << "ERROR: provide --Nx or --n\n"; usage(argv[0]); }
        MPI_Finalize();
        return 1;
    }

    const double tol = std::atof(get_opt(argc, argv, "tol", "1e-12").c_str());
    const int maxit  = std::atoi(get_opt(argc, argv, "maxit", "500000").c_str());
    const int repeat = std::max(1, std::atoi(get_opt(argc, argv, "repeat", "1").c_str()));

    // constant reaction (NOTE: large c often makes SPD problem easier, so default stays 0)
    const double c_const = std::atof(get_opt(argc, argv, "c", "0.0").c_str());

    // const mu fallback (old behavior)
    const double mu_const = std::atof(get_opt(argc, argv, "mu", "1.0").c_str());

    const std::string prec = get_opt(argc, argv, "prec", "as2"); // identity|as|as2
    const int nparts  = std::atoi(get_opt(argc, argv, "nparts", "32").c_str());
    const int overlap = std::atoi(get_opt(argc, argv, "overlap", "1").c_str());

    std::array<Index,3> n = parse_n(n_str, dim, Nx);

    // ---- CSV handling (rank 0) ----
    fs::path csv_path = get_opt(argc, argv, "csv", "../data/output/csv/final_mpi_physical.csv");
    ensure_parent_dir(csv_path);

    const bool append = has_flag(argc, argv, "append");
    const bool need_header = (rank == 0) ? !(append && file_exists_nonempty(csv_path)) : false;

    std::ofstream os;
    if (rank == 0) {
        if (append) os.open(csv_path, std::ios::out | std::ios::app);
        else        os.open(csv_path, std::ios::out | std::ios::trunc);

        if (!os) {
            std::cerr << "ERROR: cannot open CSV file: " << csv_path.string() << "\n";
            MPI_Abort(comm, 2);
        }
        if (need_header) write_csv_header(os);
    }

    // ---- Assemble physical problem (replicated on all ranks) ----
    std::array<double,3> a = {0.0, 0.0, 0.0};
    std::array<double,3> b = {1.0, 1.0, 1.0};

    // Dirichlet boundary: keep 0 for SPD and comparability
    auto g = [](const Coord3&) { return 0.0; };

    // ---------------- Variable diffusion mu(x) knobs ----------------
    const std::string mu_type = get_opt(argc, argv, "mu-type", "const"); // const|layer|checker|inclusion
    const double mu_min  = std::atof(get_opt(argc, argv, "mu-min", "1e-4").c_str());
    const double mu_max  = std::atof(get_opt(argc, argv, "mu-max", "1.0").c_str());
    const double mu_pos  = std::atof(get_opt(argc, argv, "mu-pos", "0.5").c_str());
    const int    mu_cells = std::max(1, std::atoi(get_opt(argc, argv, "mu-cells", "8").c_str()));
    const double mu_radius = std::atof(get_opt(argc, argv, "mu-radius", "0.25").c_str());

    // mu(x): domain is [0,1]^d in this benchmark
    // Goal: allow strong coefficient contrasts -> harder for identity, AS2 more useful.
    CoeffFn mu_fun = [&](const Coord3& x) -> double {
        if (mu_type == "const") {
            return mu_const;
        }
        if (mu_type == "layer") {
            // interface normal to x-axis
            return (x[0] < mu_pos) ? mu_min : mu_max;
        }
        if (mu_type == "checker") {
            const int ix = (int)std::floor(x[0] * mu_cells);
            const int iy = (dim >= 2) ? (int)std::floor(x[1] * mu_cells) : 0;
            const int iz = (dim >= 3) ? (int)std::floor(x[2] * mu_cells) : 0;
            const int parity = (ix + iy + iz) & 1;
            return (parity == 0) ? mu_min : mu_max;
        }
        if (mu_type == "inclusion") {
            const double dx = x[0] - 0.5;
            const double dy = (dim >= 2) ? (x[1] - 0.5) : 0.0;
            const double dz = (dim >= 3) ? (x[2] - 0.5) : 0.0;
            const double r2 = dx*dx + dy*dy + dz*dz;
            return (r2 <= mu_radius*mu_radius) ? mu_min : mu_max;
        }
        // unknown -> fallback
        return mu_const;
    };

    // reaction coefficient c(x): keep constant for now
    CoeffFn c_fun = [&](const Coord3&) -> double { return c_const; };

    // ---------------- Forcing f(x) knobs ----------------
    const std::string f_type = get_opt(argc, argv, "f-type", "const"); // const|sine|checker
    const double f_amp = std::atof(get_opt(argc, argv, "f-amp", "1.0").c_str());

    FieldFn f = [&](const Coord3& x) -> double {
        if (f_type == "const") {
            return 1.0;
        }
        if (f_type == "sine") {
            const double s1 = std::sin(2.0 * M_PI * x[0]);
            const double s2 = (dim >= 2) ? std::sin(2.0 * M_PI * x[1]) : 0.0;
            const double s3 = (dim >= 3) ? std::sin(2.0 * M_PI * x[2]) : 0.0;
            return 1.0 + f_amp * (s1 + s2 + s3) / (double)dim;
        }
        if (f_type == "checker") {
            // reuse mu_cells for a forcing checker
            const int ix = (int)std::floor(x[0] * mu_cells);
            const int iy = (dim >= 2) ? (int)std::floor(x[1] * mu_cells) : 0;
            const int iz = (dim >= 3) ? (int)std::floor(x[2] * mu_cells) : 0;
            const int parity = (ix + iy + iz) & 1;
            return (parity == 0) ? 1.0 : (1.0 + f_amp);
        }
        return 1.0;
    };

    MPI_Barrier(comm);
    double tA0 = MPI_Wtime();

    // NOTE: requires the PDE constructor overload that accepts CoeffFn mu_fun, CoeffFn c_fun
    PDE p(dim, n, a, b, mu_fun, c_fun, f, g);
    auto [A_global, rhs_global] = p.assembleCOO();

    MPI_Barrier(comm);
    double tA1 = MPI_Wtime();
    double t_assemble_local = tA1 - tA0;
    double t_assemble = 0.0;
    MPI_Reduce(&t_assemble_local, &t_assemble, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    const int n_global = static_cast<int>(A_global.cols());
    const std::size_t nnz_global = static_cast<std::size_t>(A_global.nnz());

    // ---- Partition rows among ranks ----
    BlockRowPartitioner part(n_global, comm);
    const int ls    = part.ls();
    const int le    = part.le();
    const int n_loc = part.nLocal();

    // ---- Build local matrix (local-local restriction, consistent with your earlier MPI bench) ----
    MatrixCOO A_loc(static_cast<MatrixCOO::Index>(n_loc),
                    static_cast<MatrixCOO::Index>(n_loc));

    A_loc.reserve(7u * static_cast<std::size_t>(n_loc));

    A_global.forEachNZ([&](MatrixCOO::Index i, MatrixCOO::Index j, MatrixCOO::Scalar v) {
        const int gi = static_cast<int>(i);
        const int gj = static_cast<int>(j);
        if (gi >= ls && gi < le && gj >= ls && gj < le) {
            const int li = gi - ls;
            const int lj = gj - ls;
            A_loc.add(static_cast<MatrixCOO::Index>(li),
                      static_cast<MatrixCOO::Index>(lj),
                      v);
        }
    });

    std::vector<double> rhs_loc;
    part.extractLocalVector(rhs_global, rhs_loc);
    if ((int)rhs_loc.size() != n_loc) MPI_Abort(comm, 3);

    std::vector<double> x_loc(static_cast<std::size_t>(n_loc), 0.0);

    // ---- Build preconditioner ----
    Preconditioner* M = nullptr;

    if (prec == "identity") {
        M = new IdentityPreconditioner();
    } else if (prec == "as") {
        M = new AdditiveSchwarz(n_global, ls, le, nparts, overlap, comm, AdditiveSchwarz::Level::OneLevel);
        if (auto* as_ptr = dynamic_cast<AdditiveSchwarz*>(M)) {
            as_ptr->setSSORSweeps(1);
            as_ptr->setOmega(1.95);
        }
    } else if (prec == "as2") {
        M = new AdditiveSchwarz(n_global, ls, le, nparts, overlap, comm, AdditiveSchwarz::Level::TwoLevels);
        if (auto* as_ptr = dynamic_cast<AdditiveSchwarz*>(M)) {
            as_ptr->setSSORSweeps(1);
            as_ptr->setOmega(1.95);
        }
    } else {
        if (rank == 0) {
            std::cerr << "ERROR: unknown --prec '" << prec << "'. Use identity|as|as2.\n";
            usage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    // ---- Setup preconditioner ----
    double time_setup_local = 0.0, time_setup = 0.0;
    if (prec != "identity") {
        MPI_Barrier(comm);
        double t0 = MPI_Wtime();
        M->setup(A_loc);
        MPI_Barrier(comm);
        double t1 = MPI_Wtime();
        time_setup_local = t1 - t0;
    }
    MPI_Reduce(&time_setup_local, &time_setup, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    // ---- Solve ----
    PCGSolverMPI solver(A_loc, M, comm, n_global, ls, le);
    solver.setTolerance(tol);
    solver.setMaxIters(maxit);

    std::size_t its_last = 0;

    double solve_best = 1e300;
    double solve_sum  = 0.0;

    for (int rep = 0; rep < repeat; ++rep) {
        std::fill(x_loc.begin(), x_loc.end(), 0.0);

        MPI_Barrier(comm);
        double t0 = MPI_Wtime();
        its_last = solver.solve(rhs_loc, x_loc);
        MPI_Barrier(comm);
        double t1 = MPI_Wtime();

        const double t_local = t1 - t0;
        double t_rep = 0.0;
        MPI_Reduce(&t_local, &t_rep, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

        if (rank == 0) {
            solve_sum += t_rep;
            if (t_rep < solve_best) solve_best = t_rep;
        }
    }

    const double solve_avg = (rank == 0) ? (solve_sum / std::max(1, repeat)) : 0.0;

    // ---- totals + CSV (rank 0 only) ----
    if (rank == 0) {
        const double total_best = t_assemble + time_setup + solve_best;
        const double total_avg  = t_assemble + time_setup + solve_avg;

        os.setf(std::ios::scientific);
        os << std::setprecision(17);

        os << dim << ","
           << n[0] << "," << n[1] << "," << n[2] << ","
           << n_global << "," << nnz_global << ","
           << prec << ","
           << nprocs << ","
           << nparts << ","
           << overlap << ","
           << static_cast<long long>(its_last) << ","
           << tol << ","
           << maxit << ","
           << repeat << ","
           << t_assemble << ","
           << time_setup << ","
           << solve_best << ","
           << solve_avg << ","
           << total_best << ","
           << total_avg
           << "\n";

        // One-line stdout summary (kept close to previous style)
        std::cout << dim << ","
                  << n[0] << "," << n[1] << "," << n[2] << ","
                  << n_global << "," << nnz_global << ","
                  << prec << ","
                  << nprocs << ","
                  << nparts << ","
                  << overlap << ","
                  << static_cast<long long>(its_last) << ","
                  << t_assemble << ","
                  << time_setup << ","
                  << solve_best << ","
                  << total_best
                  << std::endl;
    }

    delete M;
    MPI_Finalize();
    return 0;
}
