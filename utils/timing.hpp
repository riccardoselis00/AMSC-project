#pragma once
// dd/utils/timer.hpp — tiny, CPU‑only timing helpers for perf/scalability tests
// Header‑only. C++17. No MPI, no OpenMP.
// Measure wall time + CPU time; collect named sections; export CSV.
// Example:
//   dd::util::Registry reg;
//   {
//     DD_TIMED_SCOPE("assemble", reg);
//     assemble();
//   }
//   reg.print_table();
//   reg.to_csv("run.csv");

#include <chrono>
#include <cstdint>
#include <ctime>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>

#ifdef _WIN32
  #include <windows.h>
#else
  #include <unistd.h>
#endif

namespace dd { namespace util {

// Optional: capture a hostname for bookkeeping (purely local; no MPI).
inline std::string hostname() {
#ifdef _WIN32
    char buf[256]; DWORD sz = sizeof(buf);
    if (GetComputerNameA(buf, &sz)) return std::string(buf, sz);
    return "unknown";
#else
    char buf[256];
    if (gethostname(buf, sizeof(buf)) == 0) return std::string(buf);
    const char* env = std::getenv("HOSTNAME");
    return env ? std::string(env) : std::string("unknown");
#endif
}

struct Record {
    std::string   name;            // logical section name
    double        wall_seconds = 0;// elapsed wall-clock seconds
    double        cpu_seconds  = 0;// process CPU seconds
    std::uint64_t bytes        = 0;// optional: bytes touched/moved
    std::uint64_t iters        = 0;// optional: iterations performed
    std::string   note;            // free-form tag (e.g., n=..., params)
};

// Simple wall/CPU timer
class Timer {
public:
    using clock = std::chrono::steady_clock;

    void start() {
        if (running_) return;
        running_ = true;
        t0_ = clock::now();
        cpu0_ = std::clock();
    }

    // Stop and return last interval (seconds)
    double stop() {
        if (!running_) return 0.0;
        const auto t1 = clock::now();
        const double dt = std::chrono::duration<double>(t1 - t0_).count();
        accumulated_ += dt;
        running_ = false;

        const std::clock_t cpu1 = std::clock();
        if (cpu1 != (std::clock_t)-1 && cpu0_ != (std::clock_t)-1) {
            accumulated_cpu_ += double(cpu1 - cpu0_) / double(CLOCKS_PER_SEC);
        }
        return dt;
    }

    void reset() { running_ = false; accumulated_ = 0.0; accumulated_cpu_ = 0.0; }

    double elapsed() const {
        if (!running_) return accumulated_;
        return accumulated_ + std::chrono::duration<double>(clock::now() - t0_).count();
    }

    double cpu_elapsed() const { return accumulated_cpu_; }

private:
    clock::time_point t0_{};
    std::clock_t      cpu0_{};
    double            accumulated_    {0.0};
    double            accumulated_cpu_{0.0};
    bool              running_        {false};
};

// Registry: collect records and dump as CSV
class Registry {
public:
    void add(const Record& r) { records_.push_back(r); }

    // Print CSV to stdout
    void print_table(std::ostream& os = std::cout) const {
        os << std::fixed << std::setprecision(6);
        os << "name,wall_s,cpu_s,bytes,iters,host,note\n";
        for (const auto& r : records_) {
            os << r.name << "," << r.wall_seconds << "," << r.cpu_seconds << ","
               << r.bytes << "," << r.iters << "," << hostname() << ","
               << '"' << r.note << '"' << "\n";
        }
    }

    // Write CSV to file (overwrite)
    void to_csv(const std::string& path) const {
        std::ofstream f(path);
        f << std::fixed << std::setprecision(6);
        f << "name,wall_s,cpu_s,bytes,iters,host,note\n";
        for (const auto& r : records_) {
            f << r.name << "," << r.wall_seconds << "," << r.cpu_seconds << ","
              << r.bytes << "," << r.iters << "," << hostname() << ","
              << '"' << r.note << '"' << "\n";
        }
    }

    const std::vector<Record>& data() const { return records_; }

private:
    std::vector<Record> records_;
};

// RAII scope timer (CPU-only). Starts on construction, stops on destruction.
class Scoped {
public:
    Scoped(const std::string& name,
           Registry& reg,
           std::uint64_t bytes=0,
           std::uint64_t iters=0,
           const std::string& note="")
    : name_(name), reg_(reg), bytes_(bytes), iters_(iters), note_(note) {
        timer_.start();
    }

    ~Scoped() {
        timer_.stop();
        Record r;
        r.name = name_;
        r.wall_seconds = timer_.elapsed();
        r.cpu_seconds  = timer_.cpu_elapsed();
        r.bytes = bytes_;
        r.iters = iters_;
        r.note  = note_;
        reg_.add(r);
    }

private:
    std::string name_;
    Registry&   reg_;
    std::uint64_t bytes_{};
    std::uint64_t iters_{};
    std::string note_;
    Timer       timer_{};
};

// Convenience: time a callable (averaged over `repeat` runs)
template<class F>
inline double time_it(F&& fn, int warmup=1, int repeat=3) {
    for (int i = 0; i < warmup; ++i) fn();
    Timer t; t.start();
    for (int i = 0; i < repeat; ++i) fn();
    t.stop();
    return t.elapsed() / double(repeat);
}

}} // namespace dd::util

// -----------------------------------------------------------------------------
// Helper macros for scoped timing
// -----------------------------------------------------------------------------
#define DD_CONCAT_INNER(a,b) a##b
#define DD_CONCAT(a,b) DD_CONCAT_INNER(a,b)

#define DD_TIMED_SCOPE(NAME, REGISTRY) \
    dd::util::Scoped DD_CONCAT(_dd_scope_, __LINE__)(NAME, REGISTRY)

#define DD_TIMED_SCOPE_X(NAME, REGISTRY, BYTES, ITERS, NOTE) \
    dd::util::Scoped DD_CONCAT(_dd_scope_, __LINE__)(NAME, REGISTRY, BYTES, ITERS, NOTE)

