#include <mpi.h>
#include <iostream>
#include <vector>
#include <cassert>

#include "partitioner/partitioner.hpp"

// A simple unit test for BlockRowPartitioner.  This test is designed to
// exercise the key functionality of the partitioner in a parallel
// setting.  It verifies that the row partition covers the entire
// global range, that local sizes are balanced correctly, that
// extractLocalVector slices the global vector as expected, and that
// gatherVectorToRoot assembles a distributed vector back on the root.

int main(int argc, char** argv)
{
    // Initialize MPI
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int n_global = 10;
    BlockRowPartitioner part(n_global, MPI_COMM_WORLD);

    assert(part.nGlobal() == n_global);
    assert(part.size() == size);
    assert(part.rank() == rank);


    const auto& starts = part.rowStarts();
    if (rank == 0) {
        assert(static_cast<int>(starts.size()) == size + 1);
        // The first element must be 0 and the last element must equal n_global
        assert(starts[0] == 0);
        assert(starts.back() == n_global);
        // rowStarts must be non-decreasing
        for (std::size_t i = 1; i < starts.size(); ++i) {
            assert(starts[i] >= starts[i - 1]);
        }
        // The sum of part sizes must equal n_global
        int sum = 0;
        for (int p = 0; p < size; ++p) {
            sum += starts[p + 1] - starts[p];
        }
        assert(sum == n_global);
    }

    // Local range [ls, le) should correspond to part.ls() and part.le()
    int ls = part.ls();
    int le = part.le();
    int localSize = le - ls;
    // part.nLocal() should match the computed local size
    assert(part.nLocal() == localSize);
    // Each rank except possibly the first 'rem' ranks should have size base+1
    int base = n_global / size;
    int rem  = n_global % size;
    if (rank < rem) {
        assert(localSize == base + 1);
    } else {
        assert(localSize == base);
    }

    // Create a simple global vector [0, 1, 2, ..., n_global-1]
    std::vector<double> globalVec(n_global);
    for (int i = 0; i < n_global; ++i) {
        globalVec[static_cast<std::size_t>(i)] = static_cast<double>(i);
    }
    // Extract the local slice and verify its contents
    std::vector<double> localVec;
    part.extractLocalVector(globalVec, localVec);
    // localVec should be of size localSize and equal to a consecutive
    // segment of globalVec starting at index ls
    assert(static_cast<int>(localVec.size()) == localSize);
    for (int i = 0; i < localSize; ++i) {
        double expected = static_cast<double>(ls + i);
        assert(localVec[static_cast<std::size_t>(i)] == expected);
    }

    // Test gatherVectorToRoot: let each rank's local vector be filled
    // with its (1-based) rank id.  After gathering to root, the global
    // vector should reflect which rank owned each segment.
    std::vector<double> localFill(localSize);
    double fillValue = static_cast<double>(rank + 1);
    for (int i = 0; i < localSize; ++i) {
        localFill[static_cast<std::size_t>(i)] = fillValue;
    }
    
    std::vector<double> gathered;
    part.gatherVectorToRoot(localFill, gathered, /*root*/ 0);
    if (rank == 0) {
        // On root, verify that each segment [starts[p], starts[p+1]) is filled with (p+1)
        for (int p = 0; p < size; ++p) {
            int start = starts[p];
            int end   = starts[p + 1];
            double expectedValue = static_cast<double>(p + 1);
            for (int gi = start; gi < end; ++gi) {
                assert(gathered[static_cast<std::size_t>(gi)] == expectedValue);
            }
        }
    }

    MPI_Finalize();
    if (rank == 0) {
        std::cout << "All BlockRowPartitioner tests passed." << std::endl;
    }
    return 0;
}
