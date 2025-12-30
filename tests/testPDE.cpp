#include "pde/pde.hpp"

// using dd::PDE;
// using dd::Coord3;

int main() {
    // 2D on (0,1)x(0,1), nx=100, ny=80
    PDE p(
        /*dim=*/2,
        /*n=*/{100, 80, 1},
        /*a=*/{0.0, 0.0, 0.0},
        /*b=*/{1.0, 1.0, 0.0},
        /*mu=*/1.0,
        /*c=*/0.0,
        /*f=*/[](const Coord3& x){ return 1.0; },
        /*g=*/[](const Coord3& x){ (void)x; return 0.0; } // homogeneous Dirichlet
    );

    auto [Acoo, rhs] = p.assembleCOO();

    // Then: COO->CSR, build graph(A), METIS partition, overlap by hops, AS/RAS+coarse, etc.
}
