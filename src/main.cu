// src/main.cu  ─  Entry point for the GPU TDGL solver
//
// Responsibilities: GPU info, initialisation, benchmark, field sweep, cleanup.
// All physics lives in kernels.cu / simulation.cu.

#include <cstdio>
#include "../include/params.hpp"
#include "../include/cuda_check.hpp"
#include "../include/host_tdgl.hpp"
#include "kernels.cuh"
#include "memory.hpp"
#include "simulation.hpp"

int main() {
    // ── GPU information ───────────────────────────────────────────────────────
    print_gpu_info();
    printf("Grid : %d×%d  (%d pts)  dx=%.3g  dy=%.3g  dt=%.3g  κ=%.1f\n\n",
           NX+1, NY+1, NPTS, DX, DY, DT, KAPPA);

    // ── Initialise device ─────────────────────────────────────────────────────
    uploadConstants();
    allocDevice();

    // ── CUDA streams: one for compute, one for async D→H I/O ─────────────────
    cudaStream_t s_compute, s_io;
    CUDA_CHECK(cudaStreamCreate(&s_compute));
    CUDA_CHECK(cudaStreamCreate(&s_io));

    // ── Kernel launch configuration ───────────────────────────────────────────
    const dim3 block(BX, BY);
    const dim3 grid((NX + BX) / BX, (NY + BY) / BY);   // covers NX+1, NY+1 pts

    // ── Blackwell features: L2 persistence + shared-memory carveout ──────────
    setupBlackwellFeatures(s_compute);

    // ── Initialise mesh ───────────────────────────────────────────────────────
    k_initMesh<<<grid, block, 0, s_compute>>>();
    CUDA_CHECK(cudaStreamSynchronize(s_compute));

    // ── Benchmark: pure kernel throughput (no I/O) ────────────────────────────
    // NOTE: benchmark evolves the mesh at Ba=10.5 (>> Hc2 ≈ 2.83 for κ=2),
    // which drives ψ toward the normal state.  Re-initialise before the
    // physics simulation so it always starts from a known superconducting state.
    run_benchmark(grid, block, s_compute);

    // ── Simulation: M-H curve sweep ───────────────────────────────────────────
    HOSTTDGL hgl;
    const SimConfig cfg = SimConfig::make(
        /*sBa*/    0.01,
        /*eBa*/    2.5,
        /*stepBa*/ 0.05,
        /*SAMPLE*/ 100
    );

    // ZFC initial condition: ψ = 0.05 (nucleation seed), Ux = 1,
    // Uy = Landau-gauge for sBa so W_interior = W_boundary from the first step.
    // The sweep then carries state forward at each Ba step — vortices enter
    // one-by-one as Ba crosses Hc1, just as in a real ZFC measurement.
    k_initMeshForBa<<<grid, block, 0, s_compute>>>(cfg.sBa);
    CUDA_CHECK(cudaStreamSynchronize(s_compute));

    run_simulation(hgl, s_compute, s_io, grid, block, cfg);

    // ── Cleanup ───────────────────────────────────────────────────────────────
    freeDevice();
    CUDA_CHECK(cudaStreamDestroy(s_compute));
    CUDA_CHECK(cudaStreamDestroy(s_io));
    return 0;
}
