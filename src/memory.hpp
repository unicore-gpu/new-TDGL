// src/memory.hpp  ─  Device memory management: public API declarations
//
// Provides allocDevice / freeDevice / uploadConstants / setupBlackwellFeatures.
// Also exposes host-mirror pointers (hm_*) used by simulation.cu for kernel
// arguments and async D→H memcpys.

#pragma once
#include <cuda_runtime.h>
#include <cuComplex.h>

// ── Pool base pointers (host-accessible, for cudaFree + L2 window) ─────────
extern char *h_sim_pool;    // Psi, Ux, Uy, W, dPsidt, ImFux, ImFuy  (hot)
extern char *h_out_pool;    // Bz, Jsx, Jsy, VOR, ENG, Ba            (output)
extern char *h_stage_pool;  // staging buffers for async D→H          (I/O)

// ── Host-mirror pointers (device addresses, readable from host code) ────────
// Passed as kernel arguments and to cudaMemcpyAsync.
extern cuDoubleComplex *hm_Psi, *hm_Ux, *hm_Uy, *hm_W, *hm_dPsidt;
extern double          *hm_ImFux, *hm_ImFuy;
extern double          *hm_Bz, *hm_Jsx, *hm_Jsy, *hm_VOR, *hm_ENG, *hm_Ba;
extern cuDoubleComplex *hm_st_Psi, *hm_st_Ux;
extern double          *hm_st_Bz, *hm_st_Jsx, *hm_st_Jsy, *hm_st_VOR, *hm_st_ENG;

// ── API ───────────────────────────────────────────────────────────────────────

// Allocate all three device memory pools and set __device__ pointer symbols.
// Prints pool sizes to stdout.
void allocDevice();

// Free all device memory pools.
void freeDevice();

// Upload compile-time scalar constants to __device__ __constant__ memory.
void uploadConstants();

// Configure Blackwell-specific GPU features: L2 cache persistence window
// over the hot simulation pool, and shared-memory carveout for the stencil kernel.
// Safe to call on non-Blackwell devices: features are skipped with a warning.
void setupBlackwellFeatures(cudaStream_t s_compute);
