// src/kernels.cuh  ─  Public declarations of all TDGL CUDA kernels and helpers
//
// Include this header in any translation unit that launches kernels or calls
// tdgl_step().  The implementations are in src/kernels.cu.

#pragma once
#include <cuda_runtime.h>
#include <cuComplex.h>

// ── Kernel declarations ───────────────────────────────────────────────────────

// k_initMesh: zero-field initialisation (ψ=1, Ux=Uy=1) — used before benchmark
__global__ void k_initMesh();

// k_initMeshForBa: Landau-gauge initialisation for the given applied field Ba.
// Sets ψ=1, Ux=1, Uy[col,row]=exp(−iκ·Ba·DX·DY·col) so that interior W equals
// the boundary W from the very first step — no field shock, fast convergence.
__global__ void k_initMeshForBa(double Ba);

// k_calBC: apply Js·n = 0 Neumann boundary conditions on ψ
__global__ void k_calBC();

// k_calW: compute plaquette gauge variable W (interior) and apply Ba (boundary)
__global__ void k_calW(const double* __restrict__ Ba);

// k_calDerivatives: fused kernel — computes dψ/dt, ImFux, ImFuy using shared-memory tile
__global__ void k_calDerivatives();

// k_updateAll: forward-Euler update for ψ, Ux, Uy
__global__ void k_updateAll();

// k_calObservables: compute Bz, energy, supercurrent, vortex accumulator
__global__ void k_calObservables(const double* __restrict__ Ba);

// k_gatherOutput: copy live arrays → staging buffers (for async D→H transfer)
__global__ void k_gatherOutput(cuDoubleComplex* __restrict__ oPsi,
                                cuDoubleComplex* __restrict__ oUx,
                                double* __restrict__ oBz,
                                double* __restrict__ oJsx,
                                double* __restrict__ oJsy,
                                double* __restrict__ oVOR,
                                double* __restrict__ oENG);

// ── One-step convenience wrapper ─────────────────────────────────────────────
// Launches the four-kernel TDGL update sequence on the given stream.
void tdgl_step(dim3 grid, dim3 block, cudaStream_t stream);
