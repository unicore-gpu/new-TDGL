// src/device_state.cuh  ─  extern declarations of all __device__ symbols
//
// Definitions live in src/kernels.cu.  Any translation unit that needs to
// reference these symbols (e.g. memory.cu for cudaMemcpyToSymbol) includes
// this header.  NVCC separate compilation (-dc) resolves the cross-TU
// references at the device-link stage.

#pragma once
#include <cuda_runtime.h>
#include <cuComplex.h>

// ── Constant memory (scalars broadcast to all threads) ────────────────────────
extern __device__ __constant__ double c_dt;
extern __device__ __constant__ double c_dx;
extern __device__ __constant__ double c_dy;
extern __device__ __constant__ double c_kappa;
extern __device__ __constant__ double c_sigma;
extern __device__ __constant__ double c_inv_k2dx2;   // 1/(κ²·dx²)
extern __device__ __constant__ double c_inv_k2dy2;   // 1/(κ²·dy²)
extern __device__ __constant__ double c_inv_dy2;     // 1/dy²
extern __device__ __constant__ double c_inv_dx2;     // 1/dx²
extern __device__ __constant__ double c_inv_k_dxdy;  // 1/(κ·dx·dy)
extern __device__ __constant__ double c_k_dxdy;      // κ·dx·dy
extern __device__ __constant__ double c_inv_kdx;     // 1/(κ·dx)
extern __device__ __constant__ double c_inv_kdy;     // 1/(κ·dy)

// ── Simulation-state arrays (row-major: IDX(col,row) = row*STRIDE + col) ──────
extern __device__ cuDoubleComplex *d_Psi;     // order parameter ψ
extern __device__ cuDoubleComplex *d_Ux;      // link variable x
extern __device__ cuDoubleComplex *d_Uy;      // link variable y
extern __device__ cuDoubleComplex *d_W;       // plaquette gauge variable
extern __device__ cuDoubleComplex *d_dPsidt;  // ∂ψ/∂t  (intermediate)
extern __device__ double          *d_ImFux;   // Im(F_ux) for ∂Ux/∂t
extern __device__ double          *d_ImFuy;   // Im(F_uy) for ∂Uy/∂t

// ── Observable arrays ─────────────────────────────────────────────────────────
extern __device__ double *d_Bz;    // local magnetic field
extern __device__ double *d_Jsx;   // supercurrent x
extern __device__ double *d_Jsy;   // supercurrent y
extern __device__ double *d_VOR;   // vortex accumulator
extern __device__ double *d_ENG;   // energy density
extern __device__ double *d_Ba;    // applied field (scalar, device copy)

// ── Staging arrays (intermediate buffers for async D→H transfers) ─────────────
extern __device__ cuDoubleComplex *d_st_Psi;
extern __device__ cuDoubleComplex *d_st_Ux;
extern __device__ double          *d_st_Bz;
extern __device__ double          *d_st_Jsx;
extern __device__ double          *d_st_Jsy;
extern __device__ double          *d_st_VOR;
extern __device__ double          *d_st_ENG;
