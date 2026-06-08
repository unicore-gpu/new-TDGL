// include/params.hpp  ─  Compile-time physical and grid parameters
//
// All tuneable knobs are here.  Every other source file includes this header;
// nothing else needs to change when resizing the grid or adjusting physics.
//
// Reference: Gropp et al., J. Comput. Phys. 123, 254 (1996)

#pragma once

// ── Grid ──────────────────────────────────────────────────────────────────────
// Rule of thumb for RTX 5090 (170 SMs, sm_120):
//   NPTS > 170 × 256 = 43 520 keeps all SMs busy.
//
//   Fast run / testing : NX=89,  NY=45  (  90 ×  46 =   4 140 pts)
//   Vortex lattice     : NX=255, NY=127 ( 256 × 128 =  32 768 pts)  ← default
//   Full-scale RTX5090 : NX=511, NY=255 ( 512 × 256 = 131 072 pts)
inline constexpr int NX     = 255;         // grid nodes in x  (256 points)
inline constexpr int NY     = 127;         // grid nodes in y  (128 points)
inline constexpr int STRIDE = NX + 1;      // row-major row stride
inline constexpr int NPTS   = (NX+1)*(NY+1);

// Row-major flat index  (col = x = fast axis → coalesced loads)
#define IDX(col, row)  ((row) * STRIDE + (col))

// ── Physics ───────────────────────────────────────────────────────────────────
inline constexpr double KAPPA = 2.0;       // Ginzburg-Landau parameter κ
inline constexpr double SIGMA = 1.0;       // normal-state conductivity σ
inline constexpr double DX    = 0.1;       // grid spacing in x  (units of λ)
inline constexpr double DY    = 0.1;       // grid spacing in y  (units of λ)

// Timestep: for numerical stability require DT ≤ DX² / 2.
// DX=0.01 → DT=2e-5;  DX=0.1 → DT=2e-3  (scaled by DX²)
inline constexpr double DT    = 2e-3;

// ── CUDA kernel tile dimensions ───────────────────────────────────────────────
// BX = 32 = warp width → perfectly coalesced row-major loads.
// BY = 8  → 256 threads/block = 8 warps; good register pressure on Blackwell.
// TX = BX+2, TY = BY+2  → shared-memory tile including 1-cell halo on every side.
// With TX=34 doubles, thread i accesses bank (i+1): zero bank conflicts.
inline constexpr int BX = 32;
inline constexpr int BY = 8;
inline constexpr int TX = BX + 2;   // shared-memory tile width  (with halo)
inline constexpr int TY = BY + 2;   // shared-memory tile height (with halo)
