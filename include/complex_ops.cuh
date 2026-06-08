// include/complex_ops.cuh  ─  Lightweight __device__ complex arithmetic
//
// Uses double2 (re, im) directly instead of cuDoubleComplex, which avoids the
// overhead of cuComplex library calls and allows the compiler to keep operands
// in registers across fused operations.
//
// All functions are __forceinline__ to eliminate call overhead in hot loops.

#pragma once
#include <cuda_runtime.h>
#include <cuComplex.h>   // cuDoubleComplex type (for interface compatibility)

// ── Arithmetic ────────────────────────────────────────────────────────────────
__device__ __forceinline__ double2 cmul(double2 a, double2 b) {
    return {a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x};
}
__device__ __forceinline__ double2 cadd(double2 a, double2 b) {
    return {a.x + b.x, a.y + b.y};
}
__device__ __forceinline__ double2 csub(double2 a, double2 b) {
    return {a.x - b.x, a.y - b.y};
}
__device__ __forceinline__ double2 cscale(double2 a, double s) {
    return {a.x * s, a.y * s};
}
__device__ __forceinline__ double2 conj_(double2 a) {
    return {a.x, -a.y};
}
__device__ __forceinline__ double cnorm2(double2 a) {
    return a.x*a.x + a.y*a.y;
}

// ── Global-memory load/store ──────────────────────────────────────────────────
// Treat cuDoubleComplex* as double2* to bypass cuComplex struct overhead.
__device__ __forceinline__ double2 gload(const cuDoubleComplex* __restrict__ p, int i) {
    return {p[i].x, p[i].y};
}
__device__ __forceinline__ void gstore(cuDoubleComplex* __restrict__ p, int i, double2 v) {
    p[i].x = v.x;
    p[i].y = v.y;
}
