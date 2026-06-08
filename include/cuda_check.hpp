// include/cuda_check.hpp  ─  CUDA error-checking utilities

#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// ── CUDA_CHECK ────────────────────────────────────────────────────────────────
// Wraps any CUDA runtime call.  On error: prints file/line/message and aborts.
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error  %s:%d  %s\n",                         \
                    __FILE__, __LINE__, cudaGetErrorString(_e));                \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// ── print_gpu_info ────────────────────────────────────────────────────────────
// Prints device name, compute capability, memory, and peak memory bandwidth.
inline void print_gpu_info() {
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    double bw = p.memoryClockRate * 1e3 * (double)p.memoryBusWidth / 8.0 / 1e9;
    printf("GPU  : %s  (sm_%d%d)  %.1f GB  peak BW %.0f GB/s\n",
           p.name, p.major, p.minor, p.totalGlobalMem / 1e9, bw);
}
