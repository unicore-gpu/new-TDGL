// src/memory.cu  ─  Device memory pool allocation and GPU feature setup

#include <cstdio>
#include <algorithm>   // std::min
#include "../include/params.hpp"
#include "../include/cuda_check.hpp"
#include "device_state.cuh"
#include "kernels.cuh"
#include "memory.hpp"

// ── Pool base pointers ────────────────────────────────────────────────────────
char *h_sim_pool   = nullptr;
char *h_out_pool   = nullptr;
char *h_stage_pool = nullptr;

// ── Host-mirror pointer definitions ──────────────────────────────────────────
cuDoubleComplex *hm_Psi, *hm_Ux, *hm_Uy, *hm_W, *hm_dPsidt;
double          *hm_ImFux, *hm_ImFuy;
double          *hm_Bz, *hm_Jsx, *hm_Jsy, *hm_VOR, *hm_ENG, *hm_Ba;
cuDoubleComplex *hm_st_Psi, *hm_st_Ux;
double          *hm_st_Bz, *hm_st_Jsx, *hm_st_Jsy, *hm_st_VOR, *hm_st_ENG;

// ── SETPTR helper macro ───────────────────────────────────────────────────────
// Sets the host mirror AND uploads the pointer to the __device__ global symbol
// via cudaMemcpyToSymbol.  'q' is advanced by 'step' bytes (0 = last in pool).
#define SETPTR(sym, hm, type, q, step)                                          \
    do {                                                                         \
        (hm) = reinterpret_cast<type*>(q); (q) += (step);                       \
        CUDA_CHECK(cudaMemcpyToSymbol(sym, &(hm), sizeof(hm)));                 \
    } while (0)

// ─────────────────────────────────────────────────────────────────────────────
void allocDevice() {
    const size_t nc = NPTS * sizeof(cuDoubleComplex);
    const size_t nd = NPTS * sizeof(double);

    // ── Simulation pool: hot arrays read every step ───────────────────────────
    // Psi, Ux, Uy, W, dPsidt  (5×complex) + ImFux, ImFuy  (2×double)
    CUDA_CHECK(cudaMalloc(&h_sim_pool, 5*nc + 2*nd));
    char *q = h_sim_pool;
    SETPTR(d_Psi,    hm_Psi,    cuDoubleComplex, q, nc);
    SETPTR(d_Ux,     hm_Ux,     cuDoubleComplex, q, nc);
    SETPTR(d_Uy,     hm_Uy,     cuDoubleComplex, q, nc);
    SETPTR(d_W,      hm_W,      cuDoubleComplex, q, nc);
    SETPTR(d_dPsidt, hm_dPsidt, cuDoubleComplex, q, nc);
    SETPTR(d_ImFux,  hm_ImFux,  double,          q, nd);
    SETPTR(d_ImFuy,  hm_ImFuy,  double,          q,  0);  // last → no advance

    // ── Output pool: observable arrays + applied field ────────────────────────
    CUDA_CHECK(cudaMalloc(&h_out_pool, 5*nd + nd));
    q = h_out_pool;
    SETPTR(d_Bz,  hm_Bz,  double, q, nd);
    SETPTR(d_Jsx, hm_Jsx, double, q, nd);
    SETPTR(d_Jsy, hm_Jsy, double, q, nd);
    SETPTR(d_VOR, hm_VOR, double, q, nd);
    SETPTR(d_ENG, hm_ENG, double, q, nd);
    SETPTR(d_Ba,  hm_Ba,  double, q,  0);

    // ── Staging pool: snapshot buffers for async D→H pipelining ──────────────
    CUDA_CHECK(cudaMalloc(&h_stage_pool, 2*nc + 5*nd));
    q = h_stage_pool;
    SETPTR(d_st_Psi, hm_st_Psi, cuDoubleComplex, q, nc);
    SETPTR(d_st_Ux,  hm_st_Ux,  cuDoubleComplex, q, nc);
    SETPTR(d_st_Bz,  hm_st_Bz,  double,          q, nd);
    SETPTR(d_st_Jsx, hm_st_Jsx, double,          q, nd);
    SETPTR(d_st_Jsy, hm_st_Jsy, double,          q, nd);
    SETPTR(d_st_VOR, hm_st_VOR, double,          q, nd);
    SETPTR(d_st_ENG, hm_st_ENG, double,          q,  0);

    printf("Memory : sim %.1f MB  out %.1f MB  stage %.1f MB\n",
           (5.0*nc + 2.0*nd) / 1e6,
           6.0*nd             / 1e6,
           (2.0*nc + 5.0*nd) / 1e6);
}
#undef SETPTR

// ─────────────────────────────────────────────────────────────────────────────
void freeDevice() {
    cudaFree(h_sim_pool);
    cudaFree(h_out_pool);
    cudaFree(h_stage_pool);
    h_sim_pool = h_out_pool = h_stage_pool = nullptr;
}

// ─────────────────────────────────────────────────────────────────────────────
void uploadConstants() {
    auto up = [](const auto& sym, double v) {
        CUDA_CHECK(cudaMemcpyToSymbol(sym, &v, sizeof(v)));
    };
    up(c_dt,         DT);
    up(c_dx,         DX);
    up(c_dy,         DY);
    up(c_kappa,      KAPPA);
    up(c_sigma,      SIGMA);
    up(c_inv_k2dx2,  1.0 / (KAPPA * KAPPA * DX * DX));
    up(c_inv_k2dy2,  1.0 / (KAPPA * KAPPA * DY * DY));
    up(c_inv_dy2,    1.0 / (DY * DY));
    up(c_inv_dx2,    1.0 / (DX * DX));
    up(c_inv_k_dxdy, 1.0 / (KAPPA * DX * DY));
    up(c_k_dxdy,     KAPPA * DX * DY);
    up(c_inv_kdx,    1.0 / (KAPPA * DX));
    up(c_inv_kdy,    1.0 / (KAPPA * DY));
}

// ─────────────────────────────────────────────────────────────────────────────
void setupBlackwellFeatures(cudaStream_t s_compute) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    // ── 1. L2 cache persistence window (Ampere+ / Blackwell) ─────────────────
    // Hot data = 5 complex + 2 double arrays in h_sim_pool, accessed every step.
    // Allocating them contiguously lets one API call cover all seven.
    // RTX 5090 L2 = 96 MB; clamp to what the driver exposes.
    if (prop.persistingL2CacheMaxSize > 0) {
        size_t hotBytes = (size_t)5 * NPTS * sizeof(cuDoubleComplex)
                        + (size_t)2 * NPTS * sizeof(double);
        size_t reserveBytes = std::min((size_t)prop.persistingL2CacheMaxSize,
                                       hotBytes);
        cudaError_t err = cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize,
                                             reserveBytes);
        if (err == cudaSuccess) {
            cudaStreamAttrValue attr = {};
            attr.accessPolicyWindow.base_ptr  = h_sim_pool;
            attr.accessPolicyWindow.num_bytes = hotBytes;
            attr.accessPolicyWindow.hitRatio  = 1.0f;
            attr.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
            attr.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
            CUDA_CHECK(cudaStreamSetAttribute(s_compute,
                cudaStreamAttributeAccessPolicyWindow, &attr));
            printf("L2 persist : %.1f MB reserved (hot sim pool)\n",
                   reserveBytes / 1e6);
        } else {
            printf("L2 persist : not supported on this device (skipped)\n");
            cudaGetLastError();  // clear the error
        }
    } else {
        printf("L2 persist : persistingL2CacheMaxSize=0, skipped\n");
    }

    // ── 2. Shared-memory carveout for the stencil kernel ─────────────────────
    // Blackwell supports up to 228 KB shared mem per SM (vs 48 KB on Pascal).
    // This hint applies only to k_calDerivatives; other kernels are unaffected.
    CUDA_CHECK(cudaFuncSetAttribute(k_calDerivatives,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxShared));
    printf("Smem carve : max shared memory requested for k_calDerivatives\n");

    // ── 3. Thread Block Clusters (see OPTIMIZATION.md for implementation sketch)
    // Commented out: TBC eliminate inter-block halo global reads (~15-25% BW
    // reduction for k_calDerivatives), but require CUDA 12+ and careful tuning.
    // See OPTIMIZATION.md for the full implementation guide.
}
