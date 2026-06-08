# TDGL CUDA Solver — Optimization Notes

GPU target: **NVIDIA RTX 5090** (Blackwell, sm_100)  
Base algorithm: Gropp et al., *J. Comput. Phys.* **123**, 254 (1996)

---

## Grid

| Parameter | Value |
|-----------|-------|
| NX × NY   | 511 × 255 → 512 × 256 = 131 072 points |
| dx, dy    | 0.01 |
| dt        | 2 × 10⁻⁵ |
| κ (kappa) | 2.0 |

Change `NX`/`NY` in `cuTDGL.h` to scale up or down.  
Original paper code used 89 × 45 (4 140 points), which barely utilises 1 of 170 SMs.

---

## Optimizations applied

### 1. Row-major coalesced memory layout
**File:** `cuTDGL.h` — `IDX(col, row) = row * STRIDE + col`

Original code stored arrays as `d_Psi[col][row]` (column-major).  
With a 2-D thread block where `threadIdx.x` varies `col`, consecutive threads
in a warp accessed memory with stride `(Ny+1) × 16 = 736 bytes` → ~3 % bandwidth
utilisation.

Row-major layout makes consecutive threads (varying `col`) access consecutive
memory → stride = 16 bytes → ~95 % bandwidth utilisation.  
**This single change has the highest performance impact.**

---

### 2. Memory pools (3 × `cudaMalloc` total)
**File:** `main.cu` — `allocDevice()`

| Pool | Contents | Size (512×256) |
|------|----------|----------------|
| `d_sim_pool`   | Psi, Ux, Uy, W, dPsidt, ImFux, ImFuy | ~12 MB |
| `d_out_pool`   | Bz, Jsx, Jsy, VOR, ENG, Ba           | ~5 MB  |
| `d_stage_pool` | st_Psi, st_Ux, st_Bz, … (staging)    | ~6 MB  |

Contiguous allocation allows the L2 persistence window to cover all
simulation arrays with one API call.

---

### 3. Pinned host memory (`cudaHostAlloc`)
**File:** `cuTDGL.h` — `HOSTTDGL::allocate()`

Host arrays (`HGL.Psi`, `HGL.Bz`, …) are allocated with `cudaHostAlloc`.  
DMA transfers to pageable (`malloc`) memory require an internal CUDA staging
copy that blocks the CPU.  Pinned memory lets `cudaMemcpyAsync` use the DMA
engine directly → truly asynchronous, enabling the pipeline (see §8).

---

### 4. L2 cache persistence  *(Ampere+, most effective on Blackwell)*
**File:** `main.cu` — `setupBlackwellFeatures()`

```cpp
cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, reserveBytes);
attr.accessPolicyWindow.base_ptr  = d_sim_pool;
attr.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
cudaStreamSetAttribute(s_compute, cudaStreamAttributeAccessPolicyWindow, &attr);
```

RTX 5090 has 96 MB of L2.  The 12 MB simulation pool is pinned in L2 so
repeated reads every time step do not go to GDDR7.  Falls back silently if the
device does not expose `persistingL2CacheMaxSize`.

---

### 5. Shared-memory stencil tile
**File:** `main.cu` — `k_calDerivatives()`

Each thread block loads a `(BX+2) × (BY+2)` tile of Psi, Ux, Uy, W (with
1-cell halo) into shared memory before computing the 5-point stencil.  
Neighbours are read from shared memory (~1 cycle) rather than L2/GDDR7 (~100s
of cycles).

Tile shared-memory usage: 8 arrays × 34 × 10 × 8 B ≈ 22 KB per block.

---

### 6. Bank-conflict-free shared memory layout
**File:** `main.cu` — `k_calDerivatives()`

Complex arrays are stored as **split real/imaginary doubles**:

```cpp
__shared__ double s_Pr[TY][TX], s_Pi[TY][TX];   // Psi real / imag
__shared__ double s_Uxr[TY][TX], s_Uxi[TY][TX]; // Ux  real / imag
// …
```

With `TX = BX + 2 = 34`, thread `i` (i = 0…31) accesses column `i+1`
→ bank `i+1`.  All 32 banks distinct → zero bank conflicts.  
Using `cuDoubleComplex[TY][TX]` instead would cause 2-way bank conflicts
(16-byte elements, 32 banks × 8 B each).

---

### 7. Kernel fusion
**File:** `main.cu` — `k_calDerivatives()`

`dPsi/dt`, `ImFux`, and `ImFuy` are all computed in a single kernel, sharing
one shared-memory tile load.  This reduces the original 5 kernel launches per
step to **4**, and cuts global memory reads for Psi, Ux, Uy, W by ~40 %.

Time-step sequence:
```
k_calBC  →  k_calW  →  k_calDerivatives  →  k_updateAll
```

---

### 8. `__constant__` memory for scalar parameters
**File:** `main.cu`

```cpp
__device__ __constant__ double c_inv_k2dx2;  // 1 / (κ²dx²)
__device__ __constant__ double c_k_dxdy;     // κ·dx·dy
// … 13 constants total
```

Broadcast read to all threads with zero bandwidth cost.  Avoids recomputing
expressions like `1/(κ²dx²)` inside every thread every step.

---

### 9. Maximum shared-memory carveout  *(Blackwell)*
**File:** `main.cu` — `setupBlackwellFeatures()`

```cpp
cudaFuncSetAttribute(k_calDerivatives,
    cudaFuncAttributePreferredSharedMemoryCarveout,
    cudaSharedmemCarveoutMaxShared);
```

Requests up to 228 KB of shared memory per SM for `k_calDerivatives` at the
expense of L1 cache.  Applies only to this kernel; all others keep the default
L1/shared split.

---

### 10. Double-buffering pipeline (compute ∥ D→H transfer ∥ file I/O)
**File:** `main.cu` — `fire_snapshot` / `flush_snapshot`

```
s_compute: [──30 000 TDGL steps──][snapshot][──30 000 TDGL steps──] …
s_io:                                       [──D→H transfer──]
CPU:                                                           [CalTotal][fwrite]
```

`fire_snapshot`: takes a frozen copy of sim arrays into `d_stage_pool`, then
starts `cudaMemcpyAsync` to pinned host memory on `s_io` — returns immediately,
leaving `s_compute` free.

`flush_snapshot`: called at the *next* checkpoint; by then the transfer has
almost certainly finished, so `cudaStreamSynchronize(s_io)` has near-zero wait.
CPU post-processing (`CalTotal`, `fwrite`) then overlaps with the following GPU
compute batch.

---

### 11. `sincos()` combined instruction
**File:** `main.cu` — `k_updateAll()`

```cpp
double s, c;
sincos(phi, &s, &c);   // one SINCOS instruction, not two
```

Replaces separate `sin()` + `cos()` calls.  NVCC emits a single `MUFU.SINCOS`
instruction on sm_70+.

---

### 12. Compiler flags  *(Makefile)*

```makefile
-arch=sm_100
--generate-code arch=compute_100,code=sm_100
-O3
--use_fast_math
--extra-device-vectorization
-Xcompiler "-O3,-march=native"
```

| Flag | Effect |
|------|--------|
| `sm_100` | Native Blackwell code, no JIT at runtime |
| `--use_fast_math` | Fast MUFU approximations for trig/sqrt/div |
| `--extra-device-vectorization` | Auto-vectorise eligible loops in device code |

---

## Not applied (with reasons)

| Technique | Why not |
|-----------|---------|
| Thread Block Clusters (distributed shared memory) | Frequent subtle bugs reported in practice; commented skeleton left in `setupBlackwellFeatures()` for reference |
| `cuda::memcpy_async` / `cp.async` pipeline | Current shared-memory tile is simple enough; adding async copy pipeline would add significant code complexity for marginal gain |
| FP16 / BF16 mixed precision | Physics requires FP64 precision throughout |
| cuFFT / cuBLAS | Problem is a local stencil, not a matrix or spectral operation |
| Cooperative Groups grid sync | Would allow fusing all 4 per-step kernels into one, but requires careful deadlock analysis |

---

## Memory layout summary

```
Host (pinned)                Device
─────────────────            ─────────────────────────────────────────
HGL.Psi  (cuDoubleComplex)   d_sim_pool ──► d_Psi / d_Ux / d_Uy /
HGL.Bz   (double)                           d_W / d_dPsidt /
HGL.Jsx  (double)                           d_ImFux / d_ImFuy
HGL.Jsy  (double)
HGL.VOR  (double)            d_out_pool ──► d_Bz / d_Jsx / d_Jsy /
HGL.ENG  (double)                           d_VOR / d_ENG / d_Ba

                             d_stage_pool ► d_st_Psi / d_st_Ux /
                                            d_st_Bz / d_st_Jsx /
                                            d_st_Jsy / d_st_VOR /
                                            d_st_ENG
```

D→H data path per checkpoint:
```
d_sim_pool  ──[k_gatherOutput]──►  d_stage_pool  ──[cudaMemcpyAsync]──►  HGL.*
(GPU compute)                      (frozen snapshot)                      (pinned host)
```
