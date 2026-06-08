# TDGL Simulation Run Agent

## Purpose
You are a build-and-run agent for the TDGL CUDA superconductor solver.
Given a target machine and desired physics configuration, you will:
1. Verify prerequisites
2. Build the binary
3. Configure simulation parameters
4. Launch the simulation and monitor progress
5. Confirm output is complete

---

## Step 0 — Prerequisites

### Required software
| Component | Minimum version | Check command |
|-----------|----------------|---------------|
| CUDA Toolkit | 12.x | `nvcc --version` |
| GCC / G++ | 11+ | `g++ --version` |
| Python 3 | 3.8+ | `python3 --version` |
| Python packages | — | `pip3 install numpy matplotlib scipy` |

### Required GPU
The default `Makefile` targets **sm_120 (RTX 5090 / Blackwell)**.
For other GPUs edit the `ARCH` line in `Makefile`:

| GPU | Architecture | ARCH value |
|-----|-------------|------------|
| RTX 5090 | Blackwell GB202 | `sm_120` |
| RTX 4090 | Ada Lovelace | `sm_89` |
| RTX 3090 / A100 | Ampere | `sm_86` / `sm_80` |
| RTX 2080 | Turing | `sm_75` |

Check current GPU:
```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
```

---

## Step 1 — Get the code

```bash
git clone <repo_url>   # or scp the project directory to the target machine
cd TDGL
```

Project layout:
```
TDGL/
├── include/
│   ├── params.hpp        ← ALL tuneable parameters live here
│   ├── host_tdgl.hpp
│   ├── cuda_check.hpp
│   └── complex_ops.cuh
├── src/
│   ├── main.cu           ← sweep range and SAMPLE set here
│   ├── kernels.cu/cuh
│   ├── memory.cu/hpp
│   └── simulation.cu/hpp
├── scripts/
│   ├── plot_mh.py
│   ├── plot_fields.py
│   └── analyze_vortex.py
├── figures/              ← output PNG plots and CSV
├── docs/
│   ├── TDGL_paper.md
│   └── OPTIMIZATION.md
├── agents/
│   ├── tdgl-run/
│   └── tdgl-analysis/
├── Makefile
├── CMakeLists.txt
└── README.md
```

---

## Step 2 — Configure parameters

### Grid and physics — edit `include/params.hpp`

```cpp
// Grid size presets:
//   Fast test  : NX=89,  NY=45   (~4 k points, seconds per Ba step)
//   Vortex study: NX=255, NY=127  (32 k points, ~minutes per Ba step) ← default
//   Full scale : NX=511, NY=255  (131 k points, for RTX 5090)

inline constexpr int NX  = 255;   // grid nodes in x (256 points)
inline constexpr int NY  = 127;   // grid nodes in y (128 points)
inline constexpr double DX = 0.1; // spacing (units of λ); stable if DT ≤ DX²/2
inline constexpr double DY = 0.1;
inline constexpr double DT = 2e-3; // timestep: scale with DX² when changing DX
inline constexpr double KAPPA = 2.0; // GL parameter (κ > 1/√2 → type-II)
```

**Important stability rule:** `DT ≤ DX² / 2`. If you change `DX`, update `DT` accordingly:

| DX | DT (max) |
|----|---------|
| 0.01 | 5e-5 |
| 0.05 | 1.25e-3 |
| 0.1  | 5e-3 |
| 0.2  | 2e-2 |

### Sweep range and output frequency — edit `src/main.cu`

```cpp
const SimConfig cfg = SimConfig::make(
    /*sBa*/    0.01,   // starting applied field
    /*eBa*/    2.50,   // ending applied field
    /*stepBa*/ 0.05,   // field increment per step
    /*SAMPLE*/ 100     // TDGL steps between convergence checks / file writes
);
```

`SAMPLE` controls how often convergence is tested. Higher = faster but less responsive.
Convergence is declared when |ΔMag| < 1e-4 for 5 consecutive checks.

---

## Step 3 — Build

### Using Make (recommended)
```bash
make -j$(nproc)
```

Expected output (last line):
```
Built: tdgl
```

If the build fails, common fixes:
- **`nvcc: not found`** → add CUDA to PATH: `export PATH=/usr/local/cuda/bin:$PATH`
- **`sm_120` not supported** → edit `ARCH` in `Makefile` for your GPU (see Step 0)
- **Separate compilation errors** → ensure CUDA ≥ 12 (`nvcc --version`)

### Clean rebuild
```bash
make clean && make -j$(nproc)
```

### Using CMake (alternative)
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cp tdgl ../ && cd ..
```

---

## Step 4 — Run

```bash
./tdgl
```

Output files are written to the **current directory**. Run from the project root.

### Expected console output

```
GPU: NVIDIA GeForce RTX 5090
CUDA version: 12.8 / Driver: 565.57.01
=== Kernel throughput benchmark (11000 steps, 1000 warmup) ===
  Without CUDA Graphs :   523.41 ms  (  47.585 µs/step)
  With    CUDA Graphs :   489.12 ms  (  44.465 µs/step)
  Speedup             : 1.070x
  Grid                : 32768 pts  (0.033 Mpts)
============================================================

Ba=0.010  fid=   8  Mag=-0.007690  Nvor=0.00  Eng=-160.027
Ba=0.060  fid=  16  Mag=-0.046132  Nvor=0.00  Eng=-160.289
...
```

The `fid` number is the convergence step count for that Ba value.

### Normal-state progression (κ=2, DX=0.1)
| Ba range | Expected physics |
|----------|-----------------|
| 0.01 – ~0.76 | Meissner phase: Mag ≈ −Ba, Nvor = 0 |
| ~0.81 | First flux entry (Bean-Livingston barrier broken): Nvor jumps to ~12 |
| ~0.81 – ~1.80 | Mixed (Abrikosov) state: Nvor grows, Mag approaches 0 |
| > ~1.81 | Normal state: Mag ≈ 0, Nvor ≈ 0 |

---

## Step 5 — Monitor progress

While the simulation runs, the current line on stdout shows the latest Ba step.
There is no interactive control; to stop early, press `Ctrl-C`.

To check progress from another terminal:
```bash
ls -lt *.dat | head -5   # most recently written file
wc -l Mag.dat            # number of Ba steps completed so far
```

### Typical runtimes on RTX 5090 (NX=255, NY=127, stepBa=0.05)

| Ba range | Steps to convergence (per Ba) | Time per Ba step |
|----------|-------------------------------|-----------------|
| Meissner region | ~8 SAMPLE intervals | < 5 s |
| Mixed state | ~15–30 SAMPLE intervals | ~10–30 s |
| Normal state | ~8 SAMPLE intervals | < 5 s |

Total for Ba = 0.01 → 2.5 (50 steps): roughly **10–30 minutes** on RTX 5090.

---

## Step 6 — Verify output

```bash
python3 agents/tdgl-analysis/scripts/check_data.py .
```

Expected:
```
All steps complete.
Complete steps: 50 / 50
```

If any steps are missing, re-run from the last completed Ba or investigate convergence issues.

---

## Step 7 — Analyse results

Hand off to the analysis agent:

```
See: agents/tdgl-analysis/AGENT.md
```

Quick start:
```bash
python3 scripts/plot_mh.py --data Mag.dat --out figures/mh_curve.png
python3 scripts/plot_fields.py --data . --all --field psi --out figures/psi_gallery.png
python3 scripts/analyze_vortex.py --data . --out-dir figures
```

---

## Remote machine workflow (e.g. vast.ai)

```bash
# 1. Upload code
scp -P <port> -r /path/to/TDGL root@<host>:~/tdgl

# 2. SSH in
ssh -p <port> root@<host>

# 3. Build and run
cd ~/tdgl
make -j$(nproc)
./tdgl 2>&1 | tee run.log   # save log

# 4. Download results
exit
scp -P <port> "root@<host>:~/tdgl/*.dat" ./results_zfc/
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Mag ≈ 0 for all Ba | Benchmark contaminated initial state | Check that `k_initMeshForBa(cfg.sBa)` is called after benchmark in `main.cu` |
| Nvor = 2 stuck for all low Ba | Wrong initial condition (pre-existing trapped vortices) | Same as above |
| Simulation never converges | `DT` too large for `DX` | Reduce `DT`; ensure `DT ≤ DX²/2` |
| Negative vortex counts | Conjugation bug in kinetic term | Check backward link terms in `k_calDerivatives` use `conj(U)`, not `U` |
| `nvcc` error: unsupported arch | GPU architecture mismatch | Edit `ARCH` in `Makefile` |
| `cudaErrorIllegalAddress` | Out-of-bounds memory access | Reduce grid size or check `IDX` macro |
