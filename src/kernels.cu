// src/kernels.cu  ─  TDGL device state definitions and all CUDA kernels
//
// This translation unit owns:
//   • All __device__ __constant__ scalars (c_dt, c_kappa, …)
//   • All __device__ global-memory pointers (d_Psi, d_Ux, …)
//   • The seven __global__ TDGL kernels
//   • tdgl_step() — one-step wrapper

#include "../include/params.hpp"
#include "../include/cuda_check.hpp"
#include "../include/complex_ops.cuh"
#include "device_state.cuh"
#include "kernels.cuh"

// ── Constant-memory definitions ───────────────────────────────────────────────
__device__ __constant__ double c_dt;
__device__ __constant__ double c_dx;
__device__ __constant__ double c_dy;
__device__ __constant__ double c_kappa;
__device__ __constant__ double c_sigma;
__device__ __constant__ double c_inv_k2dx2;
__device__ __constant__ double c_inv_k2dy2;
__device__ __constant__ double c_inv_dy2;
__device__ __constant__ double c_inv_dx2;
__device__ __constant__ double c_inv_k_dxdy;
__device__ __constant__ double c_k_dxdy;
__device__ __constant__ double c_inv_kdx;
__device__ __constant__ double c_inv_kdy;

// ── Device-pointer definitions ────────────────────────────────────────────────
__device__ cuDoubleComplex *d_Psi, *d_Ux, *d_Uy, *d_W, *d_dPsidt;
__device__ double          *d_ImFux, *d_ImFuy;
__device__ double          *d_Bz, *d_Jsx, *d_Jsy, *d_VOR, *d_ENG, *d_Ba;
__device__ cuDoubleComplex *d_st_Psi, *d_st_Ux;
__device__ double          *d_st_Bz, *d_st_Jsx, *d_st_Jsy, *d_st_VOR, *d_st_ENG;

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 1a — initialise all fields for zero applied field (used by benchmark)
// ─────────────────────────────────────────────────────────────────────────────
__global__ void k_initMesh() {
    int col = blockIdx.x * BX + threadIdx.x;
    int row = blockIdx.y * BY + threadIdx.y;
    if (col > NX || row > NY) return;
    int i = IDX(col, row);

    gstore(d_Psi,    i, {1.0, 0.0});
    gstore(d_Ux,     i, {1.0, 0.0});
    gstore(d_Uy,     i, {1.0, 0.0});
    gstore(d_W,      i, {0.0, 0.0});
    gstore(d_dPsidt, i, {0.0, 0.0});
    d_ImFux[i] = d_ImFuy[i] = 0.0;
    d_Bz[i] = d_Jsx[i] = d_Jsy[i] = d_VOR[i] = d_ENG[i] = 0.0;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 1b — initialise with Landau-gauge link variables for a given Ba.
//
// Landau gauge: Ax = 0, Ay = −Ba·x  →  Uy[col,row] = exp(−iκ·Ba·DX·DY·col)
//
// Interior plaquette:
//   W = Ux · Uy(c+1) · Ux*(r+1) · Uy*
//     = exp(−iκ·Ba·DX·DY·(col+1)) · exp(+iκ·Ba·DX·DY·col)
//     = exp(−iκ·Ba·DX·DY)   ≡  boundary W
//
// So W_interior = W_boundary everywhere from the very first step — no field
// shock at the boundary, and the simulation converges in O(1/σ) GL time units.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void k_initMeshForBa(double Ba) {
    int col = blockIdx.x * BX + threadIdx.x;
    int row = blockIdx.y * BY + threadIdx.y;
    if (col > NX || row > NY) return;
    int i = IDX(col, row);

    // Start with ψ = 0.05 (small nucleation seed).  This simulates field-cooled
    // cooling from T > Tc with the correct gauge field for Ba already set up:
    //   Uy[col,row] = exp(−iκ·Ba·DX·DY·col)  (Landau gauge, Ay = −Ba·x)
    // so W_interior = W_boundary from the first step (no flux shock).
    // ψ grows spontaneously via the condensation term (1−|ψ|²)ψ and naturally
    // finds the Abrikosov lattice (Ba > Hc1) or Meissner state (Ba < Hc1).
    double s, c;
    sincos(-c_k_dxdy * Ba * col, &s, &c);
    gstore(d_Psi,    i, {0.05, 0.0});
    gstore(d_Ux,     i, {1.0,  0.0});
    gstore(d_Uy,     i, {c,    s  });
    gstore(d_W,      i, {0.0,  0.0});
    gstore(d_dPsidt, i, {0.0,  0.0});
    d_ImFux[i] = d_ImFuy[i] = 0.0;
    d_Bz[i] = d_Jsx[i] = d_Jsy[i] = d_VOR[i] = d_ENG[i] = 0.0;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 2 — boundary conditions  (Js·n = 0, Neumann on ψ)
//   left  : ψ[0,r]  = Ux[0,r]   · ψ[1,r]
//   right : ψ[Nx,r] = Ux*[Nx-1,r] · ψ[Nx-1,r]
//   bottom: ψ[c,0]  = Uy[c,0]   · ψ[c,1]
//   top   : ψ[c,Ny] = Uy*[c,Ny-1] · ψ[c,Ny-1]
// ─────────────────────────────────────────────────────────────────────────────
__global__ void k_calBC() {
    int col = blockIdx.x * BX + threadIdx.x;
    int row = blockIdx.y * BY + threadIdx.y;
    if (col > NX || row > NY) return;

    if (row == 0  && col > 0 && col < NX)
        gstore(d_Psi, IDX(col, 0),
               cmul(gload(d_Uy, IDX(col, 0)), gload(d_Psi, IDX(col, 1))));
    if (row == NY && col > 0 && col < NX)
        gstore(d_Psi, IDX(col, NY),
               cmul(conj_(gload(d_Uy, IDX(col, NY-1))), gload(d_Psi, IDX(col, NY-1))));
    if (col == 0  && row > 0 && row < NY)
        gstore(d_Psi, IDX(0, row),
               cmul(gload(d_Ux, IDX(0, row)), gload(d_Psi, IDX(1, row))));
    if (col == NX && row > 0 && row < NY)
        gstore(d_Psi, IDX(NX, row),
               cmul(conj_(gload(d_Ux, IDX(NX-1, row))), gload(d_Psi, IDX(NX-1, row))));
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 3 — plaquette gauge variable W
//   Interior: W[c,r] = Ux[c,r] · Uy[c+1,r] · Ux*[c,r+1] · Uy*[c,r]
//   Boundary: W = 1 − i·κ·Ba·dx·dy  (Landau gauge, uniform applied field)
//   Defined for c ∈ [0,Nx−1], r ∈ [0,Ny−1].
// ─────────────────────────────────────────────────────────────────────────────
__global__ void k_calW(const double* __restrict__ Ba) {
    int col = blockIdx.x * BX + threadIdx.x;
    int row = blockIdx.y * BY + threadIdx.y;
    if (col >= NX || row >= NY) return;

    if (row == 0 || row == NY-1 || col == 0 || col == NX-1) {
        gstore(d_W, IDX(col, row), {1.0, -c_k_dxdy * (*Ba)});
        return;
    }
    double2 w = cmul(gload(d_Ux, IDX(col,   row  )),
                     gload(d_Uy, IDX(col+1, row  )));
    w = cmul(w, conj_(gload(d_Ux, IDX(col,   row+1))));
    w = cmul(w, conj_(gload(d_Uy, IDX(col,   row  ))));
    gstore(d_W, IDX(col, row), w);
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 4 (FUSED) — dψ/dt, ImFux, ImFuy using shared-memory tile
//
// Shared-memory layout (split re/im doubles → zero bank conflicts):
//   s_Pr/Pi  [TY][TX]  ← ψ  re/im     TX=34 doubles, bank (i+1) per thread i
//   s_Uxr/xi [TY][TX]  ← Ux re/im
//   s_Uyr/yi [TY][TX]  ← Uy re/im
//   s_Wr/Wi  [TY][TX]  ← W  re/im
// ─────────────────────────────────────────────────────────────────────────────
__global__ void k_calDerivatives() {
    __shared__ double s_Pr [TY][TX], s_Pi [TY][TX];
    __shared__ double s_Uxr[TY][TX], s_Uxi[TY][TX];
    __shared__ double s_Uyr[TY][TX], s_Uyi[TY][TX];
    __shared__ double s_Wr [TY][TX], s_Wi [TY][TX];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int col = blockIdx.x * BX + tx;
    const int row = blockIdx.y * BY + ty;
    const int lx  = tx + 1;
    const int ly  = ty + 1;

    // Clamp helper: keep halo indices in-bounds at domain edges
    auto cl = [](int v, int lo, int hi) { return max(lo, min(v, hi)); };

    // Load all four fields into the shared-memory tile centre
    auto loadField = [&](int c, int r, int slx, int sly) {
        int i = cl(r, 0, NY) * STRIDE + cl(c, 0, NX);
        double2 p  = gload(d_Psi, i);
        double2 ux = gload(d_Ux,  i);
        double2 uy = gload(d_Uy,  i);
        double2 w  = gload(d_W,   i);
        s_Pr [sly][slx] = p.x;  s_Pi [sly][slx] = p.y;
        s_Uxr[sly][slx] = ux.x; s_Uxi[sly][slx] = ux.y;
        s_Uyr[sly][slx] = uy.x; s_Uyi[sly][slx] = uy.y;
        s_Wr [sly][slx] = w.x;  s_Wi [sly][slx] = w.y;
    };

    loadField(col,   row,   lx,    ly   );   // centre
    if (tx == 0)    loadField(col-1, row,   0,     ly   );   // left  halo
    if (tx == BX-1) loadField(col+1, row,   TX-1,  ly   );   // right halo
    if (ty == 0)    loadField(col,   row-1, lx,    0    );   // bottom halo
    if (ty == BY-1) loadField(col,   row+1, lx,    TY-1 );   // top   halo
    __syncthreads();

    if (col > NX || row > NY) return;

    // ── ∂ψ/∂t  (interior points only) ────────────────────────────────────────
    if (col > 0 && col < NX && row > 0 && row < NY) {
        double2 psi  = {s_Pr[ly][lx], s_Pi[ly][lx]};
        double  psi2 = cnorm2(psi);

        // Condensation: (1 − |ψ|²)ψ
        double2 dp = cscale(psi, 1.0 - psi2);

        // x kinetic: [Ux·ψ(c+1) + Ux*(c−1)·ψ(c−1) − 2ψ] / (κ²·dx²)
        double2 kinx = csub(
            cadd(cmul({s_Uxr[ly][lx],    s_Uxi[ly][lx]   }, {s_Pr[ly][lx+1], s_Pi[ly][lx+1]}),
                 cmul({s_Uxr[ly][lx-1], -s_Uxi[ly][lx-1]}, {s_Pr[ly][lx-1], s_Pi[ly][lx-1]})),
            cscale(psi, 2.0));
        dp = cadd(dp, cscale(kinx, c_inv_k2dx2));

        // y kinetic: [Uy·ψ(r+1) + Uy*(r−1)·ψ(r−1) − 2ψ] / (κ²·dy²)
        double2 kiny = csub(
            cadd(cmul({s_Uyr[ly][lx],    s_Uyi[ly][lx]   }, {s_Pr[ly+1][lx], s_Pi[ly+1][lx]}),
                 cmul({s_Uyr[ly-1][lx], -s_Uyi[ly-1][lx]}, {s_Pr[ly-1][lx], s_Pi[ly-1][lx]})),
            cscale(psi, 2.0));
        dp = cadd(dp, cscale(kiny, c_inv_k2dy2));

        gstore(d_dPsidt, IDX(col, row), dp);
    }

    // ── ImFux = Im[(W[c,r]−W[c,r−1])/dy² + Ux·ψ*·ψ(c+1)] ───────────────────
    if (col < NX && row > 0 && row < NY) {
        double2 dW  = {s_Wr[ly][lx] - s_Wr[ly-1][lx],
                       s_Wi[ly][lx] - s_Wi[ly-1][lx]};
        double2 psi = {s_Pr[ly][lx],   s_Pi[ly][lx]  };
        double2 Pp1 = {s_Pr[ly][lx+1], s_Pi[ly][lx+1]};
        double2 Ux  = {s_Uxr[ly][lx],  s_Uxi[ly][lx] };
        double2 t   = cadd(cscale(dW, c_inv_dy2), cmul(Ux, cmul(conj_(psi), Pp1)));
        d_ImFux[IDX(col, row)] = t.y;
    }

    // ── ImFuy = Im[(W[c−1,r]−W[c,r])/dx² + Uy·ψ*·ψ(r+1)] ───────────────────
    if (col > 0 && col < NX && row < NY) {
        double2 dW  = {s_Wr[ly][lx-1] - s_Wr[ly][lx],
                       s_Wi[ly][lx-1] - s_Wi[ly][lx]};
        double2 psi = {s_Pr[ly][lx],   s_Pi[ly][lx]  };
        double2 Prp1= {s_Pr[ly+1][lx], s_Pi[ly+1][lx]};
        double2 Uy  = {s_Uyr[ly][lx],  s_Uyi[ly][lx] };
        double2 t   = cadd(cscale(dW, c_inv_dx2), cmul(Uy, cmul(conj_(psi), Prp1)));
        d_ImFuy[IDX(col, row)] = t.y;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 5 — forward-Euler update for ψ, Ux, Uy
//   ψ_new  = ψ + (dψ/dt)·dt
//   Ux_new = Ux · exp(−i·ImFux·dt/σ)   (gauge field — kept unimodular)
//   Uy_new = Uy · exp(−i·ImFuy·dt/σ)
// ─────────────────────────────────────────────────────────────────────────────
__global__ void k_updateAll() {
    int col = blockIdx.x * BX + threadIdx.x;
    int row = blockIdx.y * BY + threadIdx.y;
    if (col > NX || row > NY) return;
    int i = IDX(col, row);

    if (col > 0 && col < NX && row > 0 && row < NY)
        gstore(d_Psi, i, cadd(gload(d_Psi, i), cscale(gload(d_dPsidt, i), c_dt)));

    if (col < NX && row > 0 && row < NY) {
        double s, c;
        sincos(-c_dt * d_ImFux[i] / c_sigma, &s, &c);
        double2 u = cmul(gload(d_Ux, i), {c, s});
        gstore(d_Ux, i, cscale(u, 1.0 / sqrt(cnorm2(u))));  // renormalise
    }

    if (col > 0 && col < NX && row < NY) {
        double s, c;
        sincos(-c_dt * d_ImFuy[i] / c_sigma, &s, &c);
        double2 u = cmul(gload(d_Uy, i), {c, s});
        gstore(d_Uy, i, cscale(u, 1.0 / sqrt(cnorm2(u))));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 6 — observables: Bz, energy density, supercurrent, vortex accumulator
// ─────────────────────────────────────────────────────────────────────────────
__global__ void k_calObservables(const double* __restrict__ Ba) {
    int col = blockIdx.x * BX + threadIdx.x;
    int row = blockIdx.y * BY + threadIdx.y;
    if (col > NX || row > NY) return;
    int    i  = IDX(col, row);
    double ba = *Ba;

    d_VOR[i] = 0.0;

    // Bz = −Im(W) / (κ·dx·dy)
    if (col < NX && row < NY)
        d_Bz[i] = -gload(d_W, i).y * c_inv_k_dxdy;

    // Energy density
    double CondEng = 0.0, KEng = 0.0, MagEng = 0.0;
    if (col > 0 && col < NX && row > 0 && row < NY) {
        double psi2 = cnorm2(gload(d_Psi, i));
        CondEng = 0.5 * psi2 * psi2 - psi2;
        double2 tx = csub(cmul(gload(d_Ux, i), gload(d_Psi, IDX(col+1, row))), gload(d_Psi, i));
        double2 ty = csub(cmul(gload(d_Uy, i), gload(d_Psi, IDX(col, row+1))), gload(d_Psi, i));
        KEng = cnorm2(tx) * c_inv_k2dx2 + cnorm2(ty) * c_inv_k2dy2;
    }
    if (col < NX && row < NY)
        MagEng = d_Bz[i] * d_Bz[i] - 2.0 * d_Bz[i] * ba;
    d_ENG[i] = (MagEng + KEng + CondEng) * c_dx * c_dy;

    // Supercurrent
    if (col < NX && row < NY) {
        double2 tj = cmul(gload(d_Ux, i), cmul(conj_(gload(d_Psi, i)), gload(d_Psi, IDX(col+1, row))));
        d_Jsx[i] = tj.y * c_inv_kdx;
        tj = cmul(gload(d_Uy, i), cmul(conj_(gload(d_Psi, i)), gload(d_Psi, IDX(col, row+1))));
        d_Jsy[i] = tj.y * c_inv_kdy;
    }

    // Vortex count accumulator (boundary line integral via Stokes theorem)
    if ((row == 1 || row == NY-1) && col < NX && col > 0) {
        double2 u   = gload(d_Ux, i);
        double  Ax  = asin(max(-1.0, min(1.0, -u.y / sqrt(cnorm2(u))))) / (c_kappa * c_dx);
        double  psi2= max(cnorm2(gload(d_Psi, i)), 1e-12);
        d_VOR[i]    = (d_Jsx[i] / psi2 + Ax) * c_dx;
    }
    if ((col == 1 || col == NX-1) && row < NY && row > 0) {
        double2 u   = gload(d_Uy, i);
        double  Ay  = asin(max(-1.0, min(1.0, -u.y / sqrt(cnorm2(u))))) / (c_kappa * c_dy);
        double  psi2= max(cnorm2(gload(d_Psi, i)), 1e-12);
        d_VOR[i]    = (d_Jsy[i] / psi2 + Ay) * c_dy;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 7 — gather live arrays into staging buffers for async D→H copy
// ─────────────────────────────────────────────────────────────────────────────
__global__ void k_gatherOutput(cuDoubleComplex* __restrict__ oPsi,
                                cuDoubleComplex* __restrict__ oUx,
                                double* __restrict__ oBz,
                                double* __restrict__ oJsx,
                                double* __restrict__ oJsy,
                                double* __restrict__ oVOR,
                                double* __restrict__ oENG) {
    int col = blockIdx.x * BX + threadIdx.x;
    int row = blockIdx.y * BY + threadIdx.y;
    if (col > NX || row > NY) return;
    int i = IDX(col, row);
    oPsi[i] = d_Psi[i];  oUx[i]  = d_Ux[i];
    oBz[i]  = d_Bz[i];   oJsx[i] = d_Jsx[i]; oJsy[i] = d_Jsy[i];
    oVOR[i] = d_VOR[i];  oENG[i] = d_ENG[i];
}

// ─────────────────────────────────────────────────────────────────────────────
// One TDGL time step: BC → W → derivatives → update
// ─────────────────────────────────────────────────────────────────────────────
// hm_Ba is the host-side mirror of d_Ba (a device pointer to the Ba scalar).
// We pass it as a kernel argument so k_calW can dereference it on the device.
// Using the host mirror avoids a direct __device__-variable read in host code.
#include "memory.hpp"

void tdgl_step(dim3 grid, dim3 block, cudaStream_t stream) {
    k_calBC          <<<grid, block, 0, stream>>>();
    k_calW           <<<grid, block, 0, stream>>>(hm_Ba);
    k_calDerivatives <<<grid, block, 0, stream>>>();
    k_updateAll      <<<grid, block, 0, stream>>>();
}
