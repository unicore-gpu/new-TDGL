// include/host_tdgl.hpp  ─  Host-side state: pinned memory, observables, file I/O
//
// HOSTTDGL owns all pinned host buffers.  The GPU writes into these via
// cudaMemcpyAsync; the host then reduces observables and writes output files.
// Using cudaHostAlloc (pinned / page-locked) enables DMA transfers that overlap
// with GPU computation (pipelining in simulation.cu).

#pragma once
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cstdio>
#include <cmath>
#include "params.hpp"
#include "cuda_check.hpp"

class HOSTTDGL {
public:
    // ── Pinned output buffers (filled by async D→H copies) ────────────────────
    cuDoubleComplex *Psi;   // order-parameter magnitude |ψ|
    cuDoubleComplex *Ux;    // link variable x  (saved for restart / diagnostics)
    double          *Bz;    // local magnetic field  Bz
    double          *Jsx;   // supercurrent x-component
    double          *Jsy;   // supercurrent y-component
    double          *VOR;   // vortex-count accumulator
    double          *ENG;   // energy density

    // ── Scalar observables (computed by CalTotal) ─────────────────────────────
    double NumOfVortices = 0.0;
    double SysEng        = 0.0;
    double magnetization = 0.0;
    double nameBa        = 0.0;   // current Ba value (for file naming)

    HOSTTDGL()  { allocate(); }
    ~HOSTTDGL() { free_mem(); }

    // ── Reduce observables from pinned arrays ─────────────────────────────────
    // Called after each D→H transfer to extract scalar diagnostics.
    void CalTotal(double Ba) {
        NumOfVortices = 0.0;
        SysEng        = 0.0;
        double sumBz  = 0.0;
        int    cnt    = 0;

        // Vortex count via boundary line integral (Stokes theorem)
        for (int i = 1; i < NX-1; i++) {
            NumOfVortices += VOR[IDX(i, 1)];
            NumOfVortices -= VOR[IDX(i, NY-1)];
        }
        for (int j = 1; j < NY-1; j++) {
            NumOfVortices -= VOR[IDX(1,    j)];
            NumOfVortices += VOR[IDX(NX-1, j)];
        }
        NumOfVortices *= KAPPA / (2.0 * M_PI);

        for (int j = 0; j < NY; j++)
            for (int i = 0; i < NX; i++) {
                SysEng += ENG[IDX(i, j)];
                sumBz  += Bz [IDX(i, j)];
                ++cnt;
            }
        magnetization = sumBz / cnt - Ba;
    }

    // ── File output (tab-separated, compatible with Python analysis scripts) ──
    // File naming: "{prefix}{Ba*100}_{step_id}.dat"
    void OutputPsi(int id) const {
        char fname[64];
        snprintf(fname, sizeof(fname), "Psi%.0f_%d.dat", nameBa * 100, id);
        FILE *fp = fopen(fname, "w");
        write_header(fp);
        for (int j = 0; j <= NY; j++) {
            fprintf(fp, "%.6g", j * DY);
            for (int i = 0; i <= NX; i++)
                fprintf(fp, "\t%.8f", cuCabs(Psi[IDX(i, j)]));
            fprintf(fp, "\n");
        }
        fclose(fp);
    }

    void OutputBz (int id) const { write_double_field("Bz",  Bz,  id); }
    void OutputJsx(int id) const { write_double_field("Jsx", Jsx, id); }
    void OutputJsy(int id) const { write_double_field("Jsy", Jsy, id); }
    void OutputEng(int id) const { write_double_field("ENG", ENG, id); }

private:
    void allocate() {
        const size_t nc = NPTS * sizeof(cuDoubleComplex);
        const size_t nd = NPTS * sizeof(double);
        CUDA_CHECK(cudaHostAlloc(&Psi, nc, cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&Ux,  nc, cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&Bz,  nd, cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&Jsx, nd, cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&Jsy, nd, cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&VOR, nd, cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&ENG, nd, cudaHostAllocDefault));
    }

    void free_mem() {
        cudaFreeHost(Psi); cudaFreeHost(Ux);
        cudaFreeHost(Bz);  cudaFreeHost(Jsx); cudaFreeHost(Jsy);
        cudaFreeHost(VOR); cudaFreeHost(ENG);
    }

    void write_header(FILE *fp) const {
        fprintf(fp, "0.0");
        for (int i = 0; i <= NX; i++) fprintf(fp, "\t%.6g", i * DX);
        fprintf(fp, "\n");
    }

    void write_double_field(const char *prefix, const double *field, int id) const {
        char fname[64];
        snprintf(fname, sizeof(fname), "%s%.0f_%d.dat", prefix, nameBa * 100, id);
        FILE *fp = fopen(fname, "w");
        write_header(fp);
        for (int j = 0; j <= NY; j++) {
            fprintf(fp, "%.6g", j * DY);
            for (int i = 0; i <= NX; i++)
                fprintf(fp, "\t%.8f", field[IDX(i, j)]);
            fprintf(fp, "\n");
        }
        fclose(fp);
    }
};
