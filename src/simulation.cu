// src/simulation.cu  ─  Benchmark and field-sweep simulation loop

#include <cstdio>
#include <cmath>
#include <chrono>
#include "../include/params.hpp"
#include "../include/cuda_check.hpp"
#include "../include/host_tdgl.hpp"
#include "kernels.cuh"
#include "memory.hpp"
#include "simulation.hpp"

// ─────────────────────────────────────────────────────────────────────────────
// Benchmark: pure kernel throughput, without and with CUDA Graphs
// ─────────────────────────────────────────────────────────────────────────────
void run_benchmark(dim3 grid, dim3 block, cudaStream_t s_compute) {
    constexpr int    N_WARMUP  = 500;
    constexpr int    N_BENCH   = 5000;
    constexpr double BENCH_BA  = 10.5;

    CUDA_CHECK(cudaMemcpyAsync(hm_Ba, &BENCH_BA, sizeof(double),
                               cudaMemcpyHostToDevice, s_compute));

    // Warmup
    for (int i = 0; i < N_WARMUP; ++i) tdgl_step(grid, block, s_compute);
    CUDA_CHECK(cudaStreamSynchronize(s_compute));

    // ── Without CUDA Graphs ───────────────────────────────────────────────────
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_BENCH; ++i) tdgl_step(grid, block, s_compute);
    CUDA_CHECK(cudaStreamSynchronize(s_compute));
    auto t1  = std::chrono::high_resolution_clock::now();
    double ms_base = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // ── With CUDA Graphs ──────────────────────────────────────────────────────
    // Capture one tdgl_step as a graph, then replay it N_BENCH times.
    // Graph replay eliminates per-launch CPU overhead (~1–5 µs/kernel).
    cudaGraph_t     graph;
    cudaGraphExec_t graph_exec;

    CUDA_CHECK(cudaStreamBeginCapture(s_compute, cudaStreamCaptureModeGlobal));
    tdgl_step(grid, block, s_compute);
    CUDA_CHECK(cudaStreamEndCapture(s_compute, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
    cudaGraphDestroy(graph);

    for (int i = 0; i < N_WARMUP; ++i)
        CUDA_CHECK(cudaGraphLaunch(graph_exec, s_compute));
    CUDA_CHECK(cudaStreamSynchronize(s_compute));

    auto g0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_BENCH; ++i)
        CUDA_CHECK(cudaGraphLaunch(graph_exec, s_compute));
    CUDA_CHECK(cudaStreamSynchronize(s_compute));
    auto g1  = std::chrono::high_resolution_clock::now();
    double ms_graph = std::chrono::duration<double, std::milli>(g1 - g0).count();

    cudaGraphExecDestroy(graph_exec);

    printf("\n=== Kernel throughput benchmark  (%d steps, %d warmup) ===\n",
           N_BENCH, N_WARMUP);
    printf("  Without CUDA Graphs : %8.2f ms  (%7.3f µs/step)\n",
           ms_base,  ms_base  * 1000.0 / N_BENCH);
    printf("  With    CUDA Graphs : %8.2f ms  (%7.3f µs/step)\n",
           ms_graph, ms_graph * 1000.0 / N_BENCH);
    printf("  Speedup             : %.3fx\n",   ms_base / ms_graph);
    printf("  Grid                : %d pts  (%.3f Mpts)\n", NPTS, NPTS / 1e6);
    printf("============================================================\n\n");
}

// ─────────────────────────────────────────────────────────────────────────────
// fire_snapshot: launch observables + gather kernels, then start async D→H.
// Returns immediately — transfer runs on s_io while s_compute keeps working.
// ─────────────────────────────────────────────────────────────────────────────
static void fire_snapshot(dim3 grid, dim3 block,
                           cudaStream_t s_compute, cudaStream_t s_io,
                           HOSTTDGL& hgl) {
    k_calObservables<<<grid, block, 0, s_compute>>>(hm_Ba);
    k_gatherOutput  <<<grid, block, 0, s_compute>>>(
        hm_st_Psi, hm_st_Ux, hm_st_Bz, hm_st_Jsx, hm_st_Jsy,
        hm_st_VOR, hm_st_ENG);
    CUDA_CHECK(cudaStreamSynchronize(s_compute));   // staging buffers ready

    // Async copy ALL fields so the final converged state is always available
    CUDA_CHECK(cudaMemcpyAsync(hgl.Psi, hm_st_Psi, NPTS*sizeof(cuDoubleComplex),
                               cudaMemcpyDeviceToHost, s_io));
    CUDA_CHECK(cudaMemcpyAsync(hgl.Jsx, hm_st_Jsx, NPTS*sizeof(double),
                               cudaMemcpyDeviceToHost, s_io));
    CUDA_CHECK(cudaMemcpyAsync(hgl.Jsy, hm_st_Jsy, NPTS*sizeof(double),
                               cudaMemcpyDeviceToHost, s_io));
    CUDA_CHECK(cudaMemcpyAsync(hgl.Bz,  hm_st_Bz,  NPTS*sizeof(double),
                               cudaMemcpyDeviceToHost, s_io));
    CUDA_CHECK(cudaMemcpyAsync(hgl.VOR, hm_st_VOR, NPTS*sizeof(double),
                               cudaMemcpyDeviceToHost, s_io));
    CUDA_CHECK(cudaMemcpyAsync(hgl.ENG, hm_st_ENG, NPTS*sizeof(double),
                               cudaMemcpyDeviceToHost, s_io));
    // DO NOT sync s_io here — let D→H overlap with next compute batch
}

// ─────────────────────────────────────────────────────────────────────────────
// flush_snapshot: wait for the in-flight D→H, reduce scalars, check convergence.
// ─────────────────────────────────────────────────────────────────────────────
static void flush_snapshot(cudaStream_t s_io, HOSTTDGL& hgl,
                            double ba, int fid,
                            double& prevMag, int& stabCnt, bool& converged) {
    CUDA_CHECK(cudaStreamSynchronize(s_io));
    hgl.CalTotal(ba);
    printf("Ba=%.3f  fid=%4d  Mag=%+.6f  Nvor=%.2f  Eng=%.6f\n",
           ba, fid, hgl.magnetization, hgl.NumOfVortices, hgl.SysEng);
    if (fabs(hgl.magnetization - prevMag) < 1e-4 && ++stabCnt > 5)
        converged = true;
    prevMag = hgl.magnetization;
}

// ─────────────────────────────────────────────────────────────────────────────
// run_simulation
// ─────────────────────────────────────────────────────────────────────────────
void run_simulation(HOSTTDGL& hgl,
                    cudaStream_t s_compute, cudaStream_t s_io,
                    dim3 grid, dim3 block,
                    const SimConfig& cfg) {
    // Initialise Mag.dat with header
    {
        FILE *fp = fopen("Mag.dat", "w");
        fprintf(fp, "Ba\tMag\tNumVor\tSysEng\n");
        fclose(fp);
    }

    int file_id = 0;

    for (double nowBa = cfg.sBa; nowBa < cfg.eBa; nowBa += cfg.stepBa) {
        CUDA_CHECK(cudaMemcpyAsync(hm_Ba, &nowBa, sizeof(double),
                                   cudaMemcpyHostToDevice, s_compute));
        hgl.nameBa = nowBa;

        // ── Equilibration phase ────────────────────────────────────────────
        {
            double prevMag    = 1e30;
            int    stabCnt    = 0;
            bool   converged  = false;
            bool   pending    = false;
            int    pending_id = 0;

            for (int step = 0; step < cfg.PREPARE_STEPS && !converged; ++step) {
                tdgl_step(grid, block, s_compute);

                if (step % cfg.OUTPUT_INTERVAL == 0) {
                    if (pending)
                        flush_snapshot(s_io, hgl, nowBa, pending_id,
                                       prevMag, stabCnt, converged);
                    if (converged) break;

                    ++file_id;
                    pending_id = file_id;
                    pending    = true;
                    fire_snapshot(grid, block, s_compute, s_io, hgl);
                }
            }
            if (pending)
                flush_snapshot(s_io, hgl, nowBa, pending_id,
                               prevMag, stabCnt, converged);

            // Write field files once for the final converged state
            hgl.OutputPsi(file_id);
            hgl.OutputBz (file_id);
            hgl.OutputJsx(file_id);
            hgl.OutputJsy(file_id);
        }

        // ── Measurement phase: average observables over MEASURE_STEPS ─────
        {
            double sumMag = 0.0, sumNv = 0.0, sumEng = 0.0;
            int    cnt    = 0;
            bool   pending = false;

            for (int step = 0; step < cfg.MEASURE_STEPS; ++step) {
                tdgl_step(grid, block, s_compute);

                if (step % cfg.SAMPLE == 0) {
                    if (pending) {
                        CUDA_CHECK(cudaStreamSynchronize(s_io));
                        hgl.CalTotal(nowBa);
                        sumMag += hgl.magnetization;
                        sumNv  += hgl.NumOfVortices;
                        sumEng += hgl.SysEng;
                        ++cnt;
                    }
                    fire_snapshot(grid, block, s_compute, s_io, hgl);
                    pending = true;
                }
            }
            if (pending) {
                CUDA_CHECK(cudaStreamSynchronize(s_io));
                hgl.CalTotal(nowBa);
                sumMag += hgl.magnetization;
                sumNv  += hgl.NumOfVortices;
                sumEng += hgl.SysEng;
                ++cnt;
            }

            FILE *fp = fopen("Mag.dat", "a");
            fprintf(fp, "%.4f\t%.6f\t%.4f\t%.6f\n",
                    nowBa, sumMag/cnt, sumNv/cnt, sumEng/cnt);
            fclose(fp);
        }
    }
}
