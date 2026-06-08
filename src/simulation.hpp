// src/simulation.hpp  ─  Simulation loop and benchmark API

#pragma once
#include <cuda_runtime.h>
#include "../include/host_tdgl.hpp"

// ── Simulation configuration ─────────────────────────────────────────────────
struct SimConfig {
    double sBa;            // start applied field
    double eBa;            // end applied field  (exclusive)
    double stepBa;         // field increment

    int SAMPLE;            // base time-step count  (scale factor)
    int PREPARE_STEPS;     // max equilibration steps  = 96700 * SAMPLE
    int OUTPUT_INTERVAL;   // convergence-check interval = 300  * SAMPLE
    int MEASURE_STEPS;     // averaging steps             = 50   * SAMPLE

    // Convenience constructor: derives all step counts from SAMPLE
    static SimConfig make(double sBa_, double eBa_, double stepBa_, int sample = 100) {
        SimConfig c;
        c.sBa            = sBa_;
        c.eBa            = eBa_;
        c.stepBa         = stepBa_;
        c.SAMPLE         = sample;
        c.PREPARE_STEPS  = 96700 * sample;
        c.OUTPUT_INTERVAL = 300  * sample;
        c.MEASURE_STEPS  =   50  * sample;
        return c;
    }
};

// ── API ───────────────────────────────────────────────────────────────────────

// run_benchmark: pure kernel throughput test (no I/O, no convergence checks).
// Runs N_BENCH steps without and with CUDA Graphs, prints results.
void run_benchmark(dim3 grid, dim3 block, cudaStream_t s_compute);

// run_simulation: sweep Ba from cfg.sBa to cfg.eBa, equilibrate at each step,
// write field snapshots on convergence, and append row to Mag.dat.
void run_simulation(HOSTTDGL& hgl,
                    cudaStream_t s_compute,
                    cudaStream_t s_io,
                    dim3 grid, dim3 block,
                    const SimConfig& cfg);
