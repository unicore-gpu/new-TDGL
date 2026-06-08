cmake_minimum_required(VERSION 3.20)
project(cuTDGL LANGUAGES CXX CUDA)

# ── C++/CUDA standards ────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD  17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ── GPU architecture ──────────────────────────────────────────────────────────
# RTX 5090 (Blackwell consumer, GB202) = sm_120
# Adjust for your card: sm_89 (Ada), sm_86 (Ampere), sm_75 (Turing)
set(CMAKE_CUDA_ARCHITECTURES "120")

# ── Sources ───────────────────────────────────────────────────────────────────
add_executable(tdgl
    src/kernels.cu
    src/memory.cu
    src/simulation.cu
    src/main.cu
)

target_include_directories(tdgl PRIVATE include)

# ── CUDA compile options ──────────────────────────────────────────────────────
target_compile_options(tdgl PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        --use_fast_math
        --extended-lambda
        --extra-device-vectorization
        -lineinfo
        -Xcompiler=-O3,-march=native
    >
)

# Separate compilation: allows __device__ symbols in one TU to be referenced
# from another (e.g. memory.cu calling cudaMemcpyToSymbol on kernels.cu symbols)
set_target_properties(tdgl PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# ── Install ───────────────────────────────────────────────────────────────────
install(TARGETS tdgl DESTINATION bin)
