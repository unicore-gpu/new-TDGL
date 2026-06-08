# Makefile — GPU TDGL solver
#
# Uses NVCC separate compilation (-dc) so device symbols defined in one
# translation unit (e.g. __device__ pointers in kernels.cu) are visible to
# other units (e.g. memory.cu calls cudaMemcpyToSymbol on those symbols).
#
# Targets:
#   make           — build ./tdgl
#   make clean     — remove build artefacts
#   make profile   — build + launch Nsight Systems profiler
#   make info      — print build variables

# ── Toolchain ─────────────────────────────────────────────────────────────────
NVCC     := nvcc
CXX      := g++

# ── Architecture ──────────────────────────────────────────────────────────────
# sm_120  = RTX 5090 (Blackwell consumer, GB202)
# For older cards, replace with e.g. sm_89 (Ada), sm_86 (Ampere), sm_75 (Turing)
ARCH     := sm_120

# ── Compiler flags ────────────────────────────────────────────────────────────
CXXSTD   := -std=c++17
NVCCFLAGS :=                                    \
    -arch=$(ARCH)                               \
    --generate-code arch=compute_120,code=sm_120 \
    -O3                                         \
    --use_fast_math                             \
    --extended-lambda                           \
    -Xcompiler "-O3,-march=native"              \
    --extra-device-vectorization                \
    -lineinfo                                   \
    -Iinclude                                   \
    $(CXXSTD)

# ── Sources ───────────────────────────────────────────────────────────────────
SRC_DIR  := src
SRCS     := $(SRC_DIR)/kernels.cu    \
            $(SRC_DIR)/memory.cu     \
            $(SRC_DIR)/simulation.cu \
            $(SRC_DIR)/main.cu

OBJS     := $(SRCS:.cu=.o)
TARGET   := tdgl

# ── Build rules ───────────────────────────────────────────────────────────────
.PHONY: all clean profile info

all: $(TARGET)

# Device-compile each .cu file independently (-dc = relocatable device code)
$(SRC_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -dc -o $@ $<

# Link: nvcc performs device link (-dlink) automatically when objects contain
# relocatable device code
$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^
	@echo "Built: $(TARGET)"

# ── Utility targets ───────────────────────────────────────────────────────────
clean:
	rm -f $(OBJS) $(TARGET)

profile: all
	nsys profile --stats=true -o tdgl_profile ./$(TARGET)

info:
	@echo "NVCC     = $(NVCC)"
	@echo "ARCH     = $(ARCH)"
	@echo "SRCS     = $(SRCS)"
	@echo "OBJS     = $(OBJS)"
	@echo "TARGET   = $(TARGET)"
