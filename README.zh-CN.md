# GPU TDGL 求解器

[English](README.md) | **简体中文** | [日本語](README.ja.md)

求解含时 Ginzburg-Landau(Time-Dependent Ginzburg-Landau,TDGL)方程的 CUDA 程序,
针对 NVIDIA RTX 5090(Blackwell 架构,sm_120)进行了现代化重写。

算法:W. D. Gropp 等,*J. Comput. Phys.* **123**, 254 (1996)。  
原始代码:CUDA Toolkit 3.2,2011 年。本次重写面向 CUDA 12 / sm_120。

---

## 背景 —— 本项目解决什么问题

### 物理背景

将**第二类超导体**置于磁场中时,它既不会简单地把磁场完全排出(迈斯纳态),
也不会直接转入正常态。当外场超过下临界场 **Hc1** 后,磁场以量子化**涡旋(vortex)**
点阵的形式穿入样品——每个涡旋是一根细小的正常态磁通管,携带一个磁通量子,
周围环绕着超导电流。随着外场升高,更多涡旋进入,自组织成三角形的
**Abrikosov 点阵**;直到外场达到上临界场 **Hc2**,涡旋核心相互重叠,
超导性被破坏。涡旋如何在边缘成核、越过 **Bean–Livingston 表面势垒**、运动、
被缺陷钉扎、并排列成点阵,决定了一切实用超导体(磁体、电缆、射频腔)的
磁学与输运性质。

这些动力学过程没有解析解。研究它们的标准方法是把**含时 Ginzburg-Landau(TDGL)
方程**——关于复超导序参量 ψ(x,y,t) 和磁矢势 A(x,y,t) 的耦合偏微分方程——
在网格上沿时间向前积分,直到每个外场下系统弛豫到平衡态。本求解器采用
Gropp 等(1996)提出的**规范不变的连接变量(link-variable)离散方案**,
它在格点上精确保持规范不变性(完整推导见 `docs/TDGL_paper.md`)。

### 本仓库做什么

它通过在 GPU 上对离散 TDGL 方程做时间步进,计算二维第二类超导体在外场扫描
过程中的**磁化曲线(M–H 回线)**及完整的空间结构。具体来说,对每个外加场 Ba:

1. 把 ψ 和 A 弛豫到平衡态(显式前向欧拉时间步进),
2. 记录磁化强度、涡旋数目和自由能,
3. 输出序参量 |ψ|、感应磁场 Bz 和超导电流 Js 的空间分布。

由此可以看到 Hc1 处的涡旋进入、混合态中的 Abrikosov 点阵,以及 Hc2 处向正常态
的转变,并对它们做定量分析(M–H 曲线、涡旋位置、六角序)。参见下文的 **结果**。

### 为什么要做 GPU 重写

TDGL 的更新是一个访存受限的最近邻 stencil 计算,要在大网格上迭代数百万个时间步——
天然适合 GPU。原始的 2011 年代码面向早已淘汰的 CUDA Toolkit 3.2 和 Fermi 时代的
硬件。本仓库**将其现代化到 CUDA 12 与 NVIDIA Blackwell(RTX 5090,sm_120)**,
应用了十项 CUDA 优化(合并访存布局、共享内存 stencil 分块、核函数融合、双流、
CUDA Graphs、Blackwell L2 持久化……),在 256×128 网格上达到 **约 44 µs/步**。
详见 `docs/OPTIMIZATION.md`。

---

## 结果

在 256×128 网格上(DX=0.1λ,κ=2)做零场冷却(ZFC)的 M-H 扫描,
Ba 从 0.01 → 2.5,步长 0.05。

![M-H curve](figures/mh_curve_zfc.png)

涡旋进入始于 Hc1 ≈ 0.81(Bean-Livingston 势垒),
混合态中形成 Abrikosov 点阵,在 Hc2 ≈ 1.81 处到达正常态。

---

## 项目结构

```
TDGL/
├── include/
│   ├── params.hpp        ← 所有可调参数(NX、DX、KAPPA、DT …)
│   ├── host_tdgl.hpp     ← 主机端数据容器与文件 I/O
│   ├── cuda_check.hpp    ← CUDA_CHECK 宏
│   └── complex_ops.cuh   ← 内联双精度复数运算
├── src/
│   ├── main.cu           ← 入口;扫描范围通过 SimConfig::make() 设定
│   ├── kernels.cu/cuh    ← 所有 CUDA 核函数
│   ├── memory.cu/hpp     ← 设备内存分配与常量上传
│   └── simulation.cu/hpp ← 基准测试与模拟循环
├── scripts/
│   ├── plot_mh.py        ← M-H 曲线绘图
│   ├── plot_fields.py    ← 空间场分布图
│   └── analyze_vortex.py ← 涡旋检测与点阵序分析
├── figures/              ← 输出的 PNG 图与 CSV
├── docs/
│   ├── TDGL_paper.md     ← 物理背景与方程
│   └── OPTIMIZATION.md   ← CUDA 优化细节
├── agents/
│   ├── tdgl-run/         ← agent:如何编译与运行
│   └── tdgl-analysis/    ← agent:如何分析与绘图
├── CMakeLists.txt
├── Makefile
└── README.md
```

---

## 编译

```bash
make -j$(nproc)    # 需要 CUDA 12+ 与 nvcc
make profile       # Nsight Systems 性能分析 → tdgl_profile.nsys-rep
make clean
```

自动应用的编译选项:
- `-arch=sm_120 --generate-code arch=compute_120,code=sm_120`
- `-O3 --use_fast_math --extended-lambda --extra-device-vectorization`

对于其他 GPU,修改 Makefile 中的 `ARCH`:

| GPU | ARCH |
|-----|------|
| RTX 5090(Blackwell GB202) | `sm_120` |
| RTX 4090(Ada Lovelace) | `sm_89` |
| RTX 3090 / A100(Ampere) | `sm_86` / `sm_80` |
| RTX 2080(Turing) | `sm_75` |

---

## 配置

### 物理与网格 —— `include/params.hpp`

| 参数 | 符号 | 默认值 | 含义 |
|-----------|--------|---------|---------|
| `NX`, `NY` | — | 255, 127 | 网格节点数(样品 = (NX+1)·DX × (NY+1)·DY) |
| `DX`, `DY` | Δx, Δy | 0.1 λ | 网格间距,单位为伦敦穿透深度 λ |
| `KAPPA` | κ = λ/ξ | 2.0 | GL 参数(κ > 1/√2 → 第二类) |
| `DT` | Δt | 2×10⁻³ | 时间步长 —— **必须满足 DT ≤ DX²/2** |
| `SIGMA` | σ | 1.0 | 正常态电导率 |

默认参数下的物理样品尺寸:**25.6λ × 12.8λ**  
涡旋核心半径:ξ = λ/κ = 0.5λ = DX=0.1 时的 5 个网格点

**DT 稳定性参考:**

| DX | 最大 DT |
|----|--------|
| 0.05 | 1.25×10⁻³ |
| 0.1 | 5×10⁻³ |
| 0.2 | 2×10⁻² |

### 扫描范围 —— `src/main.cu`

```cpp
const SimConfig cfg = SimConfig::make(
    /*sBa*/    0.01,   // 起始场
    /*eBa*/    2.50,   // 终止场
    /*stepBa*/ 0.05,   // 场增量
    /*SAMPLE*/ 100     // 收敛检查之间的步数
);
```

当 |ΔMag| < 10⁻⁴ 连续 5 次检查成立时,判定为收敛。

---

## 运行

```bash
./tdgl
```

所有输出文件写入**当前目录**:

| 文件 | 内容 |
|------|------|
| `Mag.dat` | 逐步汇总:`Ba  Mag  NumVor  SysEng` |
| `Psi{idx}_{step}.dat` | 收敛时的序参量 \|ψ(x,y)\| |
| `Bz{idx}_{step}.dat` | 收敛时的感应磁场 Bz(x,y) |
| `Jsx/Jsy{idx}_{step}.dat` | 收敛时的超导电流密度 |

文件索引:`idx = round(100 × Ba)`,反推为 `Ba = idx × 0.01`。

---

## 结果分析

```bash
# M-H 曲线
python3 scripts/plot_mh.py --data Mag.dat --out figures/mh_curve.png

# 指定 Ba 值的空间场分布图
python3 scripts/plot_fields.py --data . --ba 0.81 1.01 1.51 --field both --out figures/field_maps.png

# 所有 Ba 步的完整图集
python3 scripts/plot_fields.py --data . --all --field psi --out figures/psi_gallery.png

# 涡旋位置、密度与六角序
python3 scripts/analyze_vortex.py --data . --out-dir figures
```

完整的自动化分析流程见 `agents/tdgl-analysis/AGENT.md`。  
面向 AI agent 的编译/运行说明见 `agents/tdgl-run/AGENT.md`。  
物理背景与方程:`docs/TDGL_paper.md`。

---

## CUDA 优化

| # | 技术 | 收益 |
|---|-----------|--------|
| 1 | 行主序合并访存布局 | ★★★ |
| 2 | 物理标量使用 `__constant__` 内存 | ★★ |
| 3 | 共享内存 stencil 分块(无 bank 冲突,34 宽) | ★★★ |
| 4 | 核函数融合(dΨ/dt + dU/dt 单次完成) | ★★★ |
| 5 | 锁页主机内存(`cudaHostAlloc`) | ★★ |
| 6 | 双 CUDA 流(计算 ∥ 设备→主机传输) | ★★ |
| 7 | 3 个内存池 → 3 次 `cudaMalloc` 调用 | ★ |
| 8 | L2 缓存持久化窗口(Blackwell sm_120+) | ★★ |
| 9 | 为导数核函数设置最大共享内存配额 | ★★ |
| 10 | CUDA Graphs(在基准循环中测量) | +5–25% |

完整细节见 `docs/OPTIMIZATION.md`。

---

## 基准测试(RTX 5090,sm_120)

```
网格:256×128（32 768 点）
不使用 CUDA Graphs：约 47.6 µs/步
使用    CUDA Graphs：约 44.5 µs/步   （+7% 加速）
```

---

## 物理背景

规范不变 TDGL 方程的完整推导,以及本求解器所用的连接变量离散方案,
详见 `docs/TDGL_paper.md`。
