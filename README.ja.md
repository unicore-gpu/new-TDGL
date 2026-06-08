# GPU TDGL ソルバー

[English](README.md) | [简体中文](README.zh-CN.md) | **日本語**

時間依存 Ginzburg-Landau(Time-Dependent Ginzburg-Landau, TDGL)方程式を解く
CUDA ソルバーで、NVIDIA RTX 5090(Blackwell アーキテクチャ, sm_120)向けに
近代化したものです。

アルゴリズム:W. D. Gropp et al., *J. Comput. Phys.* **123**, 254 (1996)。  
オリジナルコード:CUDA Toolkit 3.2, 2011 年。本リライトは CUDA 12 / sm_120 を対象とします。

---

## 背景 —— 本プロジェクトが解決する問題

### 物理的背景

**第二種超伝導体**を磁場中に置くと、磁場を完全に排除する(マイスナー状態)わけでも、
そのまま常伝導状態になるわけでもありません。下部臨界磁場 **Hc1** を超えると、磁場は
量子化された**渦糸(vortex)**の格子として試料内に侵入します。各渦糸は 1 個の磁束量子を
担う常伝導状態の細い磁束管であり、その周囲を超伝導電流が循環しています。磁場が高く
なるにつれて渦糸が増え、三角形の **Abrikosov 格子**へと自己組織化し、上部臨界磁場
**Hc2** に達すると渦糸の芯が重なり合って超伝導性が破壊されます。渦糸が端で核形成し、
**Bean–Livingston 表面障壁**を越え、運動し、欠陥にピン留めされ、格子へと配列する過程は、
あらゆる実用超伝導体(磁石、ケーブル、RF 空洞)の磁気特性と輸送特性を支配します。

これらのダイナミクスには解析解がありません。標準的な研究手法は、**時間依存
Ginzburg-Landau(TDGL)方程式**——複素超伝導秩序パラメータ ψ(x,y,t) と磁気ベクトル
ポテンシャル A(x,y,t) に関する連立偏微分方程式——を格子上で時間方向に積分し、各印加
磁場で系が平衡状態に緩和するまで計算することです。本ソルバーは Gropp et al.(1996)による
**ゲージ不変なリンク変数(link-variable)離散化スキーム**を採用しており、格子上で
ゲージ不変性を厳密に保ちます(完全な導出は `docs/TDGL_paper.md` を参照)。

### 本リポジトリが行うこと

GPU 上で離散化 TDGL 方程式を時間ステップ計算することにより、外部磁場を掃引した際の
二次元第二種超伝導体の**磁化曲線(M–H ループ)**と完全な空間構造を計算します。
具体的には、各印加磁場 Ba について次を行います:

1. ψ と A を平衡状態へ緩和(陽的前進オイラー法による時間ステップ)、
2. 磁化、渦糸数、自由エネルギーを記録、
3. 秩序パラメータ |ψ|、誘導磁場 Bz、超伝導電流 Js の分布を出力。

これらから、Hc1 での渦糸侵入、混合状態での Abrikosov 格子、Hc2 での常伝導状態への
転移を観察でき、定量化することもできます(M–H 曲線、渦糸位置、ヘキサティック秩序)。
下記の **結果** を参照してください。

### なぜ GPU リライトなのか

TDGL の更新はメモリバウンドな最近接ステンシル計算であり、大きな格子に対して
数百万の時間ステップを反復します——GPU に理想的に適合します。オリジナルの 2011 年の
コードは、とうに陳腐化した CUDA Toolkit 3.2 と Fermi 世代のハードウェアを対象として
いました。本リポジトリは**それを CUDA 12 と NVIDIA Blackwell(RTX 5090, sm_120)へ
近代化**し、10 項目の CUDA 最適化(コアレスドなメモリ配置、共有メモリによるステンシル
タイリング、カーネル融合、デュアルストリーム、CUDA Graphs、Blackwell の L2 永続化……)を
適用して、256×128 格子で **約 44 µs/ステップ** を達成しています。
`docs/OPTIMIZATION.md` を参照してください。

---

## 結果

256×128 格子(DX=0.1λ, κ=2)でゼロ磁場冷却(ZFC)の M-H 掃引を行い、
Ba を 0.01 → 2.5 まで、ステップ 0.05 で変化させます。

![M-H curve](figures/mh_curve_zfc.png)

渦糸の侵入は Hc1 ≈ 0.81(Bean-Livingston 障壁)で始まり、
混合状態で Abrikosov 格子が形成され、Hc2 ≈ 1.81 で常伝導状態に達します。

---

## プロジェクト構成

```
TDGL/
├── include/
│   ├── params.hpp        ← すべての調整可能パラメータ(NX、DX、KAPPA、DT …)
│   ├── host_tdgl.hpp     ← ホスト側データコンテナとファイル I/O
│   ├── cuda_check.hpp    ← CUDA_CHECK マクロ
│   └── complex_ops.cuh   ← インライン倍精度複素数演算
├── src/
│   ├── main.cu           ← エントリポイント;掃引範囲は SimConfig::make() で設定
│   ├── kernels.cu/cuh    ← すべての CUDA カーネル
│   ├── memory.cu/hpp     ← デバイスメモリ割り当てと定数アップロード
│   └── simulation.cu/hpp ← ベンチマークとシミュレーションループ
├── scripts/
│   ├── plot_mh.py        ← M-H 曲線のプロット
│   ├── plot_fields.py    ← 空間場分布図
│   └── analyze_vortex.py ← 渦糸検出と格子秩序の解析
├── figures/              ← 出力 PNG 図と CSV
├── docs/
│   ├── TDGL_paper.md     ← 物理的背景と方程式
│   └── OPTIMIZATION.md   ← CUDA 最適化の詳細
├── agents/
│   ├── tdgl-run/         ← agent:ビルドと実行の方法
│   └── tdgl-analysis/    ← agent:解析とプロットの方法
├── CMakeLists.txt
├── Makefile
└── README.md
```

---

## ビルド

```bash
make -j$(nproc)    # CUDA 12+ と nvcc が必要
make profile       # Nsight Systems プロファイル → tdgl_profile.nsys-rep
make clean
```

自動的に適用されるコンパイラフラグ:
- `-arch=sm_120 --generate-code arch=compute_120,code=sm_120`
- `-O3 --use_fast_math --extended-lambda --extra-device-vectorization`

他の GPU の場合は Makefile の `ARCH` を変更してください:

| GPU | ARCH |
|-----|------|
| RTX 5090(Blackwell GB202) | `sm_120` |
| RTX 4090(Ada Lovelace) | `sm_89` |
| RTX 3090 / A100(Ampere) | `sm_86` / `sm_80` |
| RTX 2080(Turing) | `sm_75` |

---

## 設定

### 物理とグリッド —— `include/params.hpp`

| パラメータ | 記号 | デフォルト | 意味 |
|-----------|--------|---------|---------|
| `NX`, `NY` | — | 255, 127 | グリッドノード数(試料 = (NX+1)·DX × (NY+1)·DY) |
| `DX`, `DY` | Δx, Δy | 0.1 λ | グリッド間隔、単位はロンドン侵入長 λ |
| `KAPPA` | κ = λ/ξ | 2.0 | GL パラメータ(κ > 1/√2 → 第二種) |
| `DT` | Δt | 2×10⁻³ | 時間ステップ —— **DT ≤ DX²/2 を満たす必要あり** |
| `SIGMA` | σ | 1.0 | 常伝導状態の電気伝導度 |

デフォルト値での物理的試料サイズ:**25.6λ × 12.8λ**  
渦糸芯の半径:ξ = λ/κ = 0.5λ = DX=0.1 のとき 5 グリッド点

**DT 安定性ガイド:**

| DX | 最大 DT |
|----|--------|
| 0.05 | 1.25×10⁻³ |
| 0.1 | 5×10⁻³ |
| 0.2 | 2×10⁻² |

### 掃引範囲 —— `src/main.cu`

```cpp
const SimConfig cfg = SimConfig::make(
    /*sBa*/    0.01,   // 開始磁場
    /*eBa*/    2.50,   // 終了磁場
    /*stepBa*/ 0.05,   // 磁場増分
    /*SAMPLE*/ 100     // 収束チェック間のステップ数
);
```

収束は、|ΔMag| < 10⁻⁴ が 5 回連続で成立したときに判定されます。

---

## 実行

```bash
./tdgl
```

すべての出力ファイルは**カレントディレクトリ**に書き込まれます:

| ファイル | 内容 |
|------|------|
| `Mag.dat` | ステップごとの要約:`Ba  Mag  NumVor  SysEng` |
| `Psi{idx}_{step}.dat` | 収束時の秩序パラメータ \|ψ(x,y)\| |
| `Bz{idx}_{step}.dat` | 収束時の誘導磁場 Bz(x,y) |
| `Jsx/Jsy{idx}_{step}.dat` | 収束時の超伝導電流密度 |

ファイルインデックス:`idx = round(100 × Ba)`、`Ba = idx × 0.01` で復元。

---

## 結果の解析

```bash
# M-H 曲線
python3 scripts/plot_mh.py --data Mag.dat --out figures/mh_curve.png

# 指定した Ba 値の空間場分布図
python3 scripts/plot_fields.py --data . --ba 0.81 1.01 1.51 --field both --out figures/field_maps.png

# すべての Ba ステップの完全なギャラリー
python3 scripts/plot_fields.py --data . --all --field psi --out figures/psi_gallery.png

# 渦糸の位置、密度、ヘキサティック秩序
python3 scripts/analyze_vortex.py --data . --out-dir figures
```

完全な自動解析ワークフローは `agents/tdgl-analysis/AGENT.md` を参照してください。  
AI agent 向けのビルド/実行手順は `agents/tdgl-run/AGENT.md` を参照してください。  
物理的背景と方程式:`docs/TDGL_paper.md`。

---

## CUDA 最適化

| # | 手法 | 効果 |
|---|-----------|--------|
| 1 | 行優先のコアレスドメモリ配置 | ★★★ |
| 2 | 物理スカラーに `__constant__` メモリを使用 | ★★ |
| 3 | 共有メモリによるステンシルタイル(バンクコンフリクトなし、幅 34) | ★★★ |
| 4 | カーネル融合(dΨ/dt + dU/dt を 1 パスで) | ★★★ |
| 5 | ピン留めホストメモリ(`cudaHostAlloc`) | ★★ |
| 6 | デュアル CUDA ストリーム(計算 ∥ デバイス→ホスト転送) | ★★ |
| 7 | 3 つのメモリプール → 3 回の `cudaMalloc` 呼び出し | ★ |
| 8 | L2 キャッシュ永続化ウィンドウ(Blackwell sm_120+) | ★★ |
| 9 | 微分カーネル向けの最大共有メモリ割り当て | ★★ |
| 10 | CUDA Graphs(ベンチマークループで計測) | +5–25% |

詳細は `docs/OPTIMIZATION.md` を参照してください。

---

## ベンチマーク(RTX 5090, sm_120)

```
グリッド:256×128（32 768 点）
CUDA Graphs なし：約 47.6 µs/ステップ
CUDA Graphs あり：約 44.5 µs/ステップ   （+7% 高速化）
```

---

## 物理的背景

ゲージ不変 TDGL 方程式の完全な導出と、本ソルバーで使用しているリンク変数離散化
スキームについては、`docs/TDGL_paper.md` を参照してください。
