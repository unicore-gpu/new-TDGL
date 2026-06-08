# Data Format Reference

## Directory layout

The simulation writes all output files to the **directory from which `./tdgl` was launched**
(i.e. the current working directory at runtime — typically the project root).

```
<run_dir>/
├── Mag.dat               # summary table (one row per Ba step)
├── Psi{idx}_{step}.dat   # order parameter |ψ| field
├── Bz{idx}_{step}.dat    # local magnetic field Bz
├── Jsx{idx}_{step}.dat   # supercurrent x-component
└── Jsy{idx}_{step}.dat   # supercurrent y-component
```

## File naming

`{prefix}{idx}_{step}.dat`

| Token | Meaning |
|-------|---------|
| `idx` | `round(100 * Ba)` — integer key for the Ba value |
| `step` | simulation time-step at which this snapshot was saved |

Reconstruct Ba from idx: `Ba = idx * 0.01`

## Field file format

Tab-separated matrix with axis labels:

```
0.0  0.1  0.2  ...  (NX+1)*DX    ← x-coordinates (row 0, skip when loading)
0.0  v00  v01  ...  v0,NX         ← y=0 row  (col 0 = y label, skip)
0.1  v10  v11  ...  v1,NX
...
NY*DY  ...
```

Load in Python:
```python
import numpy as np
d = np.genfromtxt(path, delimiter='\t')
field = d[1:, 1:]   # shape: (NY, NX) = (128, 256) for default grid
```

## Psi file specifics

Stores `|ψ|` (real, range 0–1), **not** complex pairs.
To get `|ψ|²`: `psi2 = field ** 2`

## Mag.dat format

Tab-separated, one header row:

```
Ba    Mag    NumVor    SysEng
0.01  0.011  2.000     -155.3
...
```

| Column | Meaning |
|--------|---------|
| Ba | Applied magnetic field |
| Mag | `⟨Bz⟩ − Ba` (diamagnetic response) |
| NumVor | Topological vortex number (winding number sum) |
| SysEng | Total GL free energy |

## Default grid parameters

All parameters are defined in **`include/params.hpp`**.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| NX | 255 | Grid nodes in x (256 points) |
| NY | 127 | Grid nodes in y (128 points) |
| DX | 0.1 λ | Grid spacing x |
| DY | 0.1 λ | Grid spacing y |
| KAPPA | 2.0 | GL parameter κ |
| DT | 2e-3 | Timestep (units τ = σλ²/c²) |

The sweep range (sBa, eBa, stepBa) and output frequency (SAMPLE) are set in
`src/main.cu` inside `SimConfig::make(...)`.
