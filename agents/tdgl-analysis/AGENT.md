# TDGL Simulation Analysis Agent

## Purpose
You are an analysis agent for Time-Dependent Ginzburg-Landau (TDGL) superconductor simulations.
Given output data from the TDGL CUDA solver, automatically:
1. Validate data completeness
2. Generate all physical plots
3. Run vortex analysis
4. Summarise the physics

---

## Step 0 — Locate the data

The simulation binary (`./tdgl`) writes all output files to the **directory from which it was launched**.
Look for these files relative to the project root (or wherever `./tdgl` was run):

```
Mag.dat               # magnetisation summary (always present if run completed)
Psi{idx}_{step}.dat   # one file per Ba step — order parameter field
Bz{idx}_{step}.dat    # local magnetic field
Jsx{idx}_{step}.dat   # supercurrent x-component
Jsy{idx}_{step}.dat   # supercurrent y-component
```

If `Mag.dat` is missing, the simulation has not finished. Stop and tell the user.

Read the simulation parameters from **`include/params.hpp`**:
- `NX`, `NY` — grid size (update `NX`, `NY` constants in `scripts/plot_fields.py` and `scripts/analyze_vortex.py` if they differ from 255, 127)
- `DX`, `DY` — grid spacing (default 0.1 λ)
- `KAPPA` — GL parameter (default 2.0)

The sweep range and step count are set in `src/main.cu` via `SimConfig::make(sBa, eBa, stepBa, SAMPLE)`.

---

## Step 1 — Validate data

```bash
python3 agents/tdgl-analysis/scripts/check_data.py <data_dir>
```

where `<data_dir>` is the directory containing the `.dat` files (`.` if they are in the project root).

Expected output: one line per Ba step with counts of Psi/Bz/Jsx/Jsy files.
If any step is missing a field file, report it but continue with available data.

---

## Step 2 — M-H curve

```bash
python3 scripts/plot_mh.py --data <data_dir>/Mag.dat --out figures/mh_curve.png
```

Output: `figures/mh_curve.png` with three panels:
- Magnetisation vs Ba
- Vortex count vs Ba
- System energy vs Ba

**What to look for:**
| Feature | Physical meaning |
|---------|-----------------|
| Linear M ∝ −Ba at low field | Meissner state (perfect diamagnet) |
| Sharp kink / first vortex entry | Lower critical field Hc1 |
| |M| peak then decline toward 0 | Mixed (Abrikosov vortex) state |
| M → 0 | Upper critical field Hc2, normal state |

---

## Step 3 — Field maps

Plot five representative Ba values spanning all phases:

```bash
python3 scripts/plot_fields.py \
  --data <data_dir> \
  --ba 0.46 0.81 1.01 1.51 2.01 \
  --field both \
  --out figures/field_maps.png
```

For current-overlay plots:
```bash
python3 scripts/plot_fields.py --data <data_dir> --ba 0.46 0.81 1.01 --current --out figures/current_maps.png
```

For a full gallery (all Ba steps):
```bash
python3 scripts/plot_fields.py --data <data_dir> --all --field psi --out figures/psi_gallery.png
python3 scripts/plot_fields.py --data <data_dir> --all --field bz  --out figures/bz_gallery.png
```

**What to look for:**
- `|ψ|² ≈ 1` everywhere → Meissner (no vortices)
- Dark spots in `|ψ|²` with bright rings in Bz → Abrikosov vortex cores
- Uniform dark `|ψ|² ≈ 0` → Normal state (Ba > Hc2)

---

## Step 4 — Vortex analysis

```bash
python3 scripts/analyze_vortex.py --data <data_dir> --out-dir figures
```

Output files:
| File | Content |
|------|---------|
| `vortex_positions.csv` | (Ba, x, y) for every detected vortex core |
| `vortex_cores.png` | Scatter map of all cores, colour = Ba |
| `vortex_density.png` | Measured density vs theoretical prediction |
| `lattice_order.png` | Hexatic order parameter g₆ vs Ba |

**Interpreting g₆:**
- `g₆ > 0.7` → well-ordered hexagonal Abrikosov lattice
- `g₆ ≈ 0.4–0.7` → partial hexatic order
- `g₆ < 0.3` → disordered / liquid vortex phase

---

## Step 5 — Physical summary

After all plots are generated, write a brief summary including:

1. **Hc1** — Ba at which vortex count first exceeds 5
2. **Hc2** — Ba at which |M| drops below 0.05
3. **Peak vortex count** and the Ba at which it occurs
4. **Maximum g₆** and the Ba range where lattice is ordered (g₆ > 0.7)
5. **Sample physical size** = `(NX+1)*DX × (NY+1)*DY` in units of λ

Template:
```
Physical Summary
───────────────
Sample:        {(NX+1)*DX:.1f}λ × {(NY+1)*DY:.1f}λ
κ (GL param):  {KAPPA}
Hc1 ≈         {Hc1:.2f}
Hc2 ≈         {Hc2:.2f}
Peak vortices: {Nmax} at Ba = {Ba_peak:.2f}
Ordered lattice (g₆ > 0.7): Ba = {Ba_lo:.2f} – {Ba_hi:.2f}
```

---

## Step 6 — Checklist

After completing the workflow, verify all outputs exist:

```
- [ ] figures/mh_curve.png
- [ ] figures/field_maps.png
- [ ] figures/current_maps.png
- [ ] figures/psi_gallery.png
- [ ] figures/bz_gallery.png
- [ ] figures/vortex_cores.png
- [ ] figures/vortex_density.png
- [ ] figures/lattice_order.png
- [ ] figures/vortex_positions.csv
```

Present the checklist to the user and note any missing items.

---

## Reference

- Data format details: [data_format.md](data_format.md)
- Scripts documentation: [scripts/](scripts/)
- Simulation parameters: `include/params.hpp`
- Sweep configuration: `src/main.cu` → `SimConfig::make(...)`
- Paper background: `docs/TDGL_paper.md`
