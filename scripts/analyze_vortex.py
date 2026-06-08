"""
analyze_vortex.py  —  Vortex analysis: locate vortex cores, plot trajectories
                      across Ba steps, compute vortex lattice statistics.

Output files:
    vortex_positions.csv   — vortex (x, y) for each Ba
    vortex_cores.png       — scatter map of all cores colour-coded by Ba
    vortex_density.png     — vortex density vs Ba compared to Φ/Φ₀ prediction
    lattice_order.png      — bond-angle order parameter g6 vs Ba (hexatic order)

Usage:
    python analyze_vortex.py [--data results] [--out-dir .]
"""

import argparse, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import label, center_of_mass

NX, NY = 255, 127
DX, DY = 0.1, 0.1
KAPPA  = 2.0
# Theoretical vortex spacing at field Ba: a = sqrt(2*Phi0 / (sqrt(3)*Ba))
# In our units Phi0 = 2*pi/kappa^2 ... but we track directly from simulation

# ── file parsing ──────────────────────────────────────────────────────────────

def parse_files(data_dir):
    files = {}
    for fp in Path(data_dir).glob('Psi*.dat'):
        m = re.match(r'Psi(\d+)_(\d+)\.dat', fp.name)
        if not m:
            continue
        idx, step = int(m.group(1)), int(m.group(2))
        Ba_val = idx * 0.01          # filename index = round(100 * Ba)
        files[idx] = dict(Ba=Ba_val, step=step, Psi=fp,
                          Bz=fp.parent / f'Bz{idx}_{step}.dat')
    return dict(sorted(files.items()))


def load_field(path):
    """
    Load a tab-separated 2D field file.
    Format: row 0 = x-axis labels, col 0 = y-axis labels.
    Returns the (NY, NX) array of actual field values.
    """
    d = np.genfromtxt(path, delimiter='\t')
    return d[1:, 1:]


def load_psi_abs2(path):
    """Load Psi file (stores |ψ|) and return |ψ|²."""
    return load_field(path) ** 2


def load_scalar(path):
    """Load a real scalar field (Bz, Jsx, Jsy, etc.)."""
    return load_field(path)


# ── vortex detection ──────────────────────────────────────────────────────────

def find_vortex_cores(psi2, threshold=0.15, min_size=3):
    """
    Locate vortex cores as minima in |ψ|².
    Returns array of (row, col) pixel coordinates.
    """
    binary = psi2 < threshold
    labeled, n_features = label(binary)
    if n_features == 0:
        return np.empty((0, 2))
    cores = []
    for i in range(1, n_features + 1):
        region = labeled == i
        if region.sum() < min_size:
            continue
        cy, cx = center_of_mass(region)
        cores.append([cy, cx])
    return np.array(cores)


# ── hexatic order parameter ────────────────────────────────────────────────────

def hexatic_order(cores, rc=3.0 / DX):
    """
    Compute per-vortex bond-angle order |ψ₆| = |⟨exp(6iθ)⟩|.
    Returns mean |ψ₆| over all vortices.
    """
    if len(cores) < 3:
        return 0.0
    psi6_list = []
    for i, ci in enumerate(cores):
        diffs = cores - ci
        dists = np.hypot(diffs[:, 0], diffs[:, 1])
        neighbors = diffs[(dists > 0) & (dists < rc)]
        if len(neighbors) < 2:
            continue
        angles = np.arctan2(neighbors[:, 0], neighbors[:, 1])
        psi6 = np.abs(np.mean(np.exp(6j * angles)))
        psi6_list.append(psi6)
    return float(np.mean(psi6_list)) if psi6_list else 0.0


# ── main analysis ─────────────────────────────────────────────────────────────

def run_analysis(data_dir='results', out_dir='.'):
    out_dir = Path(out_dir)
    files   = parse_files(data_dir)

    records  = []    # (Ba, n_vortex, mean_x, mean_y, g6)
    all_cores = []   # list of (Ba, x, y) for scatter plot

    print(f"{'Ba':>6}  {'Nvort':>6}  {'g6':>6}")
    print('-' * 24)
    for idx, info in files.items():
        psi2  = load_psi_abs2(info['Psi'])
        cores = find_vortex_cores(psi2)
        n     = len(cores)
        g6    = hexatic_order(cores) if n > 2 else 0.0
        records.append((info['Ba'], n, g6))
        for r, c in cores:
            all_cores.append((info['Ba'], c * DX, r * DY))
        print(f"{info['Ba']:>6.2f}  {n:>6d}  {g6:>6.3f}")

    records   = np.array(records)
    all_cores = np.array(all_cores) if all_cores else np.empty((0, 3))
    Ba_arr, Nv_arr, g6_arr = records[:, 0], records[:, 1], records[:, 2]

    # ── save CSV ──────────────────────────────────────────────────────────────
    csv_path = out_dir / 'vortex_positions.csv'
    np.savetxt(csv_path, all_cores, header='Ba  x_lambda  y_lambda',
               fmt='%.4f', delimiter=',')
    print(f'\nVortex positions saved → {csv_path}')

    # ── plot 1: vortex core scatter ───────────────────────────────────────────
    if len(all_cores):
        fig, ax = plt.subplots(figsize=(10, 5))
        sc = ax.scatter(all_cores[:, 1], all_cores[:, 2],
                        c=all_cores[:, 0], cmap='plasma',
                        s=12, alpha=0.6, linewidths=0)
        plt.colorbar(sc, ax=ax, label='$B_a$')
        ax.set_xlim(0, (NX + 1) * DX)
        ax.set_ylim(0, (NY + 1) * DY)
        ax.set_aspect('equal')
        ax.set_xlabel('x (λ)', fontsize=11)
        ax.set_ylabel('y (λ)', fontsize=11)
        ax.set_title('Vortex core positions across all $B_a$ steps\n'
                     '(colour = $B_a$)', fontsize=12, fontweight='bold')
        p = out_dir / 'vortex_cores.png'
        plt.tight_layout()
        plt.savefig(p, dpi=150, bbox_inches='tight')
        print(f'Saved: {p}')

    # ── plot 2: vortex density vs Ba ──────────────────────────────────────────
    sample_area = (NX + 1) * DX * (NY + 1) * DY   # in λ²
    nv_density  = Nv_arr / sample_area              # vortices per λ²

    fig, ax = plt.subplots(figsize=(8, 4))
    # Theoretical: n_v = Ba / (π/κ²) in these units
    Ba_th = np.linspace(0, Ba_arr.max(), 200)
    Phi0  = np.pi / KAPPA**2          # flux quantum in simulation units
    nv_th = Ba_th / Phi0 / sample_area * sample_area   # per unit area → scaled
    ax.plot(Ba_arr, nv_density, 'o-', ms=5, lw=1.5, label='Simulation', color='steelblue')
    ax.plot(Ba_th, Ba_th / Phi0, '--', lw=1.5, label=r'Theory: $B_a/\Phi_0$', color='gray')
    ax.set_xlabel('$B_a$', fontsize=11)
    ax.set_ylabel('Vortex density  (λ⁻²)', fontsize=11)
    ax.set_title('Vortex density vs applied field', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(bottom=0)
    p = out_dir / 'vortex_density.png'
    plt.tight_layout()
    plt.savefig(p, dpi=150, bbox_inches='tight')
    print(f'Saved: {p}')

    # ── plot 3: hexatic order parameter ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    mask = Nv_arr > 2
    ax.plot(Ba_arr[mask], g6_arr[mask], '^-', ms=6, lw=1.5, color='seagreen')
    ax.axhline(0.7, color='orange', lw=1, ls='--', label='$g_6$=0.7 (hexatic threshold)')
    ax.set_xlabel('$B_a$', fontsize=11)
    ax.set_ylabel('Hexatic order  $|\\psi_6|$', fontsize=11)
    ax.set_title('Bond-angle (hexatic) order of vortex lattice', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.legend(fontsize=10)
    p = out_dir / 'lattice_order.png'
    plt.tight_layout()
    plt.savefig(p, dpi=150, bbox_inches='tight')
    print(f'Saved: {p}')


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='TDGL vortex analysis')
    ap.add_argument('--data',    default='.',         help='Directory with .dat files')
    ap.add_argument('--out-dir', default='figures',   help='Output directory for plots/CSV')
    args = ap.parse_args()
    run_analysis(data_dir=args.data, out_dir=args.out_dir)
