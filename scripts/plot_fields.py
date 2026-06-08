"""
plot_fields.py  —  Plot spatial field maps (|ψ|², Bz, current density J)
                   for selected or all Ba values.

File naming convention (from cuTDGL.h output):
    Psi{Ba_idx}_{step}.dat   — complex order parameter (NX+1)×(NY+1)
    Bz{Ba_idx}_{step}.dat    — local field
    Jsx{Ba_idx}_{step}.dat   — supercurrent x-component
    Jsy{Ba_idx}_{step}.dat   — supercurrent y-component

Usage examples:
    # Plot |ψ|² and Bz for Ba ≈ 0.81, 1.01, 2.01 (flux-jump region + dense + normal)
    python plot_fields.py --ba 0.81 1.01 2.01

    # Plot all Ba values as a grid of |ψ|² panels
    python plot_fields.py --all --field psi

    # Full set: Bz maps for every step
    python plot_fields.py --all --field bz --out bz_gallery.png
"""

import argparse, glob, os, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from pathlib import Path

NX, NY = 255, 127          # grid nodes (256×128 points)
DX, DY = 0.1, 0.1         # spacing in units of λ

# ── helpers ──────────────────────────────────────────────────────────────────

def load_field(path):
    """
    Load a tab-separated 2D field file.
    Format: row 0 = x-axis labels, col 0 = y-axis labels.
    Returns the (NY, NX) array of actual field values.
    """
    d = np.genfromtxt(path, delimiter='\t')
    return d[1:, 1:]   # skip header row and y-label column


def load_scalar(path):
    """Load a real scalar field (Bz, Jsx, Jsy, etc.)."""
    return load_field(path)


def load_psi_abs2(path):
    """Load Psi file (stores |ψ|) and return |ψ|²."""
    psi = load_field(path)
    return psi ** 2


def parse_files(data_dir='results'):
    """Return a dict: Ba_idx -> {'Ba': float, 'step': int, 'Psi': path, 'Bz': path, ...}"""
    files = {}
    for fp in Path(data_dir).glob('Psi*.dat'):
        m = re.match(r'Psi(\d+)_(\d+)\.dat', fp.name)
        if not m:
            continue
        idx, step = int(m.group(1)), int(m.group(2))
        Ba_val = idx * 0.01          # filename index = round(100 * Ba)
        files[idx] = dict(
            Ba=Ba_val, step=step,
            Psi=fp,
            Bz=fp.parent / f'Bz{idx}_{step}.dat',
            Jsx=fp.parent / f'Jsx{idx}_{step}.dat',
            Jsy=fp.parent / f'Jsy{idx}_{step}.dat',
        )
    return dict(sorted(files.items()))


def closest_idx(files, ba_target):
    """Return the key whose Ba is closest to ba_target."""
    return min(files, key=lambda k: abs(files[k]['Ba'] - ba_target))


# ── main plotting ─────────────────────────────────────────────────────────────

def plot_panel(ax, data, title, cmap, vmin=None, vmax=None, xlabel=True, ylabel=True):
    nx_pts, ny_pts = data.shape[1], data.shape[0]
    extent = [0, nx_pts * DX, 0, ny_pts * DY]
    im = ax.imshow(data, origin='lower', extent=extent, cmap=cmap,
                   vmin=vmin, vmax=vmax, aspect='equal', interpolation='nearest')
    ax.set_title(title, fontsize=9, pad=3)
    if xlabel:
        ax.set_xlabel('x (λ)', fontsize=8)
    if ylabel:
        ax.set_ylabel('y (λ)', fontsize=8)
    ax.tick_params(labelsize=7)
    return im


def plot_selected(ba_list, field='both', data_dir='results', out='field_maps.png'):
    files = parse_files(data_dir)
    if not files:
        raise FileNotFoundError(f'No Psi*.dat found in {data_dir}')

    selected = [closest_idx(files, b) for b in ba_list]
    n = len(selected)
    rows = 2 if field == 'both' else 1
    fig, axes = plt.subplots(rows, n, figsize=(4 * n, 4 * rows + 0.8),
                             squeeze=False)
    fig.suptitle(
        r'TDGL field maps  ($\kappa$=2, $\Delta x$=0.1$\lambda$)',
        fontsize=12, fontweight='bold'
    )

    for col, idx in enumerate(selected):
        info = files[idx]
        Ba_str = f"$B_a$={info['Ba']:.2f}"

        if field in ('both', 'psi'):
            psi2 = load_psi_abs2(info['Psi'])
            im = plot_panel(axes[0, col], psi2, f'|ψ|²  {Ba_str}',
                            cmap='inferno', vmin=0, vmax=1,
                            xlabel=(rows == 1), ylabel=(col == 0))
            if col == n - 1:
                plt.colorbar(im, ax=axes[0, col], fraction=0.046, pad=0.04,
                             label='|ψ|²')

        if field in ('both', 'bz'):
            row = 1 if field == 'both' else 0
            bz = load_scalar(info['Bz'])
            im = plot_panel(axes[row, col], bz, f'$B_z$  {Ba_str}',
                            cmap='RdBu_r',
                            vmin=-abs(bz).max(), vmax=abs(bz).max(),
                            ylabel=(col == 0))
            if col == n - 1:
                plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04,
                             label='$B_z$')

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved: {Path(out).resolve()}')


def plot_current_overlay(ba_list, data_dir='results', out='current_maps.png',
                         quiver_step=8):
    """Plot |ψ|² with supercurrent streamlines overlaid."""
    files = parse_files(data_dir)
    selected = [closest_idx(files, b) for b in ba_list]
    n = len(selected)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), squeeze=False)
    fig.suptitle(
        r'|ψ|² + supercurrent streamlines  ($\kappa$=2)',
        fontsize=12, fontweight='bold'
    )

    x_arr = np.arange(NX + 1) * DX
    y_arr = np.arange(NY + 1) * DY

    for col, idx in enumerate(selected):
        info = files[idx]
        psi2 = load_psi_abs2(info['Psi'])
        Jx   = load_scalar(info['Jsx'])
        Jy   = load_scalar(info['Jsy'])

        ax = axes[0, col]
        im = ax.imshow(psi2, origin='lower',
                       extent=[0, (NX+1)*DX, 0, (NY+1)*DY],
                       cmap='inferno', vmin=0, vmax=1, aspect='equal')
        # Downsample for quiver
        xs = x_arr[::quiver_step]
        ys = y_arr[::quiver_step]
        Jxs = Jx[::quiver_step, ::quiver_step]
        Jys = Jy[::quiver_step, ::quiver_step]
        ax.quiver(xs, ys, Jxs, Jys, color='white', alpha=0.6,
                  scale_units='xy', scale=None, width=0.003)
        ax.set_title(f"|ψ|² + J  $B_a$={info['Ba']:.2f}", fontsize=9)
        ax.set_xlabel('x (λ)', fontsize=8)
        if col == 0:
            ax.set_ylabel('y (λ)', fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='|ψ|²')

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved: {Path(out).resolve()}')


def plot_all_gallery(field='psi', data_dir='results', out=None, ncols=5):
    """Grid of all Ba steps for one field type."""
    files = parse_files(data_dir)
    keys  = sorted(files.keys())
    n     = len(keys)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2.5 * nrows))
    fig.suptitle(
        f'{"$|\\psi|^2$" if field=="psi" else "$B_z$"} gallery  — all Ba steps',
        fontsize=12, fontweight='bold'
    )

    for i, idx in enumerate(keys):
        r, c = divmod(i, ncols)
        ax = axes[r, c] if nrows > 1 else axes[c]
        info = files[idx]
        if field == 'psi':
            data = load_psi_abs2(info['Psi'])
            cmap, vmin, vmax = 'inferno', 0, 1
        else:
            data = load_scalar(info['Bz'])
            m = np.abs(data).max()
            cmap, vmin, vmax = 'RdBu_r', -m, m
        ax.imshow(data, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax,
                  aspect='auto', interpolation='nearest')
        ax.set_title(f"Ba={info['Ba']:.2f}", fontsize=7, pad=2)
        ax.axis('off')

    # hide spare axes
    for j in range(i + 1, nrows * ncols):
        r, c = divmod(j, ncols)
        ax = axes[r, c] if nrows > 1 else axes[c]
        ax.axis('off')

    plt.tight_layout()
    if out is None:
        out = f'{field}_gallery.png'
    plt.savefig(out, dpi=120, bbox_inches='tight')
    print(f'Saved: {Path(out).resolve()}')


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot TDGL field maps')
    parser.add_argument('--data',  default='.', help='Directory with .dat files')
    parser.add_argument('--ba',    nargs='+', type=float,
                        default=[0.46, 0.81, 1.01, 1.51, 2.01],
                        help='Ba values to plot')
    parser.add_argument('--field', choices=['psi', 'bz', 'both'], default='both',
                        help='Which field to show')
    parser.add_argument('--current', action='store_true',
                        help='Plot |ψ|² + current overlay instead')
    parser.add_argument('--all',   action='store_true',
                        help='Gallery of all Ba steps')
    parser.add_argument('--out',   default=None, help='Output image filename')
    args = parser.parse_args()

    if args.all:
        plot_all_gallery(field=args.field if args.field != 'both' else 'psi',
                         data_dir=args.data,
                         out=args.out or f'{args.field}_gallery.png')
    elif args.current:
        plot_current_overlay(args.ba, data_dir=args.data,
                             out=args.out or 'current_maps.png')
    else:
        plot_selected(args.ba, field=args.field, data_dir=args.data,
                      out=args.out or 'field_maps.png')
