"""
plot_mh.py  —  Plot the M-H (magnetisation vs applied field) curve
               and vortex-count curve from Mag.dat.

Usage:
    python plot_mh.py [--data results/Mag.dat] [--out mh_curve.png]
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ── argument parsing ────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Plot TDGL M-H curve')
parser.add_argument('--data', default='Mag.dat',              help='Path to Mag.dat')
parser.add_argument('--out',  default='figures/mh_curve.png', help='Output image path')
args = parser.parse_args()

# ── load data ───────────────────────────────────────────────────────────────
d = np.loadtxt(args.data, skiprows=1)
Ba, Mag, Nvor, Eng = d[:, 0], d[:, 1], d[:, 2], d[:, 3]

# ── locate transition fields ─────────────────────────────────────────────────
# Hc1: where |M| first starts deviating from linear (first vortex entry)
# Use the field where vortex count first exceeds 3
hc1_idx = np.argmax(Nvor > 3)
Hc1 = Ba[hc1_idx] if hc1_idx > 0 else Ba[1]

# Hc2: where magnetisation first crosses back to near 0
hc2_idx = np.argmax((Ba > 1.0) & (np.abs(Mag) < 0.05))
Hc2 = Ba[hc2_idx] if hc2_idx > 0 else Ba[-1]

# ── plot ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
fig.suptitle(
    'M–H Curve: Type-II Superconductor\n'
    r'(TDGL, $\kappa$=2, 256×128 grid, $\Delta x$=0.1$\lambda$)',
    fontsize=13, fontweight='bold'
)

# ── panel 1: magnetisation ──
ax = axes[0]
ax.axhline(0, color='gray', lw=0.8, ls='--')
ax.axvline(Hc1, color='royalblue', lw=1.3, ls=':', label=f'$H_{{c1}}$ ≈ {Hc1:.2f}')
ax.axvline(Hc2, color='firebrick', lw=1.3, ls=':', label=f'$H_{{c2}}$ ≈ {Hc2:.2f}')
ax.plot(Ba, Mag, 'o-', ms=4, lw=1.6, color='steelblue')
ax.fill_between(Ba, Mag, 0, alpha=0.12, color='steelblue')
ax.set_ylabel(r'$\langle B_z \rangle - B_a$  (magnetisation)', fontsize=11)
ax.legend(fontsize=10, loc='lower left')

# Phase annotations
ax.text(Hc1 / 2,        max(Mag) + 0.02, 'Meissner',       ha='center', color='navy',      fontsize=9)
ax.text((Hc1 + Hc2)/2,  min(Mag) * 0.6,  'Mixed state\n(Abrikosov)',   ha='center', color='steelblue', fontsize=9)
ax.text(min(Hc2 + 0.2, Ba[-1]), max(Mag) + 0.02, 'Normal', ha='center', color='firebrick', fontsize=9)

# ── panel 2: vortex count ──
ax = axes[1]
ax.axvline(Hc1, color='royalblue', lw=1.3, ls=':')
ax.axvline(Hc2, color='firebrick', lw=1.3, ls=':')
ax.plot(Ba, Nvor, 's-', ms=4, lw=1.5, color='darkorange')
ax.set_ylabel('Vortex count  $N_{vor}$', fontsize=11)
ax.set_ylim(bottom=0)

# ── panel 3: system energy ──
ax = axes[2]
ax.axvline(Hc1, color='royalblue', lw=1.3, ls=':')
ax.axvline(Hc2, color='firebrick', lw=1.3, ls=':')
ax.plot(Ba, Eng, 'd-', ms=4, lw=1.5, color='seagreen')
ax.set_ylabel('System energy  $E_{sys}$', fontsize=11)
ax.set_xlabel(r'Applied field  $B_a$  (units of $\lambda^{-2}\Phi_0/2\pi$)', fontsize=11)

plt.tight_layout()
out = Path(args.out)
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved: {out.resolve()}')
