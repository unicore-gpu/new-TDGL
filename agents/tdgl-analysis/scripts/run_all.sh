#!/usr/bin/env bash
# run_all.sh  —  Run the complete TDGL analysis pipeline in one command.
# Usage: bash agents/tdgl-analysis/scripts/run_all.sh [results_dir] [out_dir]
#
# Defaults: results_dir=results/   out_dir=. (project root)

set -e

RESULTS="${1:-results}"
OUTDIR="${2:-.}"

echo "=== TDGL Analysis Pipeline ==="
echo "Data dir : $RESULTS"
echo "Output   : $OUTDIR"
echo ""

# Step 1: validate
echo "[1/5] Checking data..."
python3 agents/tdgl-analysis/scripts/check_data.py "$RESULTS"
echo ""

# Step 2: M-H curve
echo "[2/5] Plotting M-H curve..."
python3 plot_mh.py --data "$RESULTS/Mag.dat" --out "$OUTDIR/mh_curve.png"

# Step 3: field maps (selected)
echo "[3/5] Plotting field maps..."
python3 plot_fields.py --data "$RESULTS" --ba 0.46 0.81 1.01 1.51 2.01 \
    --field both --out "$OUTDIR/field_maps.png"
python3 plot_fields.py --data "$RESULTS" --ba 0.46 0.81 1.01 \
    --current --out "$OUTDIR/current_maps.png"

# Step 4: galleries
echo "[4/5] Generating galleries..."
python3 plot_fields.py --data "$RESULTS" --all --field psi --out "$OUTDIR/psi_gallery.png"
python3 plot_fields.py --data "$RESULTS" --all --field bz  --out "$OUTDIR/bz_gallery.png"

# Step 5: vortex analysis
echo "[5/5] Running vortex analysis..."
python3 analyze_vortex.py --data "$RESULTS" --out-dir "$OUTDIR"

echo ""
echo "=== Done. Output files in $OUTDIR: ==="
ls -lh "$OUTDIR"/*.png "$OUTDIR"/vortex_positions.csv 2>/dev/null || true
