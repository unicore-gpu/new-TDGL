#!/usr/bin/env python3
"""
check_data.py  —  Validate TDGL results directory.
Usage: python3 check_data.py [results_dir]
"""

import sys
import re
from pathlib import Path
from collections import defaultdict

results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')

if not results_dir.exists():
    print(f'ERROR: directory not found: {results_dir}')
    sys.exit(1)

mag_file = results_dir / 'Mag.dat'
if not mag_file.exists():
    print('ERROR: Mag.dat not found — simulation may not have completed.')
    sys.exit(1)

# Scan for field files per Ba index
prefixes = ['Psi', 'Bz', 'Jsx', 'Jsy']
files_per_idx = defaultdict(dict)

for fp in results_dir.glob('*.dat'):
    if fp.name == 'Mag.dat':
        continue
    m = re.match(r'(Psi|Bz|Jsx|Jsy)(\d+)_(\d+)\.dat', fp.name)
    if not m:
        continue
    prefix, idx, step = m.group(1), int(m.group(2)), int(m.group(3))
    files_per_idx[idx][prefix] = step

if not files_per_idx:
    print('ERROR: No field .dat files found in', results_dir)
    sys.exit(1)

# Report
ok = 0
missing = []
print(f'{"Ba":>6}  {"idx":>5}  {"Psi":>5}  {"Bz":>5}  {"Jsx":>5}  {"Jsy":>5}  status')
print('-' * 52)
for idx in sorted(files_per_idx):
    d = files_per_idx[idx]
    Ba = idx * 0.01
    cols = [d.get(p, '---') for p in prefixes]
    status = 'OK' if all(isinstance(c, int) for c in cols) else 'MISSING'
    if status == 'OK':
        ok += 1
    else:
        missing.append(Ba)
    step_str = [str(c) if isinstance(c, int) else '---' for c in cols]
    print(f'{Ba:>6.2f}  {idx:>5d}  {step_str[0]:>5}  {step_str[1]:>5}  '
          f'{step_str[2]:>5}  {step_str[3]:>5}  {status}')

print()
print(f'Complete steps: {ok} / {len(files_per_idx)}')
if missing:
    print(f'Missing fields at Ba: {[f"{b:.2f}" for b in missing]}')
else:
    print('All steps complete.')
