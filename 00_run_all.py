"""
=============================================================================
SCRIPT 00: MASTER ORCHESTRATION SCRIPT
=============================================================================
Run the complete comparative analysis pipeline in sequence:

  01_data_harmonization.py   — load & harmonise 2015 + 2024 data
  02_clustering_by_year.py   — GMM clustering for each year
  03_comparative_analysis.py — temporal comparison
  04_subgroup_analysis.py    — residence / wealth / age subgroups
  05_predictors_membership.py — multinomial logistic + RF predictors
  06_visualizations.py       — publication-quality figures

Usage:
  cd comparative_analysis
  python 00_run_all.py

Requirements:
  pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
              scipy pyreadstat
=============================================================================
"""

import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).parent
SCRIPTS = [
    "01_data_harmonization.py",
    "02_clustering_by_year.py",
    "03_comparative_analysis.py",
    "04_subgroup_analysis.py",
    "05_predictors_membership.py",
    "06_visualizations.py",
]

print("=" * 70)
print("COMPARATIVE MATERNAL HEALTH CARE UTILISATION ANALYSIS")
print("MALAWI DHS 2015 vs 2024 — FULL PIPELINE")
print("=" * 70)
print()

start_total = time.time()
errors = []

for script in SCRIPTS:
    script_path = BASE / script
    if not script_path.exists():
        print(f"  ⚠ SKIP: {script} not found")
        errors.append(script)
        continue

    print(f"\n{'='*60}")
    print(f"  RUNNING: {script}")
    print(f"{'='*60}")
    start = time.time()

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(BASE),
        capture_output=False,  # stream stdout
        text=True
    )

    elapsed = time.time() - start
    if result.returncode == 0:
        print(f"\n  ✓ Completed in {elapsed:.1f}s")
    else:
        print(f"\n  ✗ FAILED after {elapsed:.1f}s (return code {result.returncode})")
        errors.append(script)

total_elapsed = time.time() - start_total
print("\n" + "=" * 70)
print(f"PIPELINE COMPLETE in {total_elapsed:.0f}s")
if errors:
    print(f"  ⚠ {len(errors)} script(s) had errors: {errors}")
else:
    print("  All scripts completed successfully.")
print("\nOutputs:")
print("  comparative_analysis/results/   — CSV tables")
print("  comparative_analysis/figures/   — all figures")
print("  comparative_analysis/manuscript/ — manuscript draft")
print("=" * 70)
