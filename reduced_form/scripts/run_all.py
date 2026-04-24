"""
run_all.py
==========
Run all four reduced-form table scripts in sequence.

This is a convenience wrapper; each script can be run individually.

Usage:
    python scripts/run_all.py
"""

import os
import sys
import subprocess
import time

SCRIPTS = [
    'table6_horse_race.py',
    'table7_merton_incremental.py',
    'table8_out_of_sample.py',
    'table9_supervised_lm.py',
]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)

    print("=" * 70)
    print(" REDUCED-FORM ANALYSIS: running all 4 tables")
    print("=" * 70)
    print()

    total_start = time.time()
    results = []

    for script_name in SCRIPTS:
        script_path = os.path.join(here, script_name)
        print(f"▶ {script_name}")
        t0 = time.time()
        rc = subprocess.call(
            [sys.executable, script_path],
            cwd=root,
        )
        elapsed = time.time() - t0
        status = "OK" if rc == 0 else f"FAILED (exit {rc})"
        print(f"   {status} ({elapsed:.1f}s)")
        print()
        results.append((script_name, rc, elapsed))

    # Summary
    total = time.time() - total_start
    print("=" * 70)
    print(f" Total time: {total:.1f}s")
    print("=" * 70)
    fails = [r for r in results if r[1] != 0]
    if fails:
        print(f" {len(fails)} script(s) failed:")
        for name, rc, _ in fails:
            print(f"   {name}: exit {rc}")
        sys.exit(1)
    else:
        print(" All scripts succeeded.")
        print(f" Output tables in: {os.path.join(root, 'output', 'tables')}")


if __name__ == "__main__":
    main()
