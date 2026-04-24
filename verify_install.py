#!/usr/bin/env python3
"""
verify_install.py
=================
Quick installation check for the replication package.

Verifies that:
  1. Required Python packages are installed
  2. Both test suites pass
  3. The paper compiles

Does NOT require any external data — uses synthetic fixtures only.

Usage:
    python verify_install.py
"""

import os
import subprocess
import sys


def green(s): return f"\033[92m{s}\033[0m"
def red(s):   return f"\033[91m{s}\033[0m"
def bold(s):  return f"\033[1m{s}\033[0m"


def check(name, cmd, cwd=None):
    """Run a command and report pass/fail."""
    print(f"{bold(name):.<60s}", end=' ', flush=True)
    try:
        result = subprocess.run(cmd, cwd=cwd, shell=True, capture_output=True,
                                  text=True, timeout=300)
        if result.returncode == 0:
            print(green("PASS"))
            return True
        else:
            print(red("FAIL"))
            print("  stdout:", result.stdout[-500:])
            print("  stderr:", result.stderr[-500:])
            return False
    except subprocess.TimeoutExpired:
        print(red("TIMEOUT"))
        return False


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    print(bold("\nReplication package verification") + "\n" + "=" * 60)

    # Stage 1: Python packages
    print("\n" + bold("[1/4] Python packages"))
    needed = ['pandas', 'numpy', 'statsmodels', 'sklearn', 'pytest']
    for pkg in needed:
        try:
            __import__(pkg)
            print(f"  {pkg:.<40s} {green('installed')}")
        except ImportError:
            print(f"  {pkg:.<40s} {red('MISSING')}")
            print(f"    pip install {pkg}")

    # Optional: torch (only needed for DEQN tests)
    try:
        import torch
        print(f"  {'torch (optional)':.<40s} {green('installed')}")
        torch_avail = True
    except ImportError:
        print(f"  {'torch (optional)':.<40s} {red('MISSING')}")
        print("    DEQN tests will be skipped. pip install torch to enable.")
        torch_avail = False

    # Stage 2: Reduced-form tests
    print("\n" + bold("[2/4] Reduced-form test suite (70 tests, ~5s)"))
    rf_dir = os.path.join(here, 'reduced_form')
    if os.path.isdir(rf_dir):
        check("Running reduced_form/tests/", "python -m pytest tests/ -q",
              cwd=rf_dir)
    else:
        print(red("  reduced_form/ not found"))

    # Stage 3: DEQN tests
    print("\n" + bold("[3/4] DEQN test suite (148 tests, ~5s)"))
    deqn_dir = os.path.join(here, 'deqn_solver')
    if not torch_avail:
        print(red("  Skipped: torch not installed"))
    elif os.path.isdir(deqn_dir):
        # Set up sampling.py from stub if not present
        sampling_path = os.path.join(deqn_dir, 'src', 'sampling.py')
        stub_path = os.path.join(deqn_dir, 'src', 'sampling_stub.py')
        if not os.path.exists(sampling_path) and os.path.exists(stub_path):
            import shutil
            shutil.copy(stub_path, sampling_path)
            print(f"  (copied sampling_stub.py → sampling.py for tests)")
        check("Running deqn_solver/tests/", "python -m pytest tests/ -q",
              cwd=deqn_dir)
    else:
        print(red("  deqn_solver/ not found"))

    # Stage 4: Paper compiles
    print("\n" + bold("[4/4] Paper compilation"))
    paper_dir = os.path.join(here, 'paper')
    paper_pdf = os.path.join(paper_dir, 'paper.pdf')
    paper_tex = os.path.join(paper_dir, 'paper.tex')

    if not os.path.isfile(paper_tex):
        print(red("  paper/paper.tex not found"))
    elif os.path.isfile(paper_pdf):
        print(f"  paper/paper.pdf already exists "
              f"({os.path.getsize(paper_pdf)//1024} KB)")
        print(f"  {green('Skipped recompile')}")
    else:
        check("Compiling paper.tex (pdflatex)",
              "pdflatex -interaction=nonstopmode paper.tex > /dev/null && "
              "pdflatex -interaction=nonstopmode paper.tex > /dev/null",
              cwd=paper_dir)

    print("\n" + bold("Done.") + " See README.md and REPRODUCING.md for "
          "next steps.\n")


if __name__ == "__main__":
    main()
