"""Compile reports/report.tex into reports/report.pdf.

Usage: python scripts/build_report.py   (run scripts/make_figures.py first)
Needs a LaTeX installation with latexmk (TeX Live).
The text is written by hand; its numbers come from reports/tables/ and models/detector.json.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports"


def main() -> None:
    if shutil.which("latexmk") is None:
        raise SystemExit("latexmk not found; install TeX Live or compile reports/report.tex on Overleaf")
    subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "report.tex"],
                   cwd=REP, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["latexmk", "-c", "report.tex"], cwd=REP, check=True, stdout=subprocess.DEVNULL)
    print("wrote", REP / "report.pdf")


if __name__ == "__main__":
    main()
