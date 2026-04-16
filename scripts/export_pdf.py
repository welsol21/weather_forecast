#!/usr/bin/env python3
"""Export technical_report.html to A4 PDF using Chrome headless.

Usage:
    python scripts/export_pdf.py
Output:
    docs/technical_report.pdf
"""

import subprocess, sys, time, shutil
from pathlib import Path

HTML = Path("docs/technical_report.html").resolve()
PDF  = Path("docs/technical_report.pdf").resolve()

chrome = shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chromium-browser")
if not chrome:
    sys.exit("Chrome / Chromium not found")

print(f"Chrome:  {chrome}")
print(f"Input:   {HTML}")
print(f"Output:  {PDF}")

cmd = [
    chrome,
    "--headless=new",
    "--disable-gpu",
    "--no-sandbox",
    "--disable-dev-shm-usage",
    "--run-all-compositor-stages-before-draw",
    "--virtual-time-budget=6000",       # 6 s of virtual time for JS / Chart.js
    f"--print-to-pdf={PDF}",
    "--print-to-pdf-no-header",         # no Chrome URL / date header
    "--no-pdf-header-footer",
    str(HTML),
]

result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

if result.returncode != 0:
    print("STDERR:", result.stderr[:2000])
    sys.exit(f"Chrome exited with code {result.returncode}")

size_kb = PDF.stat().st_size // 1024
print(f"Done → {PDF}  ({size_kb} KB)")
