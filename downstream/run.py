#!/usr/bin/env python
"""Single entry point for all downstream runs. Usage: python run.py configs/baseline.yaml"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import load_config
from pipeline import run_all

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run.py <config.yaml>")
        sys.exit(1)
    cfg = load_config(sys.argv[1])
    run_all(cfg)
