from __future__ import annotations

import runpy
import os
import sys
from pathlib import Path

import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parent
ENGINE_ROOT = PACKAGE_ROOT / "engines"


def preprocess(input_path: Path, output_dir: Path, selection_mode: str = "automatic", html: bool = True):
    engine = ENGINE_ROOT / "stage2.py"
    old = sys.argv[:]
    try:
        sys.argv = [str(engine), "--input", str(input_path), "--output-dir", str(output_dir),
                    "--selection-mode", selection_mode] + ([] if html else ["--no-html"])
        return runpy.run_path(str(engine), run_name="__main__")
    finally:
        sys.argv = old


def classify(input_path: Path, output_dir: Path, well_name: str | None = None):
    """Run the validated Stage-3 engine against a Stage-2-ready file."""
    suffix = input_path.suffix.lower()
    df = pd.read_parquet(input_path) if suffix in {".parquet", ".pq"} else pd.read_csv(input_path, low_memory=False)
    engine = ENGINE_ROOT / "stage3.py"
    output_dir.mkdir(parents=True, exist_ok=True)
    old_output = os.environ.get("FSM_OUTPUT_DIR")
    old_well = os.environ.get("FSM_WELL_NAME")
    try:
        os.environ["FSM_OUTPUT_DIR"] = str(output_dir.resolve())
        if well_name:
            os.environ["FSM_WELL_NAME"] = well_name
        return runpy.run_path(str(engine), init_globals={"processed_df": df})
    finally:
        if old_output is None:
            os.environ.pop("FSM_OUTPUT_DIR", None)
        else:
            os.environ["FSM_OUTPUT_DIR"] = old_output
        if old_well is None:
            os.environ.pop("FSM_WELL_NAME", None)
        else:
            os.environ["FSM_WELL_NAME"] = old_well
