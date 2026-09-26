from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from .io import combine_csvs
from .runner import preprocess, classify
from .validation import semantic_checks


def main(argv=None):
    parser = argparse.ArgumentParser(prog="rig-state-fsm")
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("combine", help="Combine split EDR CSV/ZIP exports")
    c.add_argument("inputs", nargs="+", type=Path); c.add_argument("--timestamp", required=True)
    c.add_argument("--output", required=True, type=Path)
    p = sub.add_parser("preprocess", help="Run Stage 2 on a Stage-1 table")
    p.add_argument("input", type=Path); p.add_argument("--output-dir", type=Path, default=Path("stage2_outputs"))
    p.add_argument("--selection-mode", choices=["automatic", "original", "calculated", "ask"], default="automatic")
    p.add_argument("--no-html", action="store_true")
    f = sub.add_parser("classify", help="Run FSM/dashboard on Stage-2-ready data")
    f.add_argument("input", type=Path); f.add_argument("--output-dir", type=Path, default=Path("fsm_outputs"))
    f.add_argument("--well-name", help="Well name shown in dashboard metadata")
    v = sub.add_parser("validate", help="Run semantic checks on a labeled CSV")
    v.add_argument("input", type=Path)
    args = parser.parse_args(argv)
    if args.command == "combine": print(json.dumps(combine_csvs(args.inputs, args.output, args.timestamp), indent=2))
    elif args.command == "preprocess": preprocess(args.input, args.output_dir, args.selection_mode, not args.no_html)
    elif args.command == "classify": classify(args.input, args.output_dir, args.well_name)
    else:
        result = semantic_checks(pd.read_csv(args.input, low_memory=False)); print(json.dumps(result, indent=2))
        raise SystemExit(0 if all(x == 0 for x in result.values()) else 2)


if __name__ == "__main__":
    main()
