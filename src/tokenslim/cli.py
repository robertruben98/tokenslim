"""Minimal CLI: ``python -m tokenslim perf`` / ``evals``.

Offline only — runs the bundled eval workload and prints a savings/faithfulness
report. No network, no API key required.
"""

from __future__ import annotations

import argparse
import sys

from .config import Config
from .evals import perf_report, run_suite

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tokenslim", description="tokenslim observability CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    perf = sub.add_parser("perf", help="run the bundled workload and print savings")
    perf.add_argument("--model", default=None, help="model name for cost estimation")

    evals = sub.add_parser("evals", help="run the eval suite (ratio + faithfulness)")
    evals.add_argument("--model", default=None, help="model name for token counting")

    args = parser.parse_args(argv)

    if args.command == "perf":
        print(perf_report(model=args.model))
        return 0

    if args.command == "evals":
        results = run_suite(config=Config(min_bytes=0, model=args.model))
        all_faithful = all(r.faithful for r in results)
        for r in results:
            status = "PASS" if r.faithful else "FAIL"
            print(f"[{status}] {r.name}: ratio={r.ratio:.1%} drops={r.n_markers}")
            if r.missing:
                print(f"        missing must-keep: {r.missing}")
        return 0 if all_faithful else 1

    return 2  # unreachable: subparser is required


if __name__ == "__main__":
    sys.exit(main())
