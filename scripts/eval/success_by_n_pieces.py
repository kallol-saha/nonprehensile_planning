"""Cross-tabulate success rate by piece count, pooled over one or more runs.

Reads ``results.csv`` from each ``--output_dir`` (e.g. the in-distribution and
out-of-distribution benchmarks), concatenates them, and prints a table of
success rate per (method, n_pieces).

Usage:
    python scripts/eval/success_by_n_pieces.py \\
        --output_dirs eval_output/unet_benchmark eval_output/unet_benchmark_ood
"""

import argparse
import csv
import os
from collections import defaultdict


METHOD_PRETTY = {
    "diff_cbs": "UNet + CBS",
    "rrt_cbs":  "RRT-Connect + CBS",
    "diff":     "UNet only",
    "rrt":      "RRT-Connect only",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dirs", nargs="+", required=True,
                   help="One or more directories each containing results.csv")
    return p.parse_args()


def _to_bool(x):
    return str(x).strip().lower() == "true"


def load_rows(output_dirs):
    rows = []
    for d in output_dirs:
        path = os.path.join(d, "results.csv")
        if not os.path.isfile(path):
            print(f"WARNING: {path} not found, skipping")
            continue
        with open(path) as f:
            for r in csv.DictReader(f):
                r["_source"] = os.path.basename(os.path.normpath(d))
                rows.append(r)
    return rows


def main():
    args = parse_args()
    rows = load_rows(args.output_dirs)
    if not rows:
        raise SystemExit("No rows loaded.")

    # (method, n_pieces) -> [successes, total]
    counts = defaultdict(lambda: [0, 0])
    methods = []
    piece_counts = set()
    for r in rows:
        m = r["method"]
        n = int(r["n_pieces"])
        counts[(m, n)][1] += 1
        if _to_bool(r["success"]):
            counts[(m, n)][0] += 1
        if m not in methods:
            methods.append(m)
        piece_counts.add(n)

    piece_counts = sorted(piece_counts)

    # Header
    header = f"{'n_pieces':<10} " + " ".join(
        f"{METHOD_PRETTY.get(m, m):<22}" for m in methods
    )
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    for n in piece_counts:
        row = f"{n:<10d} "
        cells = []
        for m in methods:
            succ, total = counts[(m, n)]
            if total == 0:
                cells.append(f"{'—':<22}")
            else:
                pct = 100.0 * succ / total
                cells.append(f"{pct:5.1f}%  ({succ:>3d}/{total:<3d})  ".ljust(22))
        row += " ".join(cells)
        print(row)

    # Overall totals
    print("-" * len(header))
    row = f"{'ALL':<10} "
    cells = []
    for m in methods:
        succ = sum(counts[(m, n)][0] for n in piece_counts)
        total = sum(counts[(m, n)][1] for n in piece_counts)
        pct = 100.0 * succ / total if total else 0.0
        cells.append(f"{pct:5.1f}%  ({succ:>3d}/{total:<3d})  ".ljust(22))
    row += " ".join(cells)
    print(row)
    print("=" * len(header))
    print("Cell format: success%  (# success / # episodes)")
    print("Sources pooled:", ", ".join(sorted({r["_source"] for r in rows})))


if __name__ == "__main__":
    main()
