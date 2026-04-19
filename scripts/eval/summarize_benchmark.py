"""Summarise a compare_methods.py run: print a table + reorganise videos.

Reads ``<output_dir>/results.csv`` (produced by
``scripts/eval/compare_methods.py``) and:

  1. Prints a per-method metrics table (success rate, avg planning time,
     avg conflicts, avg final pose error, avg arc length, avg nodes).
  2. For every (episode, method) row, copies the corresponding
     ``viz/episode_XXXXXX_<method>_solo.mp4`` into
     ``viz_by_method/<method>/<SUCCESS|FAIL>_episode_XXXXXX.mp4``
     so you can eyeball outcomes per method at a glance.

Usage:
    python scripts/eval/summarize_benchmark.py --output_dir eval_output/unet_benchmark
"""

import argparse
import csv
import math
import os
import shutil
from collections import defaultdict


METHOD_PRETTY = {
    "diff_cbs": "UNet + CBS",
    "rrt_cbs":  "RRT-Connect + CBS",
    "diff":     "UNet only",
    "rrt":      "RRT-Connect only",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", required=True,
                   help="Directory containing results.csv (and viz/).")
    p.add_argument("--no_viz", action="store_true",
                   help="Skip the video reorganisation step.")
    p.add_argument("--limit", type=int, default=None,
                   help="Only summarise the first N unique episodes (sorted by "
                        "episode name). All methods for each episode are kept.")
    return p.parse_args()


def _mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else float("nan")


def _to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _to_bool(x):
    return str(x).strip().lower() == "true"


def summarise(rows):
    by_method = defaultdict(list)
    for r in rows:
        by_method[r["method"]].append(r)

    header = (
        f"{'Method':<22} {'N':>4} {'Succ%':>7} {'Conf':>7} "
        f"{'Plan(s)':>9} {'FPE':>7} {'Arc(m)':>8} {'Nodes':>7}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    summary_rows = []
    for method, mrows in by_method.items():
        n = len(mrows)
        succ = sum(1 for r in mrows if _to_bool(r["success"])) / n * 100.0
        conf = _mean([_to_float(r["num_conflicts"]) for r in mrows])
        plan = _mean([_to_float(r["planning_time_s"]) for r in mrows])
        fpe = _mean([_to_float(r["final_pose_error"]) for r in mrows])
        arc = _mean([_to_float(r["arc_length"]) for r in mrows])
        nodes = _mean([_to_float(r["nodes_expanded"]) for r in mrows])

        pretty = METHOD_PRETTY.get(method, method)
        print(f"{pretty:<22} {n:>4d} {succ:>6.1f}% {conf:>7.2f} "
              f"{plan:>9.2f} {fpe:>7.3f} {arc:>8.3f} {nodes:>7.1f}")
        summary_rows.append((method, pretty, n, succ, conf, plan, fpe, arc, nodes))

    print("=" * len(header))
    print("Legend: Succ%=success rate, Conf=avg #conflicts, "
          "Plan(s)=avg planning time, FPE=avg final pose error (m+0.1*rad), "
          "Arc(m)=avg arc length, Nodes=avg CT nodes expanded\n")
    return summary_rows


def reorganise_videos(rows, output_dir):
    src_dir = os.path.join(output_dir, "viz")
    if not os.path.isdir(src_dir):
        print(f"No viz/ dir at {src_dir}; skipping video reorganisation.")
        return

    dst_root = os.path.join(output_dir, "viz_by_method")
    os.makedirs(dst_root, exist_ok=True)

    n_copied = 0
    n_missing = 0
    for r in rows:
        method = r["method"]
        ep = r["episode"]
        success = _to_bool(r["success"])
        tag = "SUCCESS" if success else "FAIL"

        src = os.path.join(src_dir, f"{ep}_{method}_solo.mp4")
        if not os.path.isfile(src):
            n_missing += 1
            continue

        dst_method_dir = os.path.join(dst_root, method)
        os.makedirs(dst_method_dir, exist_ok=True)
        dst = os.path.join(dst_method_dir, f"{tag}_{ep}.mp4")
        shutil.copy2(src, dst)
        n_copied += 1

    print(f"Reorganised {n_copied} videos into {dst_root}/<method>/"
          f"<SUCCESS|FAIL>_<episode>.mp4"
          + (f" ({n_missing} source videos missing)" if n_missing else ""))

    for method in sorted({r["method"] for r in rows}):
        mdir = os.path.join(dst_root, method)
        if os.path.isdir(mdir):
            n_ok = sum(1 for f in os.listdir(mdir) if f.startswith("SUCCESS_"))
            n_bad = sum(1 for f in os.listdir(mdir) if f.startswith("FAIL_"))
            pretty = METHOD_PRETTY.get(method, method)
            print(f"  {pretty:<22} → {mdir}  ({n_ok} success, {n_bad} fail)")


def main():
    args = parse_args()
    csv_path = os.path.join(args.output_dir, "results.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Missing {csv_path}")

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    if args.limit is not None:
        keep = sorted({r["episode"] for r in rows})[:args.limit]
        keep_set = set(keep)
        rows = [r for r in rows if r["episode"] in keep_set]
        print(f"Filtered to first {len(keep)} episodes: {keep[0]} … {keep[-1]}")

    summarise(rows)
    if not args.no_viz:
        reorganise_videos(rows, args.output_dir)


if __name__ == "__main__":
    main()
