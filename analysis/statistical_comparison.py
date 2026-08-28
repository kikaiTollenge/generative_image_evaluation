#!/usr/bin/env python3
"""Mann-Whitney U, Holm correction, and Cliff's delta for repeated CSV results."""
from __future__ import annotations
import argparse, csv
from itertools import combinations
from pathlib import Path
from statistics import median
import numpy as np
from scipy.stats import mannwhitneyu

def cliff_delta(x, y) -> float:
    x = np.asarray(x); y = np.asarray(y)
    greater = sum(float((xi > y).sum()) for xi in x); less = sum(float((xi < y).sum()) for xi in x)
    return (greater - less) / float(len(x) * len(y))

def iqr(values):
    q1, q3 = np.percentile(values, [25, 75]); return float(q1), float(q3)

def holm_adjust(rows):
    ordered = sorted(enumerate(rows), key=lambda item: item[1]["p_value"]); m = len(ordered); adjusted = [None] * m; running = 0.0
    for rank, (orig, row) in enumerate(ordered):
        value = min(1.0, (m - rank) * row["p_value"]); running = max(running, value); adjusted[orig] = running
    for row, adj in zip(rows, adjusted): row["p_holm"] = adj
    return rows

def read_groups(path: Path, metric: str, group_col: str):
    groups = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle): groups.setdefault(row[group_col], []).append(float(row[metric]))
    return groups

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True); parser.add_argument("--metric", required=True); parser.add_argument("--group-col", default="method"); parser.add_argument("--output", required=True)
    args = parser.parse_args(); groups = read_groups(Path(args.input), args.metric, args.group_col); rows = []
    for first, second in combinations(sorted(groups), 2):
        x, y = groups[first], groups[second]; stat = mannwhitneyu(x, y, alternative="two-sided"); q1x, q3x = iqr(x); q1y, q3y = iqr(y)
        rows.append({"metric": args.metric, "first_method": first, "second_method": second, "n_first": len(x), "n_second": len(y), "median_first": median(x), "q1_first": q1x, "q3_first": q3x, "median_second": median(y), "q1_second": q1y, "q3_second": q3y, "delta_median": median(x) - median(y), "mann_whitney_u": float(stat.statistic), "p_value": float(stat.pvalue), "cliffs_delta": cliff_delta(x, y)})
    rows = holm_adjust(rows); output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        fields = list(rows[0].keys()) if rows else ["metric"]; writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
if __name__ == "__main__": main()
