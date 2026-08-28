#!/usr/bin/env python3
"""Rebuild legacy Warwick-QU/GlaS Grade_modified.csv from Grade.csv."""
from __future__ import annotations
import argparse, csv
from pathlib import Path
GLAS = {"benign": 0, "malignant": 1}
SIRINUKUN = {"adenomatous": 0, "healthy": 1, "poorly differentiated": 2, "moderately differentiated": 3, "moderately-to-poorly differentated": 4, "moderately-to-poorly differentiated": 4}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="Dataset/glas/Grade.csv"); parser.add_argument("--output", default="Dataset/glas/Grade_modified.csv")
    args = parser.parse_args(); rows = []
    with open(args.input, newline="") as handle:
        for row in csv.DictReader(handle):
            clean = {k.strip(): v.strip() for k, v in row.items()}; name = clean["name"]; split = "test" if name.startswith("test") else "train"
            rows.append({"name": f"Dataset/glas/{split}/image/{name}.bmp", "patient ID": clean["patient ID"], "grade (GlaS)": GLAS[clean["grade (GlaS)"].lower()], "grade (Sirinukunwattana et al. 2015)": SIRINUKUN[clean["grade (Sirinukunwattana et al. 2015)"].lower()]})
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", "patient ID", "grade (GlaS)", "grade (Sirinukunwattana et al. 2015)"]); writer.writeheader(); writer.writerows(rows)
if __name__ == "__main__": main()
