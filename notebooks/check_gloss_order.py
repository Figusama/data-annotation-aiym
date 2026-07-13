"""
Checks whether gloss ORDER is preserved:
  1) in the raw annotation CSV(s) -- the column before any cleaning
  2) in surdobot_dataset_records.json's gloss_bag -- after your Part 2
     cleaning/dedup step

If gloss_bag entries come back alphabetically sorted while the raw CSV
values are NOT alphabetically sorted, that's strong evidence the
cleaning step discarded order (e.g. via set() or sorted()) even though
the raw source had it -- meaning order needs to be re-extracted from
the raw CSVs, not recovered from the existing records.json.
"""
import argparse
import glob
import json
from pathlib import Path

import pandas as pd


def find_gloss_column(df):
    candidates = [c for c in df.columns if "gloss" in c.lower()]
    return candidates[0] if candidates else None


def find_clip_column(df):
    for name in ("Video", "video", "clip_id", "filename", "video_path"):
        if name in df.columns:
            return name
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-csv-glob", required=True,
                     help="Glob pattern for raw annotation CSVs, e.g. './annotations/*.csv'")
    ap.add_argument("--records", required=True, type=Path,
                     help="Path to surdobot_dataset_records.json")
    ap.add_argument("--sample", type=int, default=15)
    args = ap.parse_args()

    # --- 1. Raw CSV inspection ---
    csv_files = sorted(glob.glob(args.raw_csv_glob))
    print(f"Found {len(csv_files)} raw annotation CSV(s) matching {args.raw_csv_glob!r}")
    if not csv_files:
        print("No files found -- check the glob pattern.")
        return

    df = pd.read_csv(csv_files[0])
    gloss_col = find_gloss_column(df)
    clip_col = find_clip_column(df)
    print(f"Using file: {csv_files[0]}")
    print(f"Detected gloss column: {gloss_col!r}, clip column: {clip_col!r}")

    if gloss_col is None:
        print("Could not find a gloss column -- inspect df.columns manually:", list(df.columns))
        return

    print(f"\nRaw gloss column samples (first {args.sample} non-null rows):")
    raw_samples = df[gloss_col].dropna().astype(str).head(args.sample).tolist()
    for i, val in enumerate(raw_samples):
        print(f"  [{i}] {val!r}")

    # Heuristic: for each sample, split on common delimiters and check
    # whether the split order matches alphabetical order. If MOST rows
    # are NOT alphabetically sorted, that's evidence of genuine temporal
    # order in the source data.
    import re
    unsorted_count = 0
    checked = 0
    for val in raw_samples:
        tokens = [t.strip().upper() for t in re.split(r"[,;|\t\n]+", val) if t.strip()]
        if len(tokens) < 2:
            continue
        checked += 1
        if tokens != sorted(tokens):
            unsorted_count += 1
    if checked:
        print(f"\n{unsorted_count}/{checked} multi-token raw samples are NOT alphabetically ordered "
              f"(non-alphabetical order is evidence of real temporal/sequence order in the source).")
    else:
        print("\nNot enough multi-token samples in this file to check ordering heuristically.")

    # --- 2. Processed records.json inspection ---
    with open(args.records) as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("records", list(data.values()))

    print(f"\ngloss_bag samples from {args.records} (first {args.sample} multi-gloss records):")
    shown = 0
    unsorted_in_records = 0
    checked_records = 0
    for rec in data:
        bag = rec.get("gloss_bag")
        if not bag or len(bag) < 2:
            continue
        if shown < args.sample:
            print(f"  clip_id={rec.get('clip_id')!r} gloss_bag={bag}")
            shown += 1
        checked_records += 1
        if list(bag) != sorted(bag):
            unsorted_in_records += 1
        if checked_records >= 500:  # enough for a stable estimate
            break

    if checked_records:
        print(f"\n{unsorted_in_records}/{checked_records} multi-gloss gloss_bags in records.json "
              f"are NOT alphabetically sorted.")
        if unsorted_in_records == 0:
            print("*** All checked gloss_bags are alphabetically sorted -- order was almost certainly "
                  "discarded during cleaning, even if the raw CSV preserves it. ***")
        else:
            print("Some non-alphabetical order survives in records.json -- worth double-checking "
                  "against the raw CSV row-by-row to confirm it's genuine temporal order and not "
                  "an artifact of insertion order from set/dict operations.")


if __name__ == "__main__":
    main()
