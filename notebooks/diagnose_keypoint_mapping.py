"""
Diagnoses the clip_id -> keypoint .npz filename mapping by comparing
actual filenames on disk against several candidate derivation rules,
instead of assuming one.
"""
import argparse
import json
from pathlib import Path
from collections import Counter


def load_records(path):
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("records", list(data.values()))
    return data


CANDIDATES = {
    "parent_folder_name":      lambda cid: Path(cid).parent.name.replace(".mp4", ""),
    "filename_stem":           lambda cid: Path(cid).stem,
    "filename_stem_no_dash":   lambda cid: Path(cid).stem.split("-")[0],
    "full_path_no_ext":        lambda cid: str(Path(cid).with_suffix("")),
    "full_path_slash_to_underscore": lambda cid: str(Path(cid).with_suffix("")).replace("/", "_"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True, type=Path)
    ap.add_argument("--keypoint-dir", required=True, type=Path)
    ap.add_argument("--sample", type=int, default=5)
    args = ap.parse_args()

    records = load_records(args.records)
    clip_ids = [r["clip_id"] for r in records if r.get("clip_id")]

    kp_files = list(args.keypoint_dir.rglob("*.npz")) + list(args.keypoint_dir.rglob("*.npy"))
    kp_stems = {p.stem for p in kp_files}

    print(f"{len(clip_ids)} clip_ids in records")
    print(f"{len(kp_files)} keypoint files found under {args.keypoint_dir}")
    print(f"\nSample keypoint filenames on disk:")
    for p in kp_files[:args.sample]:
        print(f"  {p.relative_to(args.keypoint_dir)}")

    print(f"\nSample clip_ids from records:")
    for cid in clip_ids[:args.sample]:
        print(f"  {cid}")

    print(f"\nMatch rate per candidate derivation rule (against {len(kp_stems)} keypoint file stems):")
    for name, fn in CANDIDATES.items():
        matches = 0
        example_hit, example_miss = None, None
        for cid in clip_ids:
            derived = fn(cid)
            if Path(derived).name in kp_stems or derived in kp_stems:
                matches += 1
                if example_hit is None:
                    example_hit = (cid, derived)
            elif example_miss is None:
                example_miss = (cid, derived)
        rate = matches / len(clip_ids)
        print(f"  {name:35s} {matches:6d}/{len(clip_ids)} ({rate:.1%})")
        if example_hit:
            print(f"      hit example:  clip_id={example_hit[0]!r} -> derived={example_hit[1]!r}")
        if example_miss:
            print(f"      miss example: clip_id={example_miss[0]!r} -> derived={example_miss[1]!r}")


if __name__ == "__main__":
    main()
