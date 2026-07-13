"""
Cross-checks coverage across the three Surdobot preprocessing outputs:
  - surdobot_dataset_records.json  (annotations: clip_id, gloss_bag, split, ...)
  - <videomae_dir>/*.npy           (VideoMAE features, one per clip_id)
  - <keypoint_dir>/*.npz           (keypoints, one per video_id)

Reports:
  - how many records have a matching VideoMAE feature file
  - how many records have a matching keypoint file
  - how many have BOTH (i.e. are actually usable for a fused model)
  - breakdown by split (train/val/test), so you know if failures are
    skewing one split disproportionately
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


def video_id_from_clip_id(clip_id: str) -> str:
    # Keypoint files are saved as <filename_stem>.npz with no session-folder
    # prefix, even though clip_id in the annotation records includes one
    # (e.g. "27_10/resized_cropped_....mp4"). Confirmed empirically against
    # files on disk -- see diagnose_keypoint_mapping.py.
    return Path(clip_id).stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True, type=Path)
    ap.add_argument("--videomae-dir", required=True, type=Path)
    ap.add_argument("--keypoint-dir", required=True, type=Path)
    ap.add_argument("--dump-missing", type=Path, default=None,
                     help="Optional path to write full list of clip_ids missing either feature")
    args = ap.parse_args()

    records = load_records(args.records)
    print(f"Total records: {len(records)}")

    # Since keypoint filenames drop the session-folder prefix, two clip_ids
    # from DIFFERENT folders that happen to share the same filename stem
    # would silently collide onto the same keypoint file -- check for that
    # before trusting any match as correct.
    stem_to_clip_ids = {}
    for rec in records:
        cid = rec.get("clip_id")
        if not cid:
            continue
        stem = video_id_from_clip_id(cid)
        stem_to_clip_ids.setdefault(stem, []).append(cid)
    collisions = {stem: cids for stem, cids in stem_to_clip_ids.items() if len(set(cids)) > 1}
    if collisions:
        print(f"\n*** WARNING: {len(collisions)} filename stems are shared by multiple distinct clip_ids ***")
        print("These clips would incorrectly receive the same keypoint file:")
        for stem, cids in list(collisions.items())[:10]:
            print(f"  stem={stem!r} <- {cids}")
        if len(collisions) > 10:
            print(f"  ... and {len(collisions) - 10} more")
    else:
        print("\nNo filename-stem collisions across different clip_ids -- mapping is safe.")

    colliding_clip_ids = set()
    for stem, cids in collisions.items():
        colliding_clip_ids.update(cids)

    missing_videomae, missing_keypoints, missing_both, excluded_collision = [], [], [], []
    have_both = 0
    split_counts = Counter()
    split_have_both = Counter()

    for rec in records:
        clip_id = rec.get("clip_id")
        split = rec.get("split")
        split_counts[split] += 1
        if clip_id is None:
            continue

        if clip_id in colliding_clip_ids:
            excluded_collision.append(clip_id)
            continue

        vmae_path = args.videomae_dir / f"{clip_id}.npy"
        vid_id = video_id_from_clip_id(clip_id)
        kp_path = args.keypoint_dir / f"{vid_id}.npz"

        has_vmae = vmae_path.exists()
        has_kp = kp_path.exists()

        if not has_vmae:
            missing_videomae.append(clip_id)
        if not has_kp:
            missing_keypoints.append(clip_id)
        if not has_vmae and not has_kp:
            missing_both.append(clip_id)
        if has_vmae and has_kp:
            have_both += 1
            split_have_both[split] += 1

    print(f"\nExcluded (unsafe: filename-stem collision): {len(excluded_collision)}")

    print(f"\nMissing VideoMAE feature: {len(missing_videomae)} ({len(missing_videomae)/len(records):.1%})")
    print(f"Missing keypoint file:    {len(missing_keypoints)} ({len(missing_keypoints)/len(records):.1%})")
    print(f"Missing BOTH:             {len(missing_both)}")
    print(f"Usable (have both):       {have_both} ({have_both/len(records):.1%})")

    print("\nPer-split breakdown (records / usable-with-both):")
    for split, total in split_counts.items():
        usable = split_have_both.get(split, 0)
        print(f"  {split!r}: {usable}/{total} ({usable/total:.1%})")

    if args.dump_missing:
        with open(args.dump_missing, "w") as f:
            json.dump({
                "missing_videomae": missing_videomae,
                "missing_keypoints": missing_keypoints,
                "missing_both": missing_both,
                "excluded_collision": excluded_collision,
            }, f, indent=2)
        print(f"\nWrote missing-clip lists to {args.dump_missing}")


if __name__ == "__main__":
    main()