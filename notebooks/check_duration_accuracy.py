"""
Compares recorded duration_sec against the ACTUAL duration of the video
file on disk (frame_count / fps), for a random sample of clips. Tells
you whether duration_sec is a real probed value or a hardcoded default
that happened to get written for every record.
"""
import argparse
import json
import random
from pathlib import Path


def load_records(path):
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("records", list(data.values()))
    return data


def resolve_video_path(record, video_root):
    vp = record.get("video_path")
    p = Path(vp)
    if p.is_absolute() and p.exists():
        return p
    candidate = video_root / vp
    if candidate.exists():
        return candidate
    matches = list(video_root.rglob(p.name))
    return matches[0] if matches else candidate


def real_duration(video_path):
    import decord
    vr = decord.VideoReader(str(video_path), num_threads=1)
    fps = vr.get_avg_fps()
    return len(vr) / fps, len(vr), fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True, type=Path)
    ap.add_argument("--video-root", required=True, type=Path)
    ap.add_argument("--sample", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    records = load_records(args.records)
    random.seed(args.seed)
    sample = random.sample(records, min(args.sample, len(records)))

    mismatches = 0
    checked = 0
    print(f"{'clip_id':60s} {'recorded':>10s} {'actual':>10s} {'frames':>8s} {'fps':>6s}")
    for rec in sample:
        clip_id = rec.get("clip_id")
        try:
            vp = resolve_video_path(rec, args.video_root)
            actual, n_frames, fps = real_duration(vp)
        except Exception as e:
            print(f"{clip_id:60s} could not probe: {e}")
            continue
        recorded = rec.get("duration_sec")
        checked += 1
        diff = abs(actual - recorded) if recorded is not None else None
        flag = "  <-- MISMATCH" if diff and diff > 0.5 else ""
        if flag:
            mismatches += 1
        print(f"{clip_id:60s} {recorded!s:>10s} {actual:10.2f} {n_frames:8d} {fps:6.1f}{flag}")

    print(f"\n{mismatches}/{checked} sampled clips have recorded duration_sec off by >0.5s from actual.")
    if mismatches == checked and checked > 0:
        print("*** ALL sampled durations are wrong -- duration_sec is not a real probed value. ***")
        print("Recommend: don't trust duration_sec anywhere downstream; re-probe if you need it "
              "for filtering/splitting logic. The windowed VideoMAE encoder already ignores it "
              "and reads real frame counts directly, so it's unaffected.")


if __name__ == "__main__":
    main()
