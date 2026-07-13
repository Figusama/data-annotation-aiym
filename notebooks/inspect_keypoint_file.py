"""
Inspects a keypoint .npz file for a colliding stem, looking for signs
that two unrelated videos got merged into one file (e.g. an implausibly
large frame count, or duplicate/overlapping frame_numbers with
conflicting hand data).
"""
import argparse
import numpy as np
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-path", required=True, type=Path)
    args = ap.parse_args()

    data = np.load(args.npz_path)
    frame_numbers = data["frame_numbers"]
    valid = data["hand_valid"]

    print(f"File: {args.npz_path}")
    print(f"Total frames stored: {len(frame_numbers)}")
    print(f"Frame number range: {frame_numbers.min()} - {frame_numbers.max()}")
    print(f"Frame numbers sorted & contiguous?: {np.array_equal(frame_numbers, np.sort(frame_numbers))}")

    # look for large gaps in frame numbers -- a gap much bigger than the
    # typical spacing could indicate two separate videos' frame ranges
    # got concatenated rather than being one continuous clip
    diffs = np.diff(np.sort(frame_numbers))
    if len(diffs):
        print(f"Median frame gap: {np.median(diffs)}")
        print(f"Max frame gap: {diffs.max()} (at sorted position {diffs.argmax()})")
        big_gaps = np.where(diffs > 5 * max(np.median(diffs), 1))[0]
        if len(big_gaps):
            print(f"\n{len(big_gaps)} suspiciously large gap(s) found -- possible boundary between two merged videos:")
            for idx in big_gaps[:5]:
                sorted_frames = np.sort(frame_numbers)
                print(f"  jump from frame {sorted_frames[idx]} to {sorted_frames[idx+1]} (gap of {diffs[idx]})")

    print(f"\nHand detection rate (left): {valid[:, 0].mean():.1%}")
    print(f"Hand detection rate (right): {valid[:, 1].mean():.1%}")


if __name__ == "__main__":
    main()
