"""
VideoMAE Encoding for the Surdobot SLR Pipeline
=================================================

Extracts VideoMAE features for every clip listed in
`surdobot_dataset_records.json` (the file produced by the annotation
preprocessing stage of the pipeline). Mirrors the conventions already
used by the keypoint stream:

  - one output file per clip (`<clip_id>.npy`), not one giant blob
  - a coverage/failure summary written at the end, matching the
    validation habits described in Part 3 of the blueprint
  - resumable: clips that already have an output file are skipped

Output per clip:
    <out_dir>/<clip_id>.npy   -> float32 array, shape (hidden_dim,)
                                  (mean-pooled VideoMAE token embedding)

If you need per-frame / per-token features instead of a single pooled
vector (e.g. to align with the (T, 2, 21, 2) keypoint tensors for a
CTC-style fusion), pass --pooling none, which saves shape
(num_frames_after_tubelet, hidden_dim) instead.

Usage
-----
    pip install torch transformers decord opencv-python-headless tqdm

    python videomae_encode.py \
        --records surdobot_dataset_records.json \
        --video-root /path/to/surdobot/videos \
        --out-dir videomae_features \
        --model MCG-NJU/videomae-base \
        --num-frames 16 \
        --batch-size 4 \
        --device cuda

Then merge into your training manifest with `--write-index`, which
produces `videomae_index.json`: {clip_id: relative_path_to_npy}.
"""

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("videomae_encode")


# --------------------------------------------------------------------------
# Video frame sampling
# --------------------------------------------------------------------------

def sample_frame_indices(total_frames: int, num_frames: int) -> np.ndarray:
    """Uniformly sample `num_frames` indices across the clip.

    If the clip has fewer frames than requested, indices are repeated
    (clamped) rather than raising, so short clips still produce a
    fixed-size input tensor.
    """
    if total_frames <= 0:
        return np.zeros(num_frames, dtype=np.int64)
    if total_frames >= num_frames:
        return np.linspace(0, total_frames - 1, num_frames).round().astype(np.int64)
    # short clip: repeat frames to pad up to num_frames
    idx = np.arange(total_frames)
    reps = int(np.ceil(num_frames / total_frames))
    idx = np.tile(idx, reps)[:num_frames]
    return np.sort(idx)


def read_video_frames(video_path: str, num_frames: int) -> np.ndarray:
    """Return an (num_frames, H, W, 3) uint8 RGB array.

    Tries decord first (fast, GPU-friendly seeking); falls back to
    OpenCV if decord isn't available or fails to open the file.
    """
    try:
        import decord
        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(video_path, num_threads=1)
        total = len(vr)
        idx = sample_frame_indices(total, num_frames)
        frames = vr.get_batch(idx).asnumpy()  # (num_frames, H, W, 3) RGB
        return frames
    except Exception as decord_err:
        log.debug("decord failed for %s (%s); falling back to OpenCV", video_path, decord_err)

    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"OpenCV could not open {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = set(sample_frame_indices(total, num_frames).tolist())
    frames = []
    frame_no = 0
    max_idx = max(idx) if idx else 0
    while frame_no <= max_idx:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_no in idx:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_no += 1
    cap.release()
    if not frames:
        raise IOError(f"No frames decoded from {video_path}")
    while len(frames) < num_frames:
        frames.append(frames[-1])  # pad by repeating last frame
    return np.stack(frames[:num_frames])


# --------------------------------------------------------------------------
# Records I/O
# --------------------------------------------------------------------------

def load_records(records_path: Path) -> list:
    with open(records_path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        # tolerate {"records": [...]}-style wrapping
        data = data.get("records", list(data.values()))
    if not isinstance(data, list):
        raise ValueError(f"Unexpected structure in {records_path}: expected a list of records")
    return data


def resolve_video_path(record: dict, video_root: Path) -> Path:
    vp = record.get("video_path")
    if vp is None:
        raise KeyError("record has no 'video_path' field")
    p = Path(vp)
    if p.is_absolute() and p.exists():
        return p
    candidate = video_root / vp
    if candidate.exists():
        return candidate
    # last resort: try matching by filename anywhere under video_root
    matches = list(video_root.rglob(p.name))
    if matches:
        return matches[0]
    return candidate  # will fail existence check upstream, surfaced in the failure log


# --------------------------------------------------------------------------
# Main encoding loop
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", required=True, type=Path, help="Path to surdobot_dataset_records.json")
    ap.add_argument("--video-root", required=True, type=Path, help="Root directory videos are relative to")
    ap.add_argument("--out-dir", required=True, type=Path, help="Where to write <clip_id>.npy feature files")
    ap.add_argument("--model", default="MCG-NJU/videomae-base", help="HF checkpoint id or local path")
    ap.add_argument("--num-frames", type=int, default=16, help="Frames sampled per clip (must match model's config, usually 16)")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    ap.add_argument("--pooling", default="mean", choices=["mean", "none"],
                     help="'mean' -> one (hidden_dim,) vector per clip. "
                          "'none' -> keep all tokens, shape (seq_len, hidden_dim)")
    ap.add_argument("--limit", type=int, default=None, help="Only process the first N records (debugging)")
    ap.add_argument("--write-index", action="store_true", help="Also write videomae_index.json mapping clip_id -> npy path")
    ap.add_argument("--overwrite", action="store_true", help="Reprocess clips even if an .npy already exists")
    args = ap.parse_args()

    import torch
    from transformers import VideoMAEImageProcessor, VideoMAEModel

    device = args.device if (args.device == "cpu" or torch.cuda.is_available() or args.device == "mps") else "cpu"
    if device != args.device:
        log.warning("Requested device %s unavailable, falling back to cpu", args.device)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading %s ...", args.model)
    processor = VideoMAEImageProcessor.from_pretrained(args.model)
    model = VideoMAEModel.from_pretrained(args.model).to(device).eval()

    records = load_records(args.records)
    if args.limit:
        records = records[: args.limit]
    log.info("Loaded %d records from %s", len(records), args.records)

    ok, skipped, failed = [], [], []

    def flush_batch(batch_clip_ids, batch_frames):
        """Run one forward pass over a batch of pre-sampled frame stacks."""
        if not batch_frames:
            return
        inputs = processor(list(batch_frames), return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        with torch.no_grad():
            out = model(pixel_values=pixel_values)
        hidden = out.last_hidden_state  # (B, seq_len, hidden_dim)
        for cid, h in zip(batch_clip_ids, hidden):
            feat = h.mean(dim=0) if args.pooling == "mean" else h
            np.save(args.out_dir / f"{cid}.npy", feat.cpu().numpy().astype(np.float32))
            ok.append(cid)

    batch_clip_ids, batch_frames = [], []
    total = len(records)
    for i, rec in enumerate(records, 1):
        clip_id = rec.get("clip_id")
        if clip_id is None:
            failed.append({"clip_id": None, "error": "missing clip_id field", "record_index": i})
            continue

        out_path = args.out_dir / f"{clip_id}.npy"
        if out_path.exists() and not args.overwrite:
            skipped.append(clip_id)
            continue

        try:
            video_path = resolve_video_path(rec, args.video_root)
            if not video_path.exists():
                raise FileNotFoundError(str(video_path))
            frames = read_video_frames(str(video_path), args.num_frames)  # (T, H, W, 3) uint8
            batch_clip_ids.append(clip_id)
            batch_frames.append(frames)
        except Exception as e:
            failed.append({"clip_id": clip_id, "error": str(e)})
            log.debug(traceback.format_exc())
            continue

        if len(batch_frames) >= args.batch_size:
            try:
                flush_batch(batch_clip_ids, batch_frames)
            except Exception as e:
                for cid in batch_clip_ids:
                    failed.append({"clip_id": cid, "error": f"forward pass failed: {e}"})
            batch_clip_ids, batch_frames = [], []

        if i % 50 == 0 or i == total:
            log.info("Progress %d/%d | ok=%d skipped=%d failed=%d", i, total, len(ok), len(skipped), len(failed))

    # flush any remainder
    try:
        flush_batch(batch_clip_ids, batch_frames)
    except Exception as e:
        for cid in batch_clip_ids:
            failed.append({"clip_id": cid, "error": f"forward pass failed: {e}"})

    summary = {
        "model": args.model,
        "num_frames": args.num_frames,
        "pooling": args.pooling,
        "total_records": total,
        "encoded": len(ok),
        "already_present_skipped": len(skipped),
        "failed": len(failed),
        "failures": failed,
    }
    summary_path = args.out_dir / "videomae_encoding_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Done. encoded=%d skipped=%d failed=%d -> summary at %s",
              len(ok), len(skipped), len(failed), summary_path)

    if args.write_index:
        index = {}
        for rec in records:
            cid = rec.get("clip_id")
            p = args.out_dir / f"{cid}.npy"
            if cid and p.exists():
                index[cid] = str(p)
        index_path = args.out_dir / "videomae_index.json"
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)
        log.info("Wrote index for %d clips -> %s", len(index), index_path)


if __name__ == "__main__":
    sys.exit(main())
