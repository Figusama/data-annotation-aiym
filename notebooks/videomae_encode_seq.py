"""
Dense sliding-window VideoMAE encoding for CTC-style continuous gloss
sequence models.

Unlike videomae_encode.py (which samples a fixed 16 frames across the
WHOLE clip -> one global feature), this processes the ENTIRE clip in
consecutive, non-overlapping 16-frame windows, spatially pools each
window's tokens (keeping the temporal axis), and concatenates windows
into one long per-clip sequence.

For a ~30s clip at 25fps (750 frames): 47 windows x 8 temporal tokens
(tubelet_size=2) = ~376 timesteps/clip -- comfortable margin above
CTC's requirement of timesteps > 2*label_length + 1, even for this
dataset's worst case (95 glosses -> needs >191).

Output per clip:
    <out_dir>/<clip_id>.npy  -> float32, shape (total_timesteps, hidden_dim)

This is substantially more compute than the global-sample encoder --
every frame of every clip gets processed, not just 16 sampled ones.
Budget for a much longer run; use the 24GB machine.

Usage:
    python videomae_encode_seq.py \
        --records surdobot_dataset_records.json \
        --video-root /path/to/videos \
        --out-dir videomae_features_seq \
        --model MCG-NJU/videomae-base \
        --window-frames 16 \
        --clip-batch-size 2 \
        --device cuda
"""

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("videomae_encode_seq")


def load_records(records_path: Path) -> list:
    with open(records_path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("records", list(data.values()))
    return data


def resolve_video_path(record: dict, video_root: Path) -> Path:
    vp = record.get("video_path")
    p = Path(vp)
    if p.is_absolute() and p.exists():
        return p
    candidate = video_root / vp
    if candidate.exists():
        return candidate
    matches = list(video_root.rglob(p.name))
    return matches[0] if matches else candidate


def read_all_frames(video_path: str):
    """Decode the entire video once. Returns (T, H, W, 3) uint8 RGB array."""
    try:
        import decord
        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(video_path, num_threads=1)
        return vr[:].asnumpy()
    except Exception as e:
        log.debug("decord failed for %s (%s); falling back to OpenCV", video_path, e)

    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"OpenCV could not open {video_path}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise IOError(f"No frames decoded from {video_path}")
    return np.stack(frames)


def make_windows(frames: np.ndarray, window_frames: int) -> list:
    """Split (T, H, W, 3) into a list of (window_frames, H, W, 3) chunks,
    padding the final short window by repeating its last frame."""
    total = frames.shape[0]
    windows = []
    for start in range(0, total, window_frames):
        chunk = frames[start:start + window_frames]
        if chunk.shape[0] < window_frames:
            pad_n = window_frames - chunk.shape[0]
            pad = np.repeat(chunk[-1:], pad_n, axis=0)
            chunk = np.concatenate([chunk, pad], axis=0)
        windows.append(chunk)
    return windows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", required=True, type=Path)
    ap.add_argument("--video-root", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--model", default="MCG-NJU/videomae-base")
    ap.add_argument("--window-frames", type=int, default=16,
                     help="Frames per window (must match tubelet/frame config the model expects, usually 16)")
    ap.add_argument("--clip-batch-size", type=int, default=2,
                     help="How many clips' windows to batch together per forward pass. "
                          "Total batch = clip_batch_size * (num_windows varies per clip, "
                          "so windows are grouped and flushed independently -- see --window-batch-size.")
    ap.add_argument("--window-batch-size", type=int, default=8,
                     help="How many windows to run through the model in a single forward pass.")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    import torch
    from transformers import VideoMAEImageProcessor, VideoMAEModel

    device = args.device if (args.device == "cpu" or torch.cuda.is_available() or args.device == "mps") else "cpu"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading %s ...", args.model)
    processor = VideoMAEImageProcessor.from_pretrained(args.model)
    model = VideoMAEModel.from_pretrained(args.model).to(device).eval()
    tubelet_size = getattr(model.config, "tubelet_size", 2)
    num_temporal_tokens_per_window = max(args.window_frames // tubelet_size, 1)

    records = load_records(args.records)
    if args.limit:
        records = records[: args.limit]
    log.info("Loaded %d records", len(records))

    def encode_windows(window_list):
        """Run a batch of windows through the model, return list of
        (num_temporal_tokens_per_window, hidden_dim) arrays, one per window."""
        results = []
        for i in range(0, len(window_list), args.window_batch_size):
            chunk = window_list[i:i + args.window_batch_size]
            videos = [list(w) for w in chunk]
            inputs = processor(videos, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            with torch.no_grad():
                out = model(pixel_values=pixel_values)
            hidden = out.last_hidden_state  # (B, seq_len, hidden_dim)
            seq_len, hidden_dim = hidden.shape[1], hidden.shape[2]
            spatial_tokens = seq_len // num_temporal_tokens_per_window
            pooled = hidden.view(hidden.shape[0], num_temporal_tokens_per_window, spatial_tokens, hidden_dim).mean(dim=2)
            for p in pooled:
                results.append(p.cpu().numpy().astype(np.float32))
        return results

    ok, skipped, failed = [], [], []
    total = len(records)
    for i, rec in enumerate(records, 1):
        clip_id = rec.get("clip_id")
        if clip_id is None:
            failed.append({"clip_id": None, "error": "missing clip_id", "record_index": i})
            continue

        out_path = args.out_dir / f"{clip_id}.npy"
        if out_path.exists() and not args.overwrite:
            skipped.append(clip_id)
            continue

        try:
            video_path = resolve_video_path(rec, args.video_root)
            if not video_path.exists():
                raise FileNotFoundError(str(video_path))
            frames = read_all_frames(str(video_path))
            windows = make_windows(frames, args.window_frames)
            window_feats = encode_windows(windows)  # list of (temporal_tokens, hidden)
            full_seq = np.concatenate(window_feats, axis=0)  # (total_timesteps, hidden)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_path, full_seq)
            ok.append(clip_id)
        except Exception as e:
            failed.append({"clip_id": clip_id, "error": str(e)})
            log.debug(traceback.format_exc())

        if i % 20 == 0 or i == total:
            log.info("Progress %d/%d | ok=%d skipped=%d failed=%d", i, total, len(ok), len(skipped), len(failed))

    summary = {
        "model": args.model,
        "window_frames": args.window_frames,
        "tubelet_size": tubelet_size,
        "total_records": total,
        "encoded": len(ok),
        "already_present_skipped": len(skipped),
        "failed": len(failed),
        "failures": failed,
    }
    summary_path = args.out_dir / "videomae_seq_encoding_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Done. encoded=%d skipped=%d failed=%d -> %s", len(ok), len(skipped), len(failed), summary_path)


if __name__ == "__main__":
    sys.exit(main())
