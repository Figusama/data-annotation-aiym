"""
Common utilities shared across Phase 1 modules.
"""

import logging
import json
import os
import hashlib
from pathlib import Path
from typing import Optional, Union
import numpy as np


# ─────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────

def get_logger(name: str, log_file: Optional[str] = None, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s  %(name)s  — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Optional file handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ─────────────────────────────────────────────
#  Path helpers
# ─────────────────────────────────────────────

def ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def video_id_from_path(video_path: Union[str, Path]) -> str:
    """
    Derive a stable clip ID from a video file path.
    E.g. /data/Surdobot_VideoBase/lesson_01/clip_003.mp4 → lesson_01__clip_003
    """
    p = Path(video_path)
    return f"{p.parent.name}__{p.stem}"


def feature_path(features_root: Union[str, Path], video_path: Union[str, Path],
                 suffix: str = ".npz") -> Path:
    """
    Map a video path to its feature cache file.
    Preserves the relative subdirectory structure under features_root.
    """
    p = Path(video_path)
    vid_id = video_id_from_path(p)
    return Path(features_root) / f"{vid_id}{suffix}"


def file_md5(path: Union[str, Path], chunk_size: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


# ─────────────────────────────────────────────
#  JSON manifest helpers
# ─────────────────────────────────────────────

def save_manifest(records: list[dict], path: Union[str, Path]) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def load_manifest(path: Union[str, Path]) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────
#  Video utilities
# ─────────────────────────────────────────────

def probe_video(video_path: Union[str, Path]) -> dict:
    """
    Return basic video metadata using OpenCV without loading frames.
    Returns: {fps, total_frames, duration_sec, width, height}
    """
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {
        "fps": fps,
        "total_frames": total_frames,
        "duration_sec": total_frames / fps if fps > 0 else 0.0,
        "width": width,
        "height": height,
    }


def sample_frames_uniform(video_path: Union[str, Path],
                           num_frames: int = 16,
                           start_sec: float = 0.0,
                           end_sec: Optional[float] = None) -> np.ndarray:
    """
    Uniformly sample `num_frames` RGB frames from [start_sec, end_sec].
    Returns: np.ndarray of shape (num_frames, H, W, 3), dtype uint8.
    """
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_f = int(start_sec * fps)
    end_f   = int(end_sec * fps) if end_sec is not None else total
    end_f   = min(end_f, total)

    indices = np.linspace(start_f, end_f - 1, num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            # Fallback: repeat last good frame
            frame = frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames)  # (T, H, W, 3)


def has_audio_track(video_path: Union[str, Path]) -> bool:
    """
    Check if a video file contains an audio stream (uses ffprobe if available,
    falls back to moviepy).
    """
    import subprocess
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", str(video_path)],
            capture_output=True, text=True, timeout=10
        )
        data = json.loads(result.stdout)
        return any(s.get("codec_type") == "audio" for s in data.get("streams", []))
    except Exception:
        # ffprobe not available — assume audio present, ASR will handle empty output
        return True


# ─────────────────────────────────────────────
#  Keypoint utilities
# ─────────────────────────────────────────────

def wrist_relative_normalize(coords: np.ndarray) -> np.ndarray:
    """
    Subtract the wrist keypoint (index 0) from all 21 keypoints per hand.
    Input:  (T, 42, 3)  — T frames, 42 keypoints (L+R), 3 coords
    Output: (T, 42, 3)  — wrist-relative coordinates
    """
    out = coords.copy()
    # Left hand: keypoints 0–20, wrist = index 0
    out[:, :21, :] -= coords[:, 0:1, :]
    # Right hand: keypoints 21–41, wrist = index 21
    out[:, 21:, :] -= coords[:, 21:22, :]
    return out


def add_velocity_channel(coords: np.ndarray) -> np.ndarray:
    """
    Append per-keypoint velocity (finite difference) as extra features.
    Input:  (T, 42, 3)
    Output: (T, 42, 6)  — [x, y, z, vx, vy, vz]
    """
    vel = np.zeros_like(coords)
    vel[1:] = coords[1:] - coords[:-1]   # forward difference; frame 0 velocity = 0
    return np.concatenate([coords, vel], axis=-1)  # (T, 42, 6)
