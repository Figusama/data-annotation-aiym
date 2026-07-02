"""
Loads config.yaml and resolves ${paths.x} interpolations.
Usage:
    from configs.load_config import load_config
    cfg = load_config()
    print(cfg.paths.surdobot_videos)
"""

import re
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Any
import yaml


def _resolve(value: str, flat: dict) -> str:
    """Replace ${section.key} references in a string value."""
    pattern = re.compile(r"\$\{([^}]+)\}")
    def replacer(m):
        key = m.group(1)
        return str(flat.get(key, m.group(0)))
    for _ in range(5):   # up to 5 levels of nesting
        new = pattern.sub(replacer, value)
        if new == value:
            break
        value = new
    return value


def _flatten(d: dict, prefix: str = "") -> dict:
    """Flatten nested dict to dot-notation keys."""
    out = {}
    for k, v in d.items():
        full = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, full))
        else:
            out[full] = v
    return out


def _resolve_all(d: dict, flat: dict) -> dict:
    """Recursively resolve interpolations in all string values."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _resolve_all(v, flat)
        elif isinstance(v, str):
            out[k] = _resolve(v, flat)
        else:
            out[k] = v
    return out


class DotDict:
    """Dot-access wrapper around a nested dict."""
    def __init__(self, d: dict):
        for k, v in d.items():
            setattr(self, k, DotDict(v) if isinstance(v, dict) else v)

    def __repr__(self):
        return f"DotDict({self.__dict__})"

    def get(self, key, default=None):
        return getattr(self, key, default)


def load_config(path: str | Path | None = None) -> DotDict:
    if path is None:
        path = Path(__file__).parent / "config.yaml"
    with open(path) as f:
        raw = yaml.safe_load(f)

    flat = _flatten(raw)
    # Iteratively resolve until stable
    for _ in range(5):
        new_flat = {k: (_resolve(v, flat) if isinstance(v, str) else v)
                    for k, v in flat.items()}
        if new_flat == flat:
            break
        flat = new_flat

    resolved = _resolve_all(raw, flat)
    return DotDict(resolved)


if __name__ == "__main__":
    cfg = load_config()
    print("surdobot_videos :", cfg.paths.surdobot_videos)
    print("elarna_videos   :", cfg.paths.elarna_videos)
    print("asr model       :", cfg.asr.model_size)
    print("videomae model  :", cfg.features.videomae_model)
