"""
Loads a checkpoint and checks what fraction of frame-level predictions
are the CTC blank token, on a real validation batch -- confirms or
rules out blank-collapse as the cause of a frozen MIL loss.
"""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import SurdobotFusionDataset, collate_fn
from slr_model import SLRModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--records", required=True)
    ap.add_argument("--video-feat-dir", required=True)
    ap.add_argument("--keypoint-dir", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device)
    vocab_size = ckpt["vocab_size"]

    ds = SurdobotFusionDataset(args.records, args.video_feat_dir, args.keypoint_dir,
                                args.vocab, require_keypoints=True, split=None)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    batch = next(iter(loader))

    model = SLRModel(vocab_size=vocab_size).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        ctc_logits, mil_scores = model(batch["video_feat"].to(device), batch["keypoints"].to(device))

    preds = ctc_logits.argmax(dim=-1)  # (B, T)
    blank_idx = vocab_size
    blank_fraction = (preds == blank_idx).float().mean().item()
    print(f"Checkpoint epoch: {ckpt.get('epoch')}, val_loss at save: {ckpt.get('val_loss'):.3f}")
    print(f"Fraction of predicted frames = blank token: {blank_fraction:.1%}")

    # distribution of non-blank predictions (how many distinct classes get predicted at all)
    non_blank = preds[preds != blank_idx]
    n_distinct = non_blank.unique().numel() if non_blank.numel() else 0
    print(f"Distinct non-blank classes predicted in this batch: {n_distinct} / {vocab_size}")

    print(f"\nmil_scores stats: min={mil_scores.min().item():.6f} "
          f"max={mil_scores.max().item():.6f} mean={mil_scores.mean().item():.6f}")

    if blank_fraction > 0.95:
        print("\n*** Strong sign of blank collapse: model predicts blank almost everywhere. ***")


if __name__ == "__main__":
    main()
