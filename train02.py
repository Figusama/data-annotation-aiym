"""
Training loop for SLRModel on the Surdobot fusion dataset.

Uses the split field already present in surdobot_dataset_records.json
(train/val/test) -- only train and val are used here, per your call to
skip test for now. Works fine with partial encoding progress (currently
~10.6k/35.3k clips have both video+keypoint features); as more clips
finish encoding, just rerun -- the dataset picks up whatever exists on
disk each time.

Loss = ctc_weight * CTC_loss + mil_weight * MIL_loss, per the
blueprint's hybrid formulation. MIL_loss is binary cross-entropy
between the model's per-gloss bag scores and a multi-hot vector of
which glosses are actually present in the clip's gloss_bag -- this
part doesn't care about order, so it's a useful auxiliary signal even
though you've confirmed order IS reliable for the CTC head.

Usage:
    python train.py \
        --records ./notebooks/surdobot_dataset_records.json \
        --video-feat-dir /path/to/Videomae_features_seq \
        --keypoint-dir /path/to/surdobot_raw_hs_keypoints \
        --vocab ./notebooks/surdobot_vocab.json \
        --checkpoint-dir ./checkpoints \
        --epochs 20 --batch-size 4 --lr 1e-4 --device cuda

rm -rf ./checkpoints/*
cp ~/Downloads/train02.py .
python train02.py \
  --records ./notebooks/surdobot_dataset_records.json \
  --video-feat-dir /media/nurobot427/T7/KRSL_OnlineSchool/Videomae_features_seq \
  --keypoint-dir /media/nurobot427/T7/KRSL_OnlineSchool/surdobot_raw_hs_keypoints \
  --vocab ./notebooks/surdobot_vocab.json \
  --checkpoint-dir ./checkpoints \
  --epochs 20 --batch-size 24 --lr 1e-4 --device cuda:1 \
  --temp-resplit --val-fraction 0.1 \
  --ctc-weight 0.2 --mil-weight 2.0
"""
import argparse
import time
from pathlib import Path
import jiwer
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import SurdobotFusionDataset, collate_fn, resplit_by_session, find_colliding_clip_ids, load_records
from slr_model02 import SLRModel

import jiwer
import numpy as np

def decode_and_compute_wer(ctc_logits, video_lengths, targets, target_lengths, itos, blank_id):
    """
    Decodes CTC logits using greedy decoding and computes WER against targets.
    """
    # Get the most likely class at each timestep: (B, T)
    preds = ctc_logits.argmax(dim=-1).cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    hypotheses = []
    references = []
    
    target_offset = 0
    for i in range(len(video_lengths)):
        # --- Decode Prediction ---
        pred_seq = preds[i, :video_lengths[i]]
        decoded_pred = []
        prev_token = -1
        for token in pred_seq:
            # Collapse repeated tokens and ignore the CTC blank token
            if token != prev_token and token != blank_id:
                decoded_pred.append(itos[token])
            prev_token = token
        
        # Join into a space-separated string
        hypotheses.append(" ".join(decoded_pred) if decoded_pred else "")
        
        # --- Decode Target ---
        target_len = int(target_lengths[i])
        target_seq = targets_np[target_offset : target_offset + target_len]
        decoded_target = [itos[t] for t in target_seq]
        references.append(" ".join(decoded_target) if decoded_target else "")
        
        target_offset += target_len

    # Edge case guard: jiwer raises an error if all references are completely empty
    valid_refs = [r for r in references if r.strip()]
    if not valid_refs:
        return 0.0 if not any(hypotheses) else 1.0

    return jiwer.wer(references, hypotheses)

def multi_hot_targets(target_concat, target_lengths, vocab_size, device):
    """Builds a (B, vocab_size) multi-hot tensor from the CTC-style
    concatenated target indices + per-sample lengths."""
    B = len(target_lengths)
    multi_hot = torch.zeros(B, vocab_size, device=device)
    offset = 0
    for i, length in enumerate(target_lengths):
        length = int(length)
        idxs = target_concat[offset:offset + length]
        multi_hot[i, idxs] = 1.0
        offset += length
    return multi_hot

def run_epoch(model, loader, device, vocab_size, ctc_loss_fn, mil_loss_fn,
              ctc_weight, mil_weight, optimizer=None, scaler=None, itos=None):
    """optimizer=None -> eval mode (no backward pass)."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    use_amp = scaler is not None and device == "cuda"

    total_loss, total_ctc, total_mil, total_wer, n_batches = 0.0, 0.0, 0.0, 0.0, 0

    for batch in loader:
        video_feat = batch["video_feat"].to(device)
        keypoints = batch["keypoints"].to(device)

        with torch.set_grad_enabled(is_train):
            with torch.autocast(device_type="cuda", enabled=use_amp):
                ctc_logits, mil_scores = model(video_feat, keypoints)
                log_probs = ctc_logits.float().log_softmax(dim=-1).transpose(0, 1)

                # MIL Loss -- BCEWithLogitsLoss is autocast-safe (unlike
                # BCELoss), so this can stay inside the autocast block now.
                multi_hot = multi_hot_targets(batch["targets"], batch["target_lengths"], vocab_size, device)
                mil_loss = mil_loss_fn(mil_scores.float(), multi_hot)

            # CTC Loss
            ctc_loss = ctc_loss_fn(
                log_probs.cpu(),
                batch["targets"],
                batch["video_lengths"],
                batch["target_lengths"],
            ).to(device)

            loss = ctc_weight * ctc_loss + mil_weight * mil_loss

        if is_train:
            optimizer.zero_grad()
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        # Calculate WER for the batch
        # vocab_size serves as the blank_id since the CTC head outputs vocab_size + 1
        batch_wer = decode_and_compute_wer(
            ctc_logits, batch["video_lengths"], batch["targets"], 
            batch["target_lengths"], itos, blank_id=vocab_size
        )

        total_loss += loss.item()
        total_ctc += ctc_loss.item()
        total_mil += mil_loss.item()
        total_wer += batch_wer
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "ctc_loss": total_ctc / max(n_batches, 1),
        "mil_loss": total_mil / max(n_batches, 1),
        "wer": total_wer / max(n_batches, 1)
    }

# def run_epoch(model, loader, device, vocab_size, ctc_loss_fn, mil_loss_fn,
#               ctc_weight, mil_weight, optimizer=None, scaler=None):
#     """optimizer=None -> eval mode (no backward pass)."""
#     is_train = optimizer is not None
#     model.train() if is_train else model.eval()
#     use_amp = scaler is not None and device == "cuda"

#     total_loss, total_ctc, total_mil, n_batches = 0.0, 0.0, 0.0, 0

#     for batch in loader:
#         video_feat = batch["video_feat"].to(device)
#         keypoints = batch["keypoints"].to(device)

#         with torch.set_grad_enabled(is_train):
#             with torch.autocast(device_type="cuda", enabled=use_amp):
#                 ctc_logits, mil_scores = model(video_feat, keypoints)
#                 log_probs = ctc_logits.float().log_softmax(dim=-1).transpose(0, 1)  # CTC needs fp32

#             # BCELoss is disallowed under autocast (raises regardless of
#             # input dtype -- it's a hard guard, not a dtype check), so
#             # this must run outside the `with torch.autocast(...)` block.
#             multi_hot = multi_hot_targets(batch["targets"], batch["target_lengths"], vocab_size, device)
#             mil_loss = mil_loss_fn(mil_scores.float().clamp(1e-6, 1 - 1e-6), multi_hot)

#             # CTCLoss on CPU stays fp32 regardless of autocast
#             ctc_loss = ctc_loss_fn(
#                 log_probs.cpu(),
#                 batch["targets"],
#                 batch["video_lengths"],
#                 batch["target_lengths"],
#             ).to(device)

#             loss = ctc_weight * ctc_loss + mil_weight * mil_loss

#         if is_train:
#             optimizer.zero_grad()
#             if use_amp:
#                 scaler.scale(loss).backward()
#                 scaler.unscale_(optimizer)
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
#                 scaler.step(optimizer)
#                 scaler.update()
#             else:
#                 loss.backward()
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
#                 optimizer.step()

#         total_loss += loss.item()
#         total_ctc += ctc_loss.item()
#         total_mil += mil_loss.item()
#         n_batches += 1

#     return {
#         "loss": total_loss / max(n_batches, 1),
#         "ctc_loss": total_ctc / max(n_batches, 1),
#         "mil_loss": total_mil / max(n_batches, 1),
#     }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--video-feat-dir", required=True)
    ap.add_argument("--keypoint-dir", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--checkpoint-dir", required=True, type=Path)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ctc-weight", type=float, default=1.0)
    ap.add_argument("--mil-weight", type=float, default=0.3)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps", "cuda:0", "cuda:1"])
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--temp-resplit", action="store_true",
                     help="TEMPORARY: val clips for the real records.json split aren't "
                          "all encoded yet. This re-splits only the currently-usable "
                          "clips by SOURCE SESSION (not clip_id) into train/val, so "
                          "clips from the same recording session never leak across the "
                          "split. Switch back to the real split field once val clips "
                          "finish encoding.")
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--amp", action="store_true", help="Use mixed precision training (recommended, saves significant GPU memory)")
    args = ap.parse_args()

    device = args.device if (args.device == "cpu" or torch.cuda.is_available() or args.device == "mps") else "cpu"
    if device != args.device:
        print(f"Requested device {args.device!r} unavailable, falling back to cpu")
    print(f"Using device: {device}")

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if args.temp_resplit:
        from pathlib import Path as _Path
        all_records = load_records(args.records)
        colliding = find_colliding_clip_ids(all_records)
        train_records, val_records = resplit_by_session(
            all_records, _Path(args.video_feat_dir), _Path(args.keypoint_dir), colliding,
            val_fraction=args.val_fraction, require_keypoints=True,
        )
        train_ds = SurdobotFusionDataset.from_records(train_records, args.video_feat_dir, args.keypoint_dir, args.vocab)
        val_ds = SurdobotFusionDataset.from_records(val_records, args.video_feat_dir, args.keypoint_dir, args.vocab)
    else:
        train_ds = SurdobotFusionDataset(
            args.records, args.video_feat_dir, args.keypoint_dir, args.vocab,
            require_keypoints=True, split="train",
        )
        val_ds = SurdobotFusionDataset(
            args.records, args.video_feat_dir, args.keypoint_dir, args.vocab,
            require_keypoints=True, split="val",
        )
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise SystemExit(
            "Train or val set is empty. This can happen with partial encoding "
            "progress if val clips haven't been encoded yet -- check counts above."
        )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               collate_fn=collate_fn, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=args.num_workers)
    
    
    vocab_size = len(train_ds.stoi)
    itos = train_ds.itos 

    model = SLRModel(vocab_size=vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # 1. Initialize the Scheduler
    # Reduces LR by half if validation WER doesn't improve for 3 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    ctc_loss_fn = nn.CTCLoss(blank=vocab_size, zero_infinity=True)
    # DualHead now returns RAW LOGITS (unbounded) for mil_scores, not
    # probabilities -- BCEWithLogitsLoss expects exactly that, and is
    # numerically safe (internally applies a stable sigmoid+log combo,
    # unlike a manual sigmoid+BCELoss which can under/overflow). No
    # clamp needed or wanted: clamping raw logits into (0,1) would kill
    # gradients for almost every real-valued output, same bug as before.
    mil_loss_fn = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device == "cuda"))
    
    # 2. Setup Early Stopping & Best Checkpoint Tracking
    best_val_loss = float("inf")
    best_val_wer = float("inf")
    early_stopping_patience = 8
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        
        # Pass 'itos' to both training and validation runs
        train_metrics = run_epoch(model, train_loader, device, vocab_size,
                                   ctc_loss_fn, mil_loss_fn,
                                   args.ctc_weight, args.mil_weight, optimizer, scaler, itos)
        val_metrics = run_epoch(model, val_loader, device, vocab_size,
                                 ctc_loss_fn, mil_loss_fn,
                                 args.ctc_weight, args.mil_weight, None, None, itos)
        dt = time.time() - t0

        print(f"[epoch {epoch:3d}/{args.epochs}] "
              f"train_loss={train_metrics['loss']:.3f} train_WER={train_metrics['wer']:.3f} | "
              f"val_loss={val_metrics['loss']:.3f} val_WER={val_metrics['wer']:.3f} | "
              f"{dt:.1f}s")

        # 3. Step the Scheduler using Validation WER
        scheduler.step(val_metrics['wer'])

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_metrics["loss"],
            "val_wer": val_metrics["wer"],
            "vocab_size": vocab_size,
        }
        
        torch.save(ckpt, args.checkpoint_dir / "last.pt")
        
        # Save based on best Loss
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(ckpt, args.checkpoint_dir / "best_loss.pt")

        # 4. Save based on best WER and Early Stopping logic
        if val_metrics["wer"] < best_val_wer:
            best_val_wer = val_metrics["wer"]
            torch.save(ckpt, args.checkpoint_dir / "best_wer.pt")
            print(f"  -> new best val_wer={best_val_wer:.3f}, saved best_wer.pt")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(f"  -> No improvement in WER. Early stopping counter: {epochs_without_improvement}/{early_stopping_patience}")

        # Trigger Early Stopping
        if epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch} epochs due to no improvement in Validation WER.")
            break

if __name__ == "__main__":
    main()