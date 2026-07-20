"""
Baseline CTC training pipeline for continuous gloss recognition,
fusing per-timestep VideoMAE features (from videomae_encode_seq.py)
with keypoint sequences, against ordered gloss_bag labels.

This is a starting point, not a tuned final model -- get it running
end-to-end first, then iterate on architecture once you trust the
data pipeline.

Usage:
    python train_ctc_baseline.py \
        --records surdobot_dataset_records.json \
        --vocab surdobot_vocab.json \
        --videomae-seq-dir /path/to/Videomae_features_seq \
        --keypoint-dir /path/to/surdobot_raw_hs_keypoints \
        --epochs 10 --batch-size 8 --device cuda
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


# ---------------------------------------------------------------------------
# Vocab
# ---------------------------------------------------------------------------

def load_vocab(vocab_path):
    with open(vocab_path) as f:
        vocab = json.load(f)
    # tolerate either {"token2idx": {...}} or a flat {token: idx} dict
    token2idx = vocab.get("token2idx", vocab) if isinstance(vocab, dict) else vocab
    if "<blank>" not in token2idx:
        raise ValueError("vocab must contain a '<blank>' token for CTC")
    blank_idx = token2idx["<blank>"]
    return token2idx, blank_idx


# ---------------------------------------------------------------------------
# Frame-index reconstruction, matching videomae_encode_seq.py's windowing
# ---------------------------------------------------------------------------

def timestep_frame_indices(num_timesteps, window_frames=16, tubelet_size=2):
    """Recreate the absolute source-frame index each output timestep
    corresponds to, given only the output sequence length. Matches the
    windowing scheme in videomae_encode_seq.py exactly (non-overlapping
    16-frame windows, tubelet_size=2 -> 8 temporal tokens/window)."""
    tokens_per_window = window_frames // tubelet_size
    indices = np.zeros(num_timesteps, dtype=np.int64)
    for i in range(num_timesteps):
        w = i // tokens_per_window
        t = i % tokens_per_window
        indices[i] = w * window_frames + t * tubelet_size + tubelet_size // 2
    return indices


def align_keypoints_to_sampled_frames(npz_path, sampled_frame_indices):
    """Same logic as in keypoint_preprocess.py -- duplicated here to keep
    this script standalone. Nearest-neighbor match on absolute frame
    number."""
    data = np.load(npz_path)
    kp = data["keypoints"]
    frame_numbers = data["frame_numbers"]
    valid = data["hand_valid"]

    out_kp = np.zeros((len(sampled_frame_indices), 2, 21, 2), dtype=np.float32)
    out_valid = np.zeros((len(sampled_frame_indices), 2), dtype=bool)
    if len(frame_numbers) == 0:
        return out_kp, out_valid

    for i, target_frame in enumerate(sampled_frame_indices):
        nearest_idx = int(np.argmin(np.abs(frame_numbers - target_frame)))
        out_kp[i] = kp[nearest_idx]
        out_valid[i] = valid[nearest_idx]
    return out_kp, out_valid


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SurdobotCTCDataset(Dataset):
    def __init__(self, records, videomae_seq_dir, keypoint_dir, token2idx):
        self.records = records
        self.videomae_seq_dir = Path(videomae_seq_dir)
        self.keypoint_dir = Path(keypoint_dir)
        self.token2idx = token2idx

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        clip_id = rec["clip_id"]

        vmae_path = self.videomae_seq_dir / f"{clip_id}.npy"
        vmae_seq = np.load(vmae_path)  # (T, 768)
        T = vmae_seq.shape[0]

        vid_id = Path(clip_id).stem  # matches the corrected keypoint mapping
        kp_path = self.keypoint_dir / f"{vid_id}.npz"
        frame_idx = timestep_frame_indices(T)
        if kp_path.exists():
            kp_seq, kp_valid = align_keypoints_to_sampled_frames(kp_path, frame_idx)
            kp_seq = kp_seq.reshape(T, -1)  # (T, 2*21*2=84)
        else:
            kp_seq = np.zeros((T, 84), dtype=np.float32)

        fused = np.concatenate([vmae_seq, kp_seq], axis=1)  # (T, 768+84)

        gloss_bag = rec["gloss_bag"]  # ordered list of gloss strings
        label_ids = [self.token2idx.get(g, self.token2idx.get("<unk>")) for g in gloss_bag]
        label_ids = [x for x in label_ids if x is not None]

        return {
            "features": torch.from_numpy(fused).float(),
            "input_length": T,
            "labels": torch.tensor(label_ids, dtype=torch.long),
            "label_length": len(label_ids),
            "clip_id": clip_id,
        }


def collate_fn(batch):
    # CTC feasibility check -- drop samples that can't satisfy CTC's
    # length constraint rather than let them silently produce inf loss.
    batch = [b for b in batch if b["input_length"] > 2 * b["label_length"] + 1]

    features = pad_sequence([b["features"] for b in batch], batch_first=True)  # (B, T_max, F)
    input_lengths = torch.tensor([b["input_length"] for b in batch], dtype=torch.long)
    labels = torch.cat([b["labels"] for b in batch])
    label_lengths = torch.tensor([b["label_length"] for b in batch], dtype=torch.long)
    clip_ids = [b["clip_id"] for b in batch]
    return features, input_lengths, labels, label_lengths, clip_ids


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CTCGlossModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, vocab_size, num_layers=2):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers,
                             batch_first=True, bidirectional=True, dropout=0.2)
        self.classifier = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, x):
        x = torch.relu(self.proj(x))
        x, _ = self.lstm(x)
        logits = self.classifier(x)
        return torch.log_softmax(logits, dim=-1)  # (B, T, vocab_size)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def load_records(path):
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("records", list(data.values()))
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True, type=Path)
    ap.add_argument("--vocab", required=True, type=Path)
    ap.add_argument("--videomae-seq-dir", required=True, type=Path)
    ap.add_argument("--keypoint-dir", required=True, type=Path)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None, help="Debug: only use first N records")
    ap.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    args = ap.parse_args()

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    token2idx, blank_idx = load_vocab(args.vocab)
    vocab_size = len(token2idx)
    print(f"Vocab size: {vocab_size}, blank_idx: {blank_idx}")

    records = load_records(args.records)
    # only keep clips that actually have a VideoMAE sequence file on disk
    records = [r for r in records if (args.videomae_seq_dir / f"{r['clip_id']}.npy").exists()]
    if args.limit:
        records = records[: args.limit]
    print(f"Usable records: {len(records)}")

    train_records = [r for r in records if r.get("split") == "train"]
    val_records = [r for r in records if r.get("split") == "val"]
    print(f"Train: {len(train_records)}, Val: {len(val_records)}")

    train_ds = SurdobotCTCDataset(train_records, args.videomae_seq_dir, args.keypoint_dir, token2idx)
    val_ds = SurdobotCTCDataset(val_records, args.videomae_seq_dir, args.keypoint_dir, token2idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    input_dim = 768 + 84  # videomae hidden dim + flattened keypoints
    model = CTCGlossModel(input_dim, args.hidden_dim, vocab_size).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    ctc_loss = nn.CTCLoss(blank=blank_idx, zero_infinity=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss, n_batches = 0.0, 0
        for features, input_lengths, labels, label_lengths, _ in train_loader:
            if features.shape[0] == 0:
                continue  # entire batch got filtered by the CTC feasibility check
            features = features.to(args.device)
            labels = labels.to(args.device)

            log_probs = model(features).transpose(0, 1)  # (T, B, vocab_size) for CTCLoss
            loss = ctc_loss(log_probs, labels, input_lengths, label_lengths)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_train_loss = total_loss / max(n_batches, 1)

        model.eval()
        val_loss, val_batches = 0.0, 0
        with torch.no_grad():
            for features, input_lengths, labels, label_lengths, _ in val_loader:
                if features.shape[0] == 0:
                    continue
                features = features.to(args.device)
                labels = labels.to(args.device)
                log_probs = model(features).transpose(0, 1)
                loss = ctc_loss(log_probs, labels, input_lengths, label_lengths)
                val_loss += loss.item()
                val_batches += 1
        avg_val_loss = val_loss / max(val_batches, 1)

        print(f"Epoch {epoch+1}/{args.epochs}  train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f}")
        torch.save(model.state_dict(), args.checkpoint_dir / f"epoch_{epoch+1}.pt")


if __name__ == "__main__":
    main()
