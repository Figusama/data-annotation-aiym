"""
Dataset for the fused SLR model, built to work with PARTIAL encoding
progress -- it filters down to only clips where BOTH a VideoMAE feature
file and a keypoint file actually exist on disk right now, so you can
start running/training on however many clips are done so far (e.g.
~12,000 out of 35,301) without waiting for the rest.

Reuses the same video_id-from-clip_id mapping and collision exclusion
logic as check_coverage.py -- keep these in sync if that logic changes.
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def video_id_from_clip_id(clip_id: str) -> str:
    # matches keypoint_preprocess.py's actual on-disk naming (confirmed
    # empirically -- see diagnose_keypoint_mapping.py)
    return Path(clip_id).stem


def load_records(records_path):
    with open(records_path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("records", list(data.values()))
    return data


def load_vocab(vocab_path):
    """surdobot_vocab.json's real structure (confirmed on-disk): a wrapper
    dict with 'gloss2idx', 'idx2gloss', and 'vocab_size' keys already
    built -- just read them directly rather than deriving stoi/itos."""
    with open(vocab_path) as f:
        vocab = json.load(f)
    if isinstance(vocab, dict) and "gloss2idx" in vocab:
        stoi = vocab["gloss2idx"]
        # idx2gloss keys may be strings (JSON object keys are always
        # strings) -- normalize to int keys for lookup by index
        itos = {int(k): v for k, v in vocab["idx2gloss"].items()}
        return stoi, itos
    elif isinstance(vocab, dict):
        stoi = vocab
    elif isinstance(vocab, list):
        stoi = {g: i for i, g in enumerate(vocab)}
    else:
        raise ValueError(f"Unexpected vocab.json structure: {type(vocab)}")
    itos = {i: g for g, i in stoi.items()}
    return stoi, itos


def find_colliding_clip_ids(records):
    """Same collision check as check_coverage.py -- clip_ids whose
    derived video_id collides with another clip_id's, making the
    keypoint file mapping ambiguous/unsafe."""
    stem_to_clip_ids = {}
    for rec in records:
        cid = rec.get("clip_id")
        if not cid:
            continue
        stem = video_id_from_clip_id(cid)
        stem_to_clip_ids.setdefault(stem, []).append(cid)
    colliding = set()
    for cids in stem_to_clip_ids.values():
        if len(set(cids)) > 1:
            colliding.update(cids)
    return colliding


def resplit_by_session(records, video_feat_dir, keypoint_dir, colliding,
                        val_fraction=0.1, seed=0, require_keypoints=True):
    """Temporary re-split of only the CURRENTLY USABLE records (have
    encoded video + keypoints on disk right now), grouped by source
    session -- NOT by clip_id -- so that clips from the same recording
    session/signer never split across train/val. Use this while the
    real records.json 'split' assignment hasn't fully finished encoding
    yet; switch back to the real split once val clips are done.

    Session key = the top-level folder in clip_id (e.g. "27_10" in
    "27_10/resized_cropped_....mp4") -- adjust if your session grouping
    convention differs.
    """
    import random
    from pathlib import Path as _Path

    usable = []
    for rec in records:
        clip_id = rec.get("clip_id")
        gloss_bag = rec.get("gloss_bag")
        if not clip_id or not gloss_bag or clip_id in colliding:
            continue
        if not (video_feat_dir / f"{clip_id}.npy").exists():
            continue
        if require_keypoints and not (keypoint_dir / f"{video_id_from_clip_id(clip_id)}.npz").exists():
            continue
        usable.append(rec)

    sessions = {}
    for rec in usable:
        session = _Path(rec["clip_id"]).parts[0]
        sessions.setdefault(session, []).append(rec)

    session_ids = sorted(sessions.keys())
    random.Random(seed).shuffle(session_ids)
    n_val_sessions = max(1, int(len(session_ids) * val_fraction))
    val_sessions = set(session_ids[:n_val_sessions])

    train_records = [r for s in session_ids if s not in val_sessions for r in sessions[s]]
    val_records = [r for s in val_sessions for r in sessions[s]]

    print(f"resplit_by_session: {len(session_ids)} sessions -> "
          f"{len(session_ids) - n_val_sessions} train sessions ({len(train_records)} clips), "
          f"{n_val_sessions} val sessions ({len(val_records)} clips)")
    return train_records, val_records


class SurdobotFusionDataset(Dataset):
    @classmethod
    def from_records(cls, records, video_feat_dir, keypoint_dir, vocab_path):
        """Build a dataset directly from an already-filtered list of
        records (e.g. from resplit_by_session), skipping the usual
        file-existence filtering since the caller already did it."""
        obj = cls.__new__(cls)
        obj.video_feat_dir = Path(video_feat_dir)
        obj.keypoint_dir = Path(keypoint_dir)
        obj.stoi, obj.itos = load_vocab(vocab_path)
        obj.records = records
        print(f"SurdobotFusionDataset.from_records: {len(records)} records")
        return obj

    def __init__(self, records_path, video_feat_dir, keypoint_dir, vocab_path,
                 require_keypoints=True, split=None):
        self.video_feat_dir = Path(video_feat_dir)
        self.keypoint_dir = Path(keypoint_dir)
        self.stoi, self.itos = load_vocab(vocab_path)

        all_records = load_records(records_path)
        colliding = find_colliding_clip_ids(all_records)

        usable = []
        for rec in all_records:
            clip_id = rec.get("clip_id")
            gloss_bag = rec.get("gloss_bag")
            if not clip_id or not gloss_bag:
                continue
            if split is not None and rec.get("split") != split:
                continue
            if clip_id in colliding:
                continue
            vmae_path = self.video_feat_dir / f"{clip_id}.npy"
            if not vmae_path.exists():
                continue  # not encoded yet -- this is what makes partial progress OK
            if require_keypoints:
                kp_path = self.keypoint_dir / f"{video_id_from_clip_id(clip_id)}.npz"
                if not kp_path.exists():
                    continue
            usable.append(rec)

        self.records = usable
        split_label = f" split={split!r}" if split else ""
        print(f"SurdobotFusionDataset{split_label}: {len(usable)}/{len(all_records)} records usable "
              f"(have video features{' + keypoints' if require_keypoints else ''}, "
              f"not a collision, non-empty gloss_bag)")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        clip_id = rec["clip_id"]

        video_feat = np.load(self.video_feat_dir / f"{clip_id}.npy")  # (T_v, 768)

        kp_path = self.keypoint_dir / f"{video_id_from_clip_id(clip_id)}.npz"
        if kp_path.exists():
            kp_data = np.load(kp_path)
            keypoints = kp_data["keypoints"]  # (T_k, 2, 21, 2)
            keypoints = keypoints.reshape(keypoints.shape[0], 42, 2)  # (T_k, 42, 2)
            keypoint_missing = False
        else:
            # allow running video-only (e.g. mirrors the Elarna scenario)
            keypoints = np.zeros((1, 42, 2), dtype=np.float32)
            keypoint_missing = True

        gloss_bag = rec["gloss_bag"]  # ordered list of gloss strings
        target = [self.stoi[g] for g in gloss_bag if g in self.stoi]

        return {
            "clip_id": clip_id,
            "video_feat": torch.from_numpy(video_feat).float(),
            "keypoints": torch.from_numpy(keypoints).float(),
            "keypoint_missing": keypoint_missing,
            "target": torch.tensor(target, dtype=torch.long),
        }


def collate_fn(batch):
    """Pads video/keypoint sequences to the batch max length and returns
    explicit lengths, required for CTC loss."""
    video_lens = [b["video_feat"].shape[0] for b in batch]
    kp_lens = [b["keypoints"].shape[0] for b in batch]
    target_lens = [b["target"].shape[0] for b in batch]

    T_v_max = max(video_lens)
    T_k_max = max(kp_lens)

    video_dim = batch[0]["video_feat"].shape[1]
    video_padded = torch.zeros(len(batch), T_v_max, video_dim)
    kp_padded = torch.zeros(len(batch), T_k_max, 42, 2)
    keypoint_missing = torch.zeros(len(batch), dtype=torch.bool)
    targets = torch.cat([b["target"] for b in batch]) if sum(target_lens) > 0 else torch.zeros(0, dtype=torch.long)

    for i, b in enumerate(batch):
        video_padded[i, :video_lens[i]] = b["video_feat"]
        kp_padded[i, :kp_lens[i]] = b["keypoints"]
        keypoint_missing[i] = b["keypoint_missing"]

    return {
        "clip_ids": [b["clip_id"] for b in batch],
        "video_feat": video_padded,
        "video_lengths": torch.tensor(video_lens, dtype=torch.long),
        "keypoints": kp_padded,
        "keypoint_lengths": torch.tensor(kp_lens, dtype=torch.long),
        "keypoint_missing": keypoint_missing,
        "targets": targets,  # concatenated, per CTCLoss's expected format
        "target_lengths": torch.tensor(target_lens, dtype=torch.long),
    }


if __name__ == "__main__":
    import argparse
    from slr_model import SLRModel

    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--video-feat-dir", required=True)
    ap.add_argument("--keypoint-dir", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--require-keypoints", action="store_true",
                     help="Only use clips that have BOTH video and keypoint features. "
                          "Omit to run video-only for clips missing keypoints "
                          "(exercises modality dropout's missing-keypoint path).")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    args = ap.parse_args()

    device = args.device if (args.device == "cpu" or torch.cuda.is_available() or args.device == "mps") else "cpu"
    if device != args.device:
        print(f"Requested device {args.device!r} unavailable, falling back to cpu")
    print(f"Using device: {device}")

    dataset = SurdobotFusionDataset(
        args.records, args.video_feat_dir, args.keypoint_dir, args.vocab,
        require_keypoints=args.require_keypoints,
    )
    if len(dataset) == 0:
        raise SystemExit("No usable clips found -- check paths and encoding progress.")

    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    batch = next(iter(loader))

    print("\nBatch shapes:")
    print("  video_feat:", batch["video_feat"].shape)
    print("  video_lengths:", batch["video_lengths"])
    print("  keypoints:", batch["keypoints"].shape)
    print("  keypoint_missing:", batch["keypoint_missing"])
    print("  targets:", batch["targets"].shape, "target_lengths:", batch["target_lengths"])

    vocab_size = len(dataset.stoi)
    model = SLRModel(vocab_size=vocab_size).to(device)
    model.eval()
    video_feat = batch["video_feat"].to(device)
    keypoints = batch["keypoints"].to(device)
    with torch.no_grad():
        ctc_logits, mil_scores = model(video_feat, keypoints)
    print("\nModel output shapes (real data):")
    print("  ctc_logits:", ctc_logits.shape)
    print("  mil_scores:", mil_scores.shape)

    # sanity-check CTC loss actually computes without error on real data
    import torch.nn as nn
    ctc_loss_fn = nn.CTCLoss(blank=vocab_size, zero_infinity=True)
    log_probs = ctc_logits.log_softmax(dim=-1).transpose(0, 1)  # (T, B, vocab+1) for CTCLoss
    loss = ctc_loss_fn(
        log_probs.cpu(),  # CTCLoss on CPU keeps this simple across cuda/mps/cpu
        batch["targets"],
        batch["video_lengths"],
        batch["target_lengths"],
    )
    print("\nCTC loss on real batch:", loss.item())
