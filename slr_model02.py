"""
Fused Video + Keypoint model for continuous gloss recognition
(CTC + MIL dual head), per the blueprint's Phase 3 architecture.

Deviations from the blueprint, and why:
  - VideoMAE is used as a FROZEN offline feature extractor (already
    computed via videomae_encode_seq.py), not fine-tuned end-to-end.
    Far cheaper; revisit fine-tuning later if needed.
  - Text/transcript (XLM-R) branch is stubbed as an interface only --
    not implemented, since ASR/lag-modeling work hasn't started yet.
    The fusion layer is modality-count-agnostic, so plugging in a third
    stream later doesn't require restructuring this module.
  - The "local window=64 frames" Temporal Transformer is approximated
    with a sliding local-attention mask rather than a custom kernel.
    Functionally equivalent, just implemented via a mask on standard
    nn.TransformerEncoderLayer.

Modality dropout is trained in from the start -- at train time, each
modality is independently dropped (replaced with a learned "missing"
embedding) with probability `modality_dropout_p`. This is exactly what
lets a model trained here run on Elarna clips that have video but no
keypoints, without retraining.

Shapes follow the blueprint's own notation:
  V ∈ R^{T_v x 768}   -- precomputed VideoMAE features (from videomae_encode_seq.py)
  K ∈ R^{T_k x 42 x C} -- keypoint sequences, 42 = 21 landmarks x 2 hands,
                          C = 2 (x,y) + optional velocity features
  F_fused ∈ R^{T x 1024} -- fused, temporally-aligned representation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Keypoint encoder: ST-GCN over the MediaPipe 21-point hand topology
# ---------------------------------------------------------------------------

# Standard MediaPipe hand landmark skeletal edges (wrist=0 is the root).
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (0, 9), (9, 10), (10, 11), (11, 12),     # middle
    (0, 13), (13, 14), (14, 15), (15, 16),   # ring
    (0, 17), (17, 18), (18, 19), (19, 20),   # pinky
    (5, 9), (9, 13), (13, 17),               # palm cross-links
]


def build_two_hand_adjacency() -> torch.Tensor:
    """42x42 adjacency for two independent 21-node hand graphs
    (block-diagonal -- no edges between left/right hand nodes, since
    they aren't physically connected)."""
    n = 42
    adj = torch.zeros(n, n)
    for offset in (0, 21):
        for a, b in HAND_EDGES:
            i, j = a + offset, b + offset
            adj[i, j] = 1.0
            adj[j, i] = 1.0
        for i in range(21):
            adj[i + offset, i + offset] = 1.0  # self-loops
    # symmetric normalization (D^-1/2 A D^-1/2)
    deg = adj.sum(dim=1)
    deg_inv_sqrt = torch.pow(deg.clamp(min=1e-6), -0.5)
    return deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)


class STGCNBlock(nn.Module):
    """One spatial graph conv + temporal conv block."""

    def __init__(self, in_channels, out_channels, adjacency: torch.Tensor, temporal_kernel=9):
        super().__init__()
        self.register_buffer("adj", adjacency)
        self.spatial_conv = nn.Linear(in_channels, out_channels)
        self.temporal_conv = nn.Conv1d(
            out_channels, out_channels, kernel_size=temporal_kernel,
            padding=temporal_kernel // 2,
        )
        self.norm = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU()
        self.residual = (
            nn.Identity() if in_channels == out_channels
            else nn.Linear(in_channels, out_channels)
        )

    def forward(self, x):
        # x: (B, T, N, C_in)
        res = self.residual(x)
        # spatial graph conv: aggregate over neighbor nodes per the adjacency
        x = torch.einsum("btnc,nm->btmc", x, self.adj)  # (B, T, N, C_in)
        x = self.spatial_conv(x)  # (B, T, N, C_out)
        B, T, N, C = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B * N, C, T)  # (B*N, C_out, T)
        x = self.temporal_conv(x)
        x = self.norm(x)
        x = x.reshape(B, N, C, T).permute(0, 3, 1, 2)  # (B, T, N, C_out)
        return self.act(x + res)


class KeypointEncoder(nn.Module):
    """T x 42 x C keypoint sequence -> T x d_keypoint.

    Includes a velocity feature (frame-to-frame delta) per the
    blueprint's note that fingertip velocity is the strongest
    discriminative signal for handshape transitions.
    """

    def __init__(self, in_channels=2, hidden=64, out_dim=256, num_blocks=3):
        super().__init__()
        adj = build_two_hand_adjacency()
        # in_channels * 2 because we concat position + velocity
        blocks = []
        c_in = in_channels * 2
        for i in range(num_blocks):
            c_out = hidden if i < num_blocks - 1 else out_dim
            blocks.append(STGCNBlock(c_in, c_out, adj))
            c_in = c_out
        self.blocks = nn.ModuleList(blocks)
        self.out_dim = out_dim

    def forward(self, keypoints, valid_mask=None):
        # keypoints: (B, T, 42, C)   [42 = 21 landmarks x 2 hands]
        vel = torch.zeros_like(keypoints)
        vel[:, 1:] = keypoints[:, 1:] - keypoints[:, :-1]
        x = torch.cat([keypoints, vel], dim=-1)  # (B, T, 42, 2C)
        for block in self.blocks:
            x = block(x)
        # pool over nodes -> per-frame vector
        node_feat = x.mean(dim=2)  # (B, T, out_dim)
        if valid_mask is not None:
            # zero out frames where neither hand was detected
            node_feat = node_feat * valid_mask.any(dim=-1, keepdim=True).float()
        return node_feat


# ---------------------------------------------------------------------------
# Cross-modal fusion with modality dropout
# ---------------------------------------------------------------------------

class ModalityDropout(nn.Module):
    """Randomly replaces a modality's features with a learned 'missing'
    embedding during training, so the model learns to operate on any
    subset of modalities at inference (critical for Elarna clips that
    lack keypoints)."""

    def __init__(self, dim, p=0.15):
        super().__init__()
        self.p = p
        self.missing_embed = nn.Parameter(torch.randn(dim) * 0.02)

    def forward(self, x, force_drop=False):
        # x: (B, T, dim)
        if force_drop:
            return self.missing_embed.expand_as(x)
        if self.training and torch.rand(1).item() < self.p:
            return self.missing_embed.expand_as(x)
        return x


class CrossModalFusion(nn.Module):
    """Projects each modality to d_model, applies cross-attention between
    video and keypoint streams, then interpolates the result onto a
    single canonical time axis (the video stream's length), matching the
    blueprint's 'output at video framerate, interpolated' spec."""

    def __init__(self, video_dim=768, keypoint_dim=256, d_model=1024, n_heads=8):
        super().__init__()
        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads}) -- "
            f"nn.MultiheadAttention requires an equal, whole head_dim per head. "
            f"(Note: the blueprint's spec of d_model=1024/n_heads=6 doesn't divide "
            f"evenly -- 8 heads at head_dim=128 is used here instead.)"
        )
        self.video_proj = nn.Linear(video_dim, d_model)
        self.keypoint_proj = nn.Linear(keypoint_dim, d_model)
        self.video_dropout = ModalityDropout(d_model)
        self.keypoint_dropout = ModalityDropout(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, 4096, d_model) * 0.02)  # max length cap

    def forward(self, video_feat, keypoint_feat, video_missing=False, keypoint_missing=False):
        # video_feat: (B, T_v, video_dim), keypoint_feat: (B, T_k, keypoint_dim)
        v = self.video_proj(video_feat)
        k = self.keypoint_proj(keypoint_feat)
        v = self.video_dropout(v, force_drop=video_missing)
        k = self.keypoint_dropout(k, force_drop=keypoint_missing)

        # keypoints attend to video, then interpolate keypoint-informed
        # context back onto the video's own time axis (our canonical grid)
        attended, _ = self.cross_attn(query=v, key=k, value=k)
        fused = self.norm(v + attended)

        T = fused.shape[1]
        fused = fused + self.pos_embed[:, :T, :]
        return fused  # (B, T_v, d_model)


# ---------------------------------------------------------------------------
# Temporal modeling stack: multi-scale (micro/meso/macro), per blueprint 3B
# ---------------------------------------------------------------------------

class DilatedSTConvBlock(nn.Module):
    def __init__(self, dim, kernel=7, dilations=(1, 2, 4)):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(dim, dim, kernel_size=kernel, padding=(kernel - 1) * d // 2, dilation=d)
            for d in dilations
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: (B, T, dim)
        res = x
        h = x.transpose(1, 2)  # (B, dim, T)
        for conv in self.convs:
            h = F.gelu(conv(h))
        h = h.transpose(1, 2)  # (B, T, dim)
        return self.norm(h + res)


def local_window_mask(T, window=64, device=None):
    """Additive attention mask restricting each position to attend only
    within +-window/2 frames (approximates the blueprint's local-window
    Temporal Transformer)."""
    idx = torch.arange(T, device=device)
    dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
    mask = torch.zeros(T, T, device=device)
    mask[dist > window // 2] = float("-inf")
    return mask


class TemporalStack(nn.Module):
    def __init__(self, dim=1024, n_stconv=3, lstm_hidden=512, n_transformer_layers=4,
                 local_window=64, n_heads=8, dropout=0.2):
        super().__init__()
        self.stconv_blocks = nn.ModuleList([DilatedSTConvBlock(dim) for _ in range(n_stconv)])
        self.bilstm = nn.LSTM(dim, lstm_hidden, num_layers=2, batch_first=True,
                               bidirectional=True, dropout=dropout)
        self.lstm_proj = nn.Linear(lstm_hidden * 2, dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=dim * 4,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_transformer_layers)
        self.local_window = local_window

    def forward(self, x):
        # x: (B, T, dim)
        for block in self.stconv_blocks:
            x = block(x)
        lstm_out, _ = self.bilstm(x)
        x = x + self.lstm_proj(lstm_out)
        mask = local_window_mask(x.shape[1], self.local_window, device=x.device)
        x = self.transformer(x, mask=mask)
        return x  # (B, T, dim)


# ---------------------------------------------------------------------------
# Dual head: CTC + MIL
# ---------------------------------------------------------------------------

class DualHead(nn.Module):
    def __init__(self, dim, vocab_size):
        super().__init__()
        self.ctc_head = nn.Linear(dim, vocab_size + 1)  # +1 for CTC blank

    def forward(self, x):
        # x: (B, T, dim)
        ctc_logits = self.ctc_head(x)                 # (B, T, vocab+1)
        gloss_logits = ctc_logits[..., :-1]            # (B, T, vocab) -- exclude blank
        # MIL bag score: max-pool RAW LOGITS over time (matches blueprint
        # spec: "max-pool over T -> P(g in clip)"). Max-pooling in logit
        # space is unbounded/safe (no probability-overshoot risk) and
        # pairs with BCEWithLogitsLoss, which is autocast-safe -- unlike
        # an earlier log-sum-exp-over-probabilities version, which could
        # exceed 1.0 and silently kill gradients when clamped.
        mil_logits = gloss_logits.max(dim=1).values     # (B, vocab)
        return ctc_logits, mil_logits


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class SLRModel(nn.Module):
    def __init__(self, vocab_size, video_dim=768, keypoint_dim=256, d_model=1024,
                 modality_dropout_p=0.15):
        super().__init__()
        self.keypoint_encoder = KeypointEncoder(in_channels=2, out_dim=keypoint_dim)
        self.fusion = CrossModalFusion(video_dim, keypoint_dim, d_model)
        self.fusion.video_dropout.p = modality_dropout_p
        self.fusion.keypoint_dropout.p = modality_dropout_p
        self.temporal_stack = TemporalStack(dim=d_model)
        self.head = DualHead(d_model, vocab_size)
        # TODO (deferred until ASR/Elarna work starts): a third modality
        # branch for transcript embeddings (XLM-R, frozen) would project
        # into the same d_model and enter CrossModalFusion as an
        # additional cross-attention source. Not implemented yet --
        # CrossModalFusion currently only fuses video + keypoints.

    def forward(self, video_feat, keypoints, video_missing=False, keypoint_missing=False):
        """
        video_feat: (B, T_v, 768)      -- from videomae_encode_seq.py output
        keypoints:  (B, T_k, 42, 2)    -- from keypoint_preprocess.py .npz output
                                           (concat left/right hands -> 42 nodes)
        """
        keypoint_feat = self.keypoint_encoder(keypoints)
        fused = self.fusion(video_feat, keypoint_feat, video_missing, keypoint_missing)
        temporal = self.temporal_stack(fused)
        ctc_logits, mil_scores = self.head(temporal)
        return ctc_logits, mil_scores


if __name__ == "__main__":
    # smoke test with dummy shapes
    B, T_v, T_k, vocab = 2, 408, 300, 500
    model = SLRModel(vocab_size=vocab)
    video_feat = torch.randn(B, T_v, 768)
    keypoints = torch.randn(B, T_k, 42, 2)
    ctc_logits, mil_scores = model(video_feat, keypoints)
    print("ctc_logits:", ctc_logits.shape)   # (B, T_v, vocab+1)
    print("mil_scores:", mil_scores.shape)   # (B, vocab)

    # modality dropout smoke test (keypoints missing, e.g. Elarna clip)
    ctc_logits2, mil_scores2 = model(video_feat, keypoints, keypoint_missing=True)
    print("with keypoints missing:", ctc_logits2.shape, mil_scores2.shape)
