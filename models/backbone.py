"""
╔══════════════════════════════════════════════════════════════════════════╗
║         G.O.D.S E.Y.E — Shared AI Backbone                             ║
║         Project : MultiStockLSTMBot                                      ║
║         File    : models/backbone.py                                     ║
║         Phase   : 2 — AI Backbone (Supervised Pre-training)             ║
║                                                                          ║
║  What this module does:                                                  ║
║    Fuses the LSTMEncoder (temporal memory) and TransformerEncoder        ║
║    (cross-stock attention) into a single unified 128-dimensional         ║
║    shared embedding per stock per day.                                   ║
║                                                                          ║
║    This shared embedding is the single most important artifact in the    ║
║    entire system. It is:                                                 ║
║      • Pre-trained here (Phase 2) on supervised return prediction        ║
║      • Frozen / fine-tuned in Phase 3 as input to both RL heads          ║
║      • The representation both the Swing and Intraday agents use to      ║
║        make buy/hold/sell decisions                                       ║
║                                                                          ║
║  Full data flow:                                                         ║
║                                                                          ║
║    Input: (N_stocks, seq=60, features=28) from features/fusion.py        ║
║        │                                                                  ║
║        ├── Per-stock: LSTMEncoder → (N_stocks, 64)   [temporal memory]  ║
║        │                                                                  ║
║        ├── Stack all stocks → (1, N_stocks, 64)                          ║
║        │                                                                  ║
║        ├── TransformerEncoder → (1, N_stocks, 128)   [cross-stock attn] ║
║        │                                                                  ║
║        ├── Squeeze batch dim → (N_stocks, 128)                           ║
║        │                                                                  ║
║        ├── Concat with LSTM output → (N_stocks, 192)                    ║
║        │                                                                  ║
║        └── FusionHead (Linear→GELU→LN) → (N_stocks, 128)  ← OUTPUT     ║
║                                                                          ║
║  Supervised pre-training targets (Phase 2):                              ║
║    • direction_logit : binary classification (up=1 / down=0)            ║
║      from a PredictionHead(128 → 1) with BCE loss                       ║
║    • return_pred     : regression (5-day forward return magnitude)       ║
║      from a PredictionHead(128 → 1) with MSE loss                       ║
║    • Combined loss   : 0.6 × BCE + 0.4 × MSE                            ║
║                                                                          ║
║  Checkpoint behaviour:                                                   ║
║    • save_checkpoint() saves full model state + training metadata        ║
║    • load_checkpoint() restores model and optionally freezes backbone    ║
║    • freeze_backbone() / unfreeze_backbone() for Phase 3 RL transfer    ║
║                                                                          ║
║  Dependencies:                                                           ║
║    models/lstm_encoder.py, models/transformer_encoder.py                ║
║    pip install torch>=2.1.0                                              ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from typing  import Dict, Optional, Tuple

from models.lstm_encoder        import LSTMEncoder,        build_lstm_encoder
from models.transformer_encoder import TransformerEncoder, build_transformer_encoder


# ── Constants ─────────────────────────────────────────────────────────────
LSTM_OUT_DIM        = 64     # LSTMEncoder output dim
TRANSFORMER_OUT_DIM = 128    # TransformerEncoder output dim
CONCAT_DIM          = LSTM_OUT_DIM + TRANSFORMER_OUT_DIM   # 192
BACKBONE_OUT_DIM    = 128    # Final shared embedding dim
CHECKPOINT_DIR      = Path("checkpoints")


# ══════════════════════════════════════════════════════════════════════════
#  FUSION HEAD
#  Compresses [LSTM_out || Transformer_out] → 128d shared embedding
# ══════════════════════════════════════════════════════════════════════════

class FusionHead(nn.Module):
    """
    Projects concatenated [LSTM(64) || Transformer(128)] = 192d
    down to the 128-dimensional shared embedding.

    Architecture:
        Linear(192 → 256) → GELU → Dropout(0.2) → Linear(256 → 128) → LayerNorm

    The two-layer MLP allows non-linear interaction between the temporal
    (LSTM) and cross-stock (Transformer) representations before they are
    compressed into the final embedding.

    Args:
        input_dim  : Concatenated input dim (LSTM + Transformer = 192)
        output_dim : Final embedding dim (default: 128)
        dropout    : Dropout rate (default: 0.2)
    """

    def __init__(
        self,
        input_dim : int   = CONCAT_DIM,
        output_dim: int   = BACKBONE_OUT_DIM,
        dropout   : float = 0.2,
    ):
        super().__init__()

        hidden_dim = max(input_dim, output_dim)   # 256 for default config

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

        # Xavier init
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (N_stocks, 192) — concatenated LSTM + Transformer outputs

        Returns:
            (N_stocks, 128) — shared embedding
        """
        return self.net(x)


# ══════════════════════════════════════════════════════════════════════════
#  PREDICTION HEADS (for supervised pre-training only)
#  Removed / detached after Phase 2. Phase 3 RL heads replace these.
# ══════════════════════════════════════════════════════════════════════════

class DirectionHead(nn.Module):
    """
    Binary classification head: predicts whether the stock goes UP or DOWN
    over the next 5 trading days.

    Used only during Phase 2 supervised pre-training.
    Discarded after pre-training — Phase 3 RL heads take over.

    Input  : (batch, 128) — shared embedding
    Output : (batch, 1)   — logit (pass through sigmoid for probability)
    """

    def __init__(self, input_dim: int = BACKBONE_OUT_DIM):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logit. Apply sigmoid externally for BCE loss."""
        return self.head(x).squeeze(-1)   # (batch,)


class ReturnHead(nn.Module):
    """
    Regression head: predicts the 5-day forward return magnitude.

    Used only during Phase 2 supervised pre-training.
    Discarded after pre-training — Phase 3 RL heads take over.

    Input  : (batch, 128) — shared embedding
    Output : (batch, 1)   — predicted return (unbounded, in % units)
    """

    def __init__(self, input_dim: int = BACKBONE_OUT_DIM):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns predicted return in % (no activation — unbounded)."""
        return self.head(x).squeeze(-1)   # (batch,)


# ══════════════════════════════════════════════════════════════════════════
#  GODSEYE BACKBONE
# ══════════════════════════════════════════════════════════════════════════

class GodsEyeBackbone(nn.Module):
    """
    The shared AI backbone of G.O.D.S E.Y.E.

    Combines LSTMEncoder (temporal) + TransformerEncoder (cross-stock)
    into a unified 128-dimensional representation per stock.

    This module has two operational modes:

    1. PRETRAIN mode (Phase 2):
       - direction_head and return_head are active
       - Loss = 0.6 × BCE(direction) + 0.4 × MSE(return)
       - forward() returns (embedding, direction_logit, return_pred)

    2. INFERENCE mode (Phase 3+):
       - Only the backbone runs (no prediction heads)
       - forward() returns embedding only: (N_stocks, 128)
       - Phase 3 RL heads receive this embedding as input

    Args:
        lstm_config        : Config dict for LSTMEncoder (or None for defaults)
        transformer_config : Config dict for TransformerEncoder (or None)
        backbone_dropout   : Dropout in FusionHead (default: 0.2)
        pretrain_mode      : If True, include direction + return heads

    Example (Phase 2 training):
        backbone = GodsEyeBackbone(pretrain_mode=True)
        x = torch.randn(500, 60, 28)   # all 500 stocks, 60-day window
        emb, dir_logit, ret_pred = backbone(x)

    Example (Phase 3 inference):
        backbone = GodsEyeBackbone.from_checkpoint("checkpoints/best.pt")
        backbone.set_inference_mode()
        x = torch.randn(500, 60, 28)
        emb = backbone(x)   # (500, 128)
    """

    def __init__(
        self,
        lstm_config       : Optional[dict] = None,
        transformer_config: Optional[dict] = None,
        backbone_dropout  : float          = 0.2,
        pretrain_mode     : bool           = True,
    ):
        super().__init__()

        self.pretrain_mode = pretrain_mode

        # ── Core components ───────────────────────────────────────────────
        self.lstm_encoder        = build_lstm_encoder(lstm_config)
        self.transformer_encoder = build_transformer_encoder(transformer_config)
        self.fusion_head         = FusionHead(
            input_dim  = CONCAT_DIM,
            output_dim = BACKBONE_OUT_DIM,
            dropout    = backbone_dropout,
        )

        # ── Supervised pre-training heads (Phase 2 only) ──────────────────
        if pretrain_mode:
            self.direction_head = DirectionHead(BACKBONE_OUT_DIM)
            self.return_head    = ReturnHead(BACKBONE_OUT_DIM)
        else:
            self.direction_head = None
            self.return_head    = None

    # ── Forward pass ──────────────────────────────────────────────────────

    def forward(
        self,
        x            : torch.Tensor,
        stock_indices: Optional[torch.Tensor] = None,
        padding_mask : Optional[torch.Tensor] = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full backbone forward pass.

        Args:
            x             : (N_stocks, seq=60, features=28)
                            OR (batch, N_stocks, seq, features) for batched mode
                            N_stocks stocks, each with 60-day feature sequence
            stock_indices : (N_stocks,) — integer IDs for Transformer pos embedding
            padding_mask  : (1, N_stocks) — True for stocks with missing data

        Returns:
            INFERENCE mode (pretrain_mode=False):
                embedding : (N_stocks, 128) — shared embedding for RL heads

            PRETRAIN mode (pretrain_mode=True):
                embedding     : (N_stocks, 128)
                direction_logit : (N_stocks,) — raw logit for BCE loss
                return_pred     : (N_stocks,) — predicted 5-day return (%)
        """
        # ── Input validation ──────────────────────────────────────────────
        if x.dim() == 3:
            # Standard: (N_stocks, 60, 28)
            n_stocks, seq_len, n_features = x.shape
        elif x.dim() == 4:
            # Batched: (batch, N_stocks, 60, 28) — reshape for processing
            batch, n_stocks, seq_len, n_features = x.shape
            x = x.view(batch * n_stocks, seq_len, n_features)
            n_stocks = batch * n_stocks
        else:
            raise ValueError(
                f"Expected 3D (N_stocks, seq, features) or "
                f"4D (batch, N_stocks, seq, features) input, got {x.dim()}D"
            )

        # ── Step 1: LSTM — encode each stock's 60-day sequence ───────────
        # Process all N_stocks simultaneously (LSTM has no cross-stock info)
        # (N_stocks, 60, 28) → (N_stocks, 64)
        lstm_out = self.lstm_encoder(x)

        # ── Step 2: Transformer — cross-stock attention ───────────────────
        # Add batch dimension for Transformer: (N_stocks, 64) → (1, N_stocks, 64)
        transformer_in = lstm_out.unsqueeze(0)

        # (1, N_stocks, 64) → (1, N_stocks, 128)
        transformer_out = self.transformer_encoder(
            transformer_in,
            stock_indices = stock_indices,
            padding_mask  = padding_mask,
        )

        # Remove batch dim: (1, N_stocks, 128) → (N_stocks, 128)
        transformer_out = transformer_out.squeeze(0)

        # ── Step 3: Fusion — concatenate LSTM + Transformer outputs ───────
        # (N_stocks, 64) || (N_stocks, 128) → (N_stocks, 192)
        fused = torch.cat([lstm_out, transformer_out], dim=-1)

        # (N_stocks, 192) → (N_stocks, 128)
        embedding = self.fusion_head(fused)

        # ── Step 4: Prediction heads (Phase 2 only) ───────────────────────
        if self.pretrain_mode and self.direction_head is not None:
            direction_logit = self.direction_head(embedding)   # (N_stocks,)
            return_pred     = self.return_head(embedding)      # (N_stocks,)
            return embedding, direction_logit, return_pred

        return embedding

    # ── Loss computation ──────────────────────────────────────────────────

    def compute_loss(
        self,
        embedding       : torch.Tensor,
        direction_logit : torch.Tensor,
        return_pred     : torch.Tensor,
        direction_target: torch.Tensor,
        return_target   : torch.Tensor,
        direction_weight: float = 0.6,
        return_weight   : float = 0.4,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes the combined supervised pre-training loss.

        Loss = direction_weight × BCE(direction) + return_weight × MSE(return)

        Default weights: 0.6 × BCE + 0.4 × MSE
        Rationale: direction accuracy matters more than precise magnitude;
        getting the direction right is 80% of profitable trading.

        Args:
            embedding        : (N_stocks, 128) — backbone output (unused in loss,
                               passed for potential regularization)
            direction_logit  : (N_stocks,) — raw logit from DirectionHead
            return_pred      : (N_stocks,) — predicted return from ReturnHead
            direction_target : (N_stocks,) — binary labels (1=up, 0=down)
            return_target    : (N_stocks,) — actual 5-day forward return (%)
            direction_weight : Weight for BCE loss (default: 0.6)
            return_weight    : Weight for MSE loss (default: 0.4)

        Returns:
            Dict with keys:
                total_loss      : Scalar tensor (backprop through this)
                direction_loss  : BCE component
                return_loss     : MSE component
                direction_acc   : Fraction of correctly predicted directions
        """
        # Binary cross-entropy for direction (up/down)
        direction_loss = F.binary_cross_entropy_with_logits(
            direction_logit,
            direction_target.float(),
        )

        # Mean squared error for return magnitude
        return_loss = F.mse_loss(return_pred, return_target.float())

        # Combined loss
        total_loss = direction_weight * direction_loss + return_weight * return_loss

        # Direction accuracy (for monitoring — not differentiable)
        with torch.no_grad():
            pred_dir = (direction_logit.sigmoid() > 0.5).float()
            direction_acc = (pred_dir == direction_target.float()).float().mean()

        return {
            "total_loss"    : total_loss,
            "direction_loss": direction_loss,
            "return_loss"   : return_loss,
            "direction_acc" : direction_acc,
        }

    # ── Mode switching ────────────────────────────────────────────────────

    def set_inference_mode(self):
        """
        Switches backbone to inference mode (Phase 3+).
        Disables prediction heads. forward() returns only embedding.
        """
        self.pretrain_mode  = False
        self.direction_head = None
        self.return_head    = None
        self.eval()

    def freeze_backbone(self):
        """
        Freezes all backbone parameters (LSTM + Transformer + FusionHead).
        Used in Phase 3 to freeze pre-trained weights while training RL heads.
        Call unfreeze_backbone() for end-to-end fine-tuning.
        """
        for param in self.lstm_encoder.parameters():
            param.requires_grad = False
        for param in self.transformer_encoder.parameters():
            param.requires_grad = False
        for param in self.fusion_head.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """
        Unfreezes all backbone parameters for end-to-end fine-tuning.
        Call after RL heads have stabilized (typically 25% into Phase 3).
        """
        for param in self.parameters():
            param.requires_grad = True

    def freeze_lstm_only(self):
        """
        Freezes only the LSTM encoder (keeps Transformer + FusionHead trainable).
        Useful for domain adaptation — LSTM captures generic temporal patterns
        while Transformer adapts to new market regimes.
        """
        for param in self.lstm_encoder.parameters():
            param.requires_grad = False

    # ── Checkpoint management ─────────────────────────────────────────────

    def save_checkpoint(
        self,
        path       : str | Path,
        epoch      : int,
        optimizer  : Optional[torch.optim.Optimizer] = None,
        metrics    : Optional[dict] = None,
        config     : Optional[dict] = None,
    ):
        """
        Saves full model checkpoint with training state.

        Args:
            path      : File path for checkpoint (e.g. 'checkpoints/epoch_10.pt')
            epoch     : Current epoch number
            optimizer : Optimizer state (for resuming training)
            metrics   : Dict of validation metrics at this epoch
            config    : Model config dict (for reproducibility)

        Example:
            backbone.save_checkpoint(
                "checkpoints/best.pt",
                epoch=42,
                optimizer=optimizer,
                metrics={"val_ic": 0.062, "val_acc": 0.591},
            )
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch"         : epoch,
            "model_state"   : self.state_dict(),
            "pretrain_mode" : self.pretrain_mode,
            "metrics"       : metrics or {},
            "config"        : config  or {},
            "timestamp"     : time.strftime("%Y-%m-%d %H:%M:%S"),
            "pytorch_version": torch.__version__,
        }

        if optimizer is not None:
            checkpoint["optimizer_state"] = optimizer.state_dict()

        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(
        cls,
        path           : str | Path,
        map_location   : Optional[str] = None,
        inference_only : bool          = False,
    ) -> Tuple["GodsEyeBackbone", dict]:
        """
        Loads a backbone from a checkpoint file.

        Args:
            path           : Path to checkpoint file
            map_location   : Device to load to (e.g. 'cpu', 'cuda:0')
                             If None, loads to original device.
            inference_only : If True, switches to inference mode after loading
                             (disables prediction heads)

        Returns:
            backbone  : Loaded GodsEyeBackbone
            meta      : Dict with epoch, metrics, config, timestamp

        Example:
            # Load for continued training
            backbone, meta = GodsEyeBackbone.load_checkpoint("checkpoints/best.pt")
            print(f"Resuming from epoch {meta['epoch']}, val_acc={meta['metrics']}")

            # Load for Phase 3 RL (inference only)
            backbone, _ = GodsEyeBackbone.load_checkpoint(
                "checkpoints/best.pt", inference_only=True
            )
        """
        if map_location is None:
            map_location = "cuda" if torch.cuda.is_available() else "cpu"

        checkpoint = torch.load(path, map_location=map_location, weights_only=False)

        pretrain_mode = checkpoint.get("pretrain_mode", True)
        if inference_only:
            pretrain_mode = False

        backbone = cls(pretrain_mode=pretrain_mode)
        backbone.load_state_dict(checkpoint["model_state"], strict=False)
        backbone.to(map_location)

        if inference_only:
            backbone.set_inference_mode()

        meta = {
            "epoch"    : checkpoint.get("epoch", 0),
            "metrics"  : checkpoint.get("metrics", {}),
            "config"   : checkpoint.get("config", {}),
            "timestamp": checkpoint.get("timestamp", ""),
        }

        return backbone, meta

    # ── Utilities ─────────────────────────────────────────────────────────

    @property
    def num_parameters(self) -> int:
        """Total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def num_backbone_parameters(self) -> int:
        """Parameters in LSTM + Transformer + FusionHead only (not prediction heads)."""
        backbone_params = (
            list(self.lstm_encoder.parameters()) +
            list(self.transformer_encoder.parameters()) +
            list(self.fusion_head.parameters())
        )
        return sum(p.numel() for p in backbone_params if p.requires_grad)

    def summary(self) -> str:
        """Human-readable architecture summary."""
        lines = [
            "╔══════════════════════════════════════════════╗",
            "║         GodsEyeBackbone Architecture         ║",
            "╚══════════════════════════════════════════════╝",
            f"  Mode       : {'PRETRAIN' if self.pretrain_mode else 'INFERENCE'}",
            f"",
            f"  [1] LSTMEncoder",
            f"      Input  : (N_stocks, 60, 28)",
            f"      Output : (N_stocks, 64)",
            f"      Params : {self.lstm_encoder.num_parameters:,}",
            f"",
            f"  [2] TransformerEncoder",
            f"      Input  : (1, N_stocks, 64)",
            f"      Output : (1, N_stocks, 128)",
            f"      Params : {self.transformer_encoder.num_parameters:,}",
            f"",
            f"  [3] FusionHead",
            f"      Input  : (N_stocks, 192)  [64 || 128 concat]",
            f"      Output : (N_stocks, 128)",
            f"",
        ]
        if self.pretrain_mode:
            lines += [
                f"  [4] DirectionHead (Phase 2 only)",
                f"      Input  : (N_stocks, 128)",
                f"      Output : (N_stocks,)  [logit for BCE]",
                f"",
                f"  [5] ReturnHead (Phase 2 only)",
                f"      Input  : (N_stocks, 128)",
                f"      Output : (N_stocks,)  [predicted 5d return %]",
                f"",
            ]
        lines += [
            f"  Backbone params : {self.num_backbone_parameters:,}",
            f"  Total params    : {self.num_parameters:,}",
            f"  Final embedding : (N_stocks, 128)",
        ]
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════
#  FACTORY FUNCTION
# ══════════════════════════════════════════════════════════════════════════

def build_backbone(
    pretrain_mode: bool = True,
    config       : Optional[dict] = None,
) -> GodsEyeBackbone:
    """
    Builds and returns a GodsEyeBackbone on the available device.

    Args:
        pretrain_mode : True for Phase 2 training, False for Phase 3+ inference
        config        : Optional config dict (keys: lstm, transformer, fusion)

    Returns:
        GodsEyeBackbone on GPU if available, else CPU
    """
    lstm_cfg        = config.get("lstm", None)        if config else None
    transformer_cfg = config.get("transformer", None) if config else None
    dropout         = config.get("fusion_dropout", 0.2) if config else 0.2

    backbone = GodsEyeBackbone(
        lstm_config        = lstm_cfg,
        transformer_config = transformer_cfg,
        backbone_dropout   = dropout,
        pretrain_mode      = pretrain_mode,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return backbone.to(device)


# ══════════════════════════════════════════════════════════════════════════
#  UNIT TESTS
#  Run with: python -m pytest models/backbone.py -v
# ══════════════════════════════════════════════════════════════════════════

class TestGodsEyeBackbone:
    """Unit tests for GodsEyeBackbone in both pretrain and inference modes."""

    def setup_method(self):
        torch.manual_seed(42)
        self.device   = torch.device("cpu")
        self.n_stocks = 10    # small for CPU tests
        self.seq_len  = 60
        self.n_feat   = 28

        # Standard input: (N_stocks, seq, features)
        self.x = torch.randn(self.n_stocks, self.seq_len, self.n_feat)

        # Pretrain + inference backbones
        self.backbone_pretrain   = GodsEyeBackbone(pretrain_mode=True).to(self.device)
        self.backbone_inference  = GodsEyeBackbone(pretrain_mode=False).to(self.device)
        self.backbone_pretrain.eval()
        self.backbone_inference.eval()

    # ── Shape tests (pretrain mode) ───────────────────────────────────────

    def test_pretrain_output_shapes(self):
        """Pretrain mode must return (embedding, direction_logit, return_pred)."""
        with torch.no_grad():
            emb, dir_logit, ret_pred = self.backbone_pretrain(self.x)

        assert emb.shape      == (self.n_stocks, BACKBONE_OUT_DIM), \
            f"Embedding shape wrong: {emb.shape}"
        assert dir_logit.shape == (self.n_stocks,), \
            f"Direction logit shape wrong: {dir_logit.shape}"
        assert ret_pred.shape  == (self.n_stocks,), \
            f"Return pred shape wrong: {ret_pred.shape}"

    def test_embedding_dim(self):
        """Embedding must always be 128-dimensional."""
        with torch.no_grad():
            emb, _, _ = self.backbone_pretrain(self.x)
        assert emb.shape[-1] == 128

    # ── Shape tests (inference mode) ─────────────────────────────────────

    def test_inference_output_shape(self):
        """Inference mode must return only embedding (N_stocks, 128)."""
        with torch.no_grad():
            emb = self.backbone_inference(self.x)
        assert emb.shape == (self.n_stocks, BACKBONE_OUT_DIM), \
            f"Inference embedding shape wrong: {emb.shape}"

    def test_inference_returns_tensor_not_tuple(self):
        """Inference mode forward must return a Tensor, not a tuple."""
        with torch.no_grad():
            out = self.backbone_inference(self.x)
        assert isinstance(out, torch.Tensor), \
            f"Inference mode returned {type(out)}, expected Tensor"

    def test_variable_n_stocks(self):
        """Must handle any number of stocks."""
        for n in [1, 5, 20, 50]:
            x = torch.randn(n, self.seq_len, self.n_feat)
            with torch.no_grad():
                emb = self.backbone_inference(x)
            assert emb.shape == (n, 128), f"Shape wrong for n_stocks={n}"

    # ── Value tests ───────────────────────────────────────────────────────

    def test_no_nan_pretrain(self):
        """No NaN in any pretrain output."""
        with torch.no_grad():
            emb, dir_logit, ret_pred = self.backbone_pretrain(self.x)
        assert not torch.isnan(emb).any(),      "NaN in embedding"
        assert not torch.isnan(dir_logit).any(),"NaN in direction logit"
        assert not torch.isnan(ret_pred).any(), "NaN in return pred"

    def test_no_nan_inference(self):
        """No NaN in inference output."""
        with torch.no_grad():
            emb = self.backbone_inference(self.x)
        assert not torch.isnan(emb).any(), "NaN in inference embedding"

    def test_no_inf(self):
        """No Inf in any output."""
        with torch.no_grad():
            emb, dir_logit, ret_pred = self.backbone_pretrain(self.x)
        assert not torch.isinf(emb).any()
        assert not torch.isinf(dir_logit).any()
        assert not torch.isinf(ret_pred).any()

    def test_direction_logit_unbounded(self):
        """Direction logit must be unbounded (before sigmoid)."""
        with torch.no_grad():
            _, dir_logit, _ = self.backbone_pretrain(self.x)
        # Raw logit — should not be clipped to [0,1]
        # At init, values are typically in [-3, +3]
        assert dir_logit.abs().max() < 100, "Direction logit suspiciously large"

    def test_different_inputs_different_embeddings(self):
        """Different inputs must produce different embeddings."""
        x1 = torch.randn(self.n_stocks, self.seq_len, self.n_feat)
        x2 = torch.randn(self.n_stocks, self.seq_len, self.n_feat)
        with torch.no_grad():
            emb1 = self.backbone_inference(x1)
            emb2 = self.backbone_inference(x2)
        assert not torch.allclose(emb1, emb2, atol=1e-4)

    # ── Loss computation tests ────────────────────────────────────────────

    def test_loss_computation(self):
        """compute_loss must return valid positive loss values."""
        with torch.no_grad():
            emb, dir_logit, ret_pred = self.backbone_pretrain(self.x)

        dir_target = torch.randint(0, 2, (self.n_stocks,)).float()
        ret_target = torch.randn(self.n_stocks) * 2   # ±2% typical

        losses = self.backbone_pretrain.compute_loss(
            emb, dir_logit, ret_pred, dir_target, ret_target
        )

        assert "total_loss"     in losses
        assert "direction_loss" in losses
        assert "return_loss"    in losses
        assert "direction_acc"  in losses

        assert losses["total_loss"].item() > 0, "Total loss must be positive"
        assert not math.isnan(losses["total_loss"].item()), "NaN loss"
        assert 0.0 <= losses["direction_acc"].item() <= 1.0, \
            "Direction accuracy must be in [0, 1]"

    def test_loss_weights_applied(self):
        """
        Total loss = 0.6 × BCE + 0.4 × MSE.
        Verify the combination is correct.
        """
        with torch.no_grad():
            emb, dir_logit, ret_pred = self.backbone_pretrain(self.x)

        dir_target = torch.ones(self.n_stocks)
        ret_target = torch.zeros(self.n_stocks)

        losses = self.backbone_pretrain.compute_loss(
            emb, dir_logit, ret_pred, dir_target, ret_target,
            direction_weight=0.6, return_weight=0.4
        )

        expected = (
            0.6 * losses["direction_loss"] +
            0.4 * losses["return_loss"]
        )
        assert torch.allclose(losses["total_loss"], expected, atol=1e-5)

    # ── Gradient tests ────────────────────────────────────────────────────

    def test_gradients_flow_end_to_end(self):
        """Gradients must flow from loss through all backbone components."""
        backbone = GodsEyeBackbone(pretrain_mode=True).train()
        x = torch.randn(self.n_stocks, self.seq_len, self.n_feat)

        emb, dir_logit, ret_pred = backbone(x)

        dir_target = torch.randint(0, 2, (self.n_stocks,)).float()
        ret_target = torch.randn(self.n_stocks)

        losses = backbone.compute_loss(
            emb, dir_logit, ret_pred, dir_target, ret_target
        )
        losses["total_loss"].backward()

        for name, param in backbone.named_parameters():
            if param.requires_grad:
                assert param.grad is not None,        f"No gradient: {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient: {name}"

    # ── Mode switching tests ──────────────────────────────────────────────

    def test_set_inference_mode(self):
        """set_inference_mode must disable prediction heads."""
        backbone = GodsEyeBackbone(pretrain_mode=True)
        assert backbone.pretrain_mode is True

        backbone.set_inference_mode()
        assert backbone.pretrain_mode  is False
        assert backbone.direction_head is None
        assert backbone.return_head    is None

        # forward() should return tensor, not tuple
        x = torch.randn(5, 60, 28)
        with torch.no_grad():
            out = backbone(x)
        assert isinstance(out, torch.Tensor)

    def test_freeze_backbone(self):
        """freeze_backbone must set requires_grad=False for LSTM+Transformer+Fusion."""
        backbone = GodsEyeBackbone(pretrain_mode=True)
        backbone.freeze_backbone()

        for param in backbone.lstm_encoder.parameters():
            assert not param.requires_grad, "LSTM param not frozen"
        for param in backbone.transformer_encoder.parameters():
            assert not param.requires_grad, "Transformer param not frozen"
        for param in backbone.fusion_head.parameters():
            assert not param.requires_grad, "FusionHead param not frozen"

        # Prediction heads should still be trainable
        if backbone.direction_head:
            any_trainable = any(
                p.requires_grad for p in backbone.direction_head.parameters()
            )
            assert any_trainable, "Direction head should remain trainable after freeze"

    def test_unfreeze_backbone(self):
        """unfreeze_backbone must re-enable all gradients."""
        backbone = GodsEyeBackbone(pretrain_mode=True)
        backbone.freeze_backbone()
        backbone.unfreeze_backbone()

        for name, param in backbone.named_parameters():
            assert param.requires_grad, f"Parameter still frozen after unfreeze: {name}"

    def test_freeze_lstm_only(self):
        """freeze_lstm_only must freeze only LSTM, not Transformer."""
        backbone = GodsEyeBackbone(pretrain_mode=False)
        backbone.freeze_lstm_only()

        for param in backbone.lstm_encoder.parameters():
            assert not param.requires_grad, "LSTM not frozen"
        for param in backbone.transformer_encoder.parameters():
            assert param.requires_grad, "Transformer should not be frozen"

    # ── Checkpoint tests ──────────────────────────────────────────────────

    def test_save_and_load_checkpoint(self, tmp_path):
        """Save then load must produce identical outputs."""
        backbone = GodsEyeBackbone(pretrain_mode=True).eval()
        x = torch.randn(self.n_stocks, self.seq_len, self.n_feat)

        with torch.no_grad():
            emb_before, _, _ = backbone(x)

        # Save
        ckpt_path = tmp_path / "test_backbone.pt"
        backbone.save_checkpoint(
            ckpt_path, epoch=5,
            metrics={"val_acc": 0.59, "val_ic": 0.06}
        )

        # Load
        loaded, meta = GodsEyeBackbone.load_checkpoint(
            ckpt_path, map_location="cpu"
        )
        loaded.eval()

        with torch.no_grad():
            emb_after, _, _ = loaded(x)

        assert torch.allclose(emb_before, emb_after, atol=1e-5), \
            "Embeddings changed after save/load"
        assert meta["epoch"]   == 5
        assert meta["metrics"] == {"val_acc": 0.59, "val_ic": 0.06}

    def test_load_checkpoint_inference_only(self, tmp_path):
        """Loading with inference_only=True must disable prediction heads."""
        backbone = GodsEyeBackbone(pretrain_mode=True)
        ckpt_path = tmp_path / "inference_test.pt"
        backbone.save_checkpoint(ckpt_path, epoch=10)

        loaded, _ = GodsEyeBackbone.load_checkpoint(
            ckpt_path, map_location="cpu", inference_only=True
        )

        assert loaded.pretrain_mode  is False
        assert loaded.direction_head is None
        assert loaded.return_head    is None

    # ── Parameter count test ──────────────────────────────────────────────

    def test_parameter_count(self):
        """Total parameter count must be in a reasonable range."""
        backbone = GodsEyeBackbone(pretrain_mode=True)
        n = backbone.num_parameters
        assert 1_000_000 < n < 15_000_000, \
            f"Parameter count {n:,} outside expected [1M, 15M]"

    # ── Integration: full pre-training step simulation ────────────────────

    def test_full_pretrain_step(self):
        """
        Simulates one complete Phase 2 training step:
        forward → loss → backward → optimizer step.
        """
        backbone  = GodsEyeBackbone(pretrain_mode=True).train()
        optimizer = torch.optim.AdamW(backbone.parameters(), lr=1e-4, weight_decay=0.01)

        # Simulate fusion.py output: (N_stocks=10, seq=60, features=28)
        x = torch.rand(10, 60, 28) * 2 - 1   # values in [-1, +1]

        # Simulated labels
        dir_target = torch.randint(0, 2, (10,)).float()
        ret_target = torch.randn(10) * 2

        # Forward
        emb, dir_logit, ret_pred = backbone(x)

        # Loss
        losses = backbone.compute_loss(
            emb, dir_logit, ret_pred, dir_target, ret_target
        )

        # Backward
        optimizer.zero_grad()
        losses["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(backbone.parameters(), max_norm=1.0)
        optimizer.step()

        assert not math.isnan(losses["total_loss"].item())
        assert losses["total_loss"].item() > 0


# ── Run when executed directly ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import pytest

    backbone = GodsEyeBackbone(pretrain_mode=True)
    print(backbone.summary())
    print()

    sys.exit(pytest.main([__file__, "-v", "--tb=short", "--no-header"]))