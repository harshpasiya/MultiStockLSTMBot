"""
╔══════════════════════════════════════════════════════════════════════════╗
║         G.O.D.S E.Y.E — BiLSTM Temporal Encoder                        ║
║         Project : MultiStockLSTMBot                                      ║
║         File    : models/lstm_encoder.py                                 ║
║         Phase   : 2 — AI Backbone (Supervised Pre-training)             ║
║                                                                          ║
║  What this module does:                                                  ║
║    Encodes a 60-day sequence of 28 features per stock into a compact     ║
║    64-dimensional temporal context vector that captures:                 ║
║      • Multi-week trend patterns (accumulation, distribution)            ║
║      • Earnings cycle effects (quarterly rhythm)                         ║
║      • Sector rotation momentum over weeks                               ║
║      • Volatility regime transitions                                     ║
║                                                                          ║
║  Why BiLSTM (Bidirectional)?                                             ║
║    In supervised pre-training, we CAN look at the full sequence          ║
║    (no future leakage because target = forward return, not current bar). ║
║    BiLSTM reads both forward and backward through the 60-day window,     ║
║    capturing context from both directions — e.g., a spike mid-sequence   ║
║    is understood better when the model knows what came after it.         ║
║    During inference, only the final hidden state is used (no leakage).  ║
║                                                                          ║
║  Architecture:                                                           ║
║    Input  : (batch_size, seq_len=60, input_dim=28)                      ║
║    Layer 1: BiLSTM(256) → dropout(0.3) → (batch, 60, 512)              ║
║    Layer 2: BiLSTM(128) → dropout(0.3) → (batch, 60, 256)              ║
║    Layer 3: BiLSTM(64)  → (batch, 60, 128)                              ║
║    Pool   : Concat[last_fwd(64) + last_bwd(64)] → (batch, 128)         ║
║    Project: Linear(128→64) + LayerNorm → (batch, 64)                   ║
║    Output : (batch_size, 64)   ← temporal context vector                ║
║                                                                          ║
║  Key design decisions:                                                   ║
║    • Decreasing hidden dims (256→128→64): forces compression / learning ║
║    • Dropout between layers (not on last): regularization without        ║
║      destroying the final representation                                 ║
║    • LayerNorm on output: stabilizes Transformer input in backbone.py   ║
║    • Attention pooling option: alternative to last-state pooling,        ║
║      learns which timesteps matter most for prediction                   ║
║                                                                          ║
║  Dependencies:                                                           ║
║    pip install torch>=2.1.0                                              ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple


# ── Module-level constants (match Phase 1 fusion output) ─────────────────
DEFAULT_INPUT_DIM  = 28    # 28 features from fusion.py (features [0]–[27])
DEFAULT_SEQ_LEN    = 60    # 60-day lookback window
DEFAULT_OUTPUT_DIM = 64    # temporal context vector dimension


# ══════════════════════════════════════════════════════════════════════════
#  ATTENTION POOLING
#  Used instead of simple last-state pooling when use_attention=True.
#  Learns a weighted sum over all 60 timesteps — some days matter more.
# ══════════════════════════════════════════════════════════════════════════

class TemporalAttentionPool(nn.Module):
    """
    Soft attention over the time dimension.

    Given hidden states of shape (batch, seq_len, hidden_dim),
    learns a scalar attention weight for each timestep and returns
    a weighted sum: (batch, hidden_dim).

    This is superior to simple last-state pooling because:
        - An earnings announcement 30 bars ago may matter more than yesterday
        - Volatility spikes mid-sequence are explicitly attended to
        - The model learns WHICH days are informative for predicting returns

    Args:
        hidden_dim : Dimension of LSTM hidden states to attend over
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        # Single linear layer produces scalar score per timestep
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states : (batch, seq_len, hidden_dim)

        Returns:
            context       : (batch, hidden_dim) — attention-weighted sum
            weights       : (batch, seq_len) — attention weights (for interpretability)
        """
        # Score each timestep: (batch, seq_len, 1)
        scores = self.attention(hidden_states)

        # Normalize across time: (batch, seq_len, 1)
        weights = F.softmax(scores, dim=1)

        # Weighted sum across time: (batch, hidden_dim)
        context = (weights * hidden_states).sum(dim=1)

        return context, weights.squeeze(-1)


# ══════════════════════════════════════════════════════════════════════════
#  LSTM ENCODER
# ══════════════════════════════════════════════════════════════════════════

class LSTMEncoder(nn.Module):
    """
    3-layer stacked Bidirectional LSTM for temporal sequence encoding.

    Encodes a 60-day window of 28 market features into a 64-dimensional
    temporal context vector representing the stock's recent market state.

    Args:
        input_dim     : Number of input features per timestep (default: 28)
        seq_len       : Sequence length / lookback window (default: 60)
        output_dim    : Final output embedding dimension (default: 64)
        hidden_dims   : Tuple of hidden sizes for each BiLSTM layer
                        (default: (256, 128, 64))
        dropout       : Dropout rate between layers (default: 0.3)
        use_attention : If True, use attention pooling over all timesteps
                        instead of using only the final hidden state.
                        Attention is more expressive but slower.
                        (default: True)

    Input shape  : (batch_size, seq_len, input_dim)
    Output shape : (batch_size, output_dim)

    Example:
        encoder = LSTMEncoder()
        x = torch.randn(64, 60, 28)   # batch of 64 stocks, 60 days, 28 features
        out = encoder(x)              # → (64, 64)
    """

    def __init__(
        self,
        input_dim    : int              = DEFAULT_INPUT_DIM,
        seq_len      : int              = DEFAULT_SEQ_LEN,
        output_dim   : int              = DEFAULT_OUTPUT_DIM,
        hidden_dims  : Tuple[int, ...] = (256, 128, 64),
        dropout      : float            = 0.3,
        use_attention: bool             = True,
    ):
        super().__init__()

        self.input_dim     = input_dim
        self.seq_len       = seq_len
        self.output_dim    = output_dim
        self.hidden_dims   = hidden_dims
        self.dropout_rate  = dropout
        self.use_attention = use_attention

        # ── Input projection ──────────────────────────────────────────────
        # Project raw features into a richer representation before LSTM.
        # This acts as a learned feature embedding layer.
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
        )

        # ── Stacked BiLSTM layers ─────────────────────────────────────────
        # BiLSTM hidden_size is half the total because fwd + bwd are concat'd.
        # e.g. BiLSTM(hidden_size=256) outputs 512 dims (256 fwd + 256 bwd)
        self.lstm_layers = nn.ModuleList()
        self.dropouts    = nn.ModuleList()

        lstm_input_dim = hidden_dims[0]   # after input_proj

        for i, hidden_size in enumerate(hidden_dims):
            self.lstm_layers.append(
                nn.LSTM(
                    input_size  = lstm_input_dim,
                    hidden_size = hidden_size,
                    num_layers  = 1,
                    batch_first = True,
                    bidirectional=True,
                )
            )
            # Dropout after all layers except the last
            if i < len(hidden_dims) - 1:
                self.dropouts.append(nn.Dropout(dropout))
            else:
                self.dropouts.append(nn.Identity())

            # Next layer input = bidirectional output (hidden_size × 2)
            lstm_input_dim = hidden_size * 2

        # ── Pooling ───────────────────────────────────────────────────────
        # Final BiLSTM outputs hidden_dims[-1] * 2 dimensions
        final_hidden = hidden_dims[-1] * 2   # 64 * 2 = 128

        if use_attention:
            self.pool = TemporalAttentionPool(final_hidden)
        else:
            self.pool = None   # will use last hidden state instead

        # ── Output projection ─────────────────────────────────────────────
        # Project from final_hidden (128) to output_dim (64)
        self.output_proj = nn.Sequential(
            nn.Linear(final_hidden, output_dim),
            nn.LayerNorm(output_dim),
        )

        # ── Weight initialization ─────────────────────────────────────────
        self._init_weights()

    def _init_weights(self):
        """
        Initializes LSTM weights using orthogonal initialization for
        recurrent weights (reduces vanishing/exploding gradients) and
        Xavier uniform for input weights.
        """
        for lstm in self.lstm_layers:
            for name, param in lstm.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param.data)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param.data)
                elif "bias" in name:
                    # Set forget gate bias to 1.0 — helps remember long-term patterns
                    # LSTM bias layout: [input, forget, cell, output] gates
                    n = param.size(0)
                    param.data.fill_(0)
                    param.data[n // 4: n // 2].fill_(1.0)  # forget gate

        # Linear layers: Xavier uniform
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x         : torch.Tensor,
        return_all: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the BiLSTM encoder.

        Args:
            x          : Input tensor of shape (batch, seq_len, input_dim)
            return_all : If True, also return per-timestep hidden states
                         from the last LSTM layer (used by Transformer encoder
                         in backbone.py for cross-stock attention).

        Returns:
            If return_all=False:
                output : (batch, output_dim) — temporal context vector

            If return_all=True:
                output      : (batch, output_dim) — temporal context vector
                all_hidden  : (batch, seq_len, hidden_dims[-1]*2)
                              all hidden states from last layer

        Raises:
            ValueError : If input shape doesn't match expected dimensions
        """
        if x.dim() != 3:
            raise ValueError(
                f"Expected 3D input (batch, seq_len, features), got {x.dim()}D"
            )

        batch_size, seq_len, feat_dim = x.shape

        if feat_dim != self.input_dim:
            raise ValueError(
                f"Input feature dim {feat_dim} doesn't match "
                f"expected {self.input_dim}. "
                f"Check fusion.py output has {self.input_dim} features."
            )

        # ── Input projection: (batch, seq, input_dim) → (batch, seq, h0) ─
        x = self.input_proj(x)

        # ── Stacked BiLSTM forward pass ───────────────────────────────────
        all_hidden_states = None

        for i, (lstm, drop) in enumerate(zip(self.lstm_layers, self.dropouts)):
            # lstm output shape: (batch, seq_len, hidden_size * 2)
            x, (h_n, c_n) = lstm(x)

            # Save last layer's all-timestep hidden states for Transformer
            if i == len(self.lstm_layers) - 1:
                all_hidden_states = x   # (batch, seq, hidden_dims[-1]*2)

            # Apply dropout between layers
            x = drop(x)

        # all_hidden_states: (batch, seq_len, 128) after last BiLSTM
        assert all_hidden_states is not None

        # ── Pooling: sequence → vector ────────────────────────────────────
        if self.use_attention and self.pool is not None:
            # Attention pooling: (batch, 128) + weights (batch, seq)
            pooled, attn_weights = self.pool(all_hidden_states)
        else:
            # Last hidden state pooling: concat final fwd + bwd hidden states
            # h_n shape: (num_directions=2, batch, hidden_dims[-1]=64)
            # After concat: (batch, 128)
            h_n = h_n.view(2, batch_size, self.hidden_dims[-1])
            pooled = torch.cat([h_n[0], h_n[1]], dim=-1)   # (batch, 128)
            attn_weights = None

        # ── Output projection: (batch, 128) → (batch, 64) ────────────────
        output = self.output_proj(pooled)

        if return_all:
            return output, all_hidden_states

        return output

    def get_sequence_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns per-timestep embeddings for all 60 days.
        Used when the Transformer needs full sequence context,
        not just the summary vector.

        Args:
            x : (batch, seq_len, input_dim)

        Returns:
            all_hidden : (batch, seq_len, hidden_dims[-1]*2)
                         i.e. (batch, 60, 128)
        """
        _, all_hidden = self.forward(x, return_all=True)
        return all_hidden

    @property
    def num_parameters(self) -> int:
        """Returns total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self) -> str:
        """Returns a human-readable model summary string."""
        lines = [
            "╔══════════════════════════════════════╗",
            "║       LSTMEncoder Architecture       ║",
            "╚══════════════════════════════════════╝",
            f"  Input      : (batch, {self.seq_len}, {self.input_dim})",
            f"  Input proj : Linear({self.input_dim} → {self.hidden_dims[0]})",
        ]
        lstm_in = self.hidden_dims[0]
        for i, h in enumerate(self.hidden_dims):
            lines.append(
                f"  BiLSTM {i+1}   : ({lstm_in} → {h}×2={h*2})"
                + (f" + Dropout({self.dropout_rate})" if i < len(self.hidden_dims)-1 else "")
            )
            lstm_in = h * 2
        pool_type = "TemporalAttention" if self.use_attention else "LastHidden"
        lines.append(f"  Pooling    : {pool_type}")
        lines.append(f"  Output proj: Linear({self.hidden_dims[-1]*2} → {self.output_dim}) + LayerNorm")
        lines.append(f"  Output     : (batch, {self.output_dim})")
        lines.append(f"  Parameters : {self.num_parameters:,}")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════
#  FACTORY FUNCTION
# ══════════════════════════════════════════════════════════════════════════

def build_lstm_encoder(config: Optional[dict] = None) -> LSTMEncoder:
    """
    Builds an LSTMEncoder from a config dict (from model_config.yaml).

    Args:
        config : Optional dict with keys matching LSTMEncoder __init__ params.
                 If None, uses all defaults (suitable for Phase 2 training).

    Returns:
        LSTMEncoder instance, moved to GPU if available.

    Example:
        # Default config (matches Phase 2 spec)
        encoder = build_lstm_encoder()

        # Custom config from yaml
        import yaml
        with open("config/model_config.yaml") as f:
            cfg = yaml.safe_load(f)
        encoder = build_lstm_encoder(cfg["lstm_encoder"])
    """
    defaults = {
        "input_dim"    : DEFAULT_INPUT_DIM,
        "seq_len"      : DEFAULT_SEQ_LEN,
        "output_dim"   : DEFAULT_OUTPUT_DIM,
        "hidden_dims"  : (256, 128, 64),
        "dropout"      : 0.3,
        "use_attention": True,
    }

    if config:
        defaults.update(config)
        # yaml loads lists, not tuples — convert hidden_dims
        if isinstance(defaults["hidden_dims"], list):
            defaults["hidden_dims"] = tuple(defaults["hidden_dims"])

    encoder = LSTMEncoder(**defaults)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device)

    return encoder


# ══════════════════════════════════════════════════════════════════════════
#  UNIT TESTS
#  Run with: python -m pytest models/lstm_encoder.py -v
# ══════════════════════════════════════════════════════════════════════════

class TestLSTMEncoder:
    """
    Unit tests for LSTMEncoder.
    All tests use CPU to avoid GPU requirement in CI environments.
    """

    def setup_method(self):
        """Create default encoder and sample inputs for each test."""
        torch.manual_seed(42)
        self.device  = torch.device("cpu")
        self.encoder = LSTMEncoder().to(self.device)
        self.encoder.eval()

        # Standard batch: 64 stocks, 60 days, 28 features
        self.batch_size = 64
        self.seq_len    = 60
        self.input_dim  = 28
        self.x = torch.randn(self.batch_size, self.seq_len, self.input_dim)

    # ── Shape tests ───────────────────────────────────────────────────────

    def test_output_shape_default(self):
        """Standard forward pass must return (batch, 64)."""
        with torch.no_grad():
            out = self.encoder(self.x)
        assert out.shape == (self.batch_size, DEFAULT_OUTPUT_DIM), \
            f"Expected ({self.batch_size}, {DEFAULT_OUTPUT_DIM}), got {out.shape}"

    def test_output_shape_batch_1(self):
        """Single-sample inference must work (batch_size=1)."""
        x = torch.randn(1, self.seq_len, self.input_dim)
        with torch.no_grad():
            out = self.encoder(x)
        assert out.shape == (1, DEFAULT_OUTPUT_DIM)

    def test_output_shape_batch_1_stock(self):
        """Minimum viable inference for live signal generation."""
        x = torch.randn(1, self.seq_len, self.input_dim)
        with torch.no_grad():
            out = self.encoder(x)
        assert out.shape == (1, 64)

    def test_return_all_shapes(self):
        """return_all=True must return (batch,64) and (batch,60,128)."""
        with torch.no_grad():
            out, hidden = self.encoder(self.x, return_all=True)
        assert out.shape    == (self.batch_size, 64), \
            f"Output shape wrong: {out.shape}"
        assert hidden.shape == (self.batch_size, self.seq_len, 128), \
            f"Hidden shape wrong: {hidden.shape}"

    def test_sequence_embeddings_shape(self):
        """get_sequence_embeddings must return (batch, seq_len, 128)."""
        with torch.no_grad():
            seq_emb = self.encoder.get_sequence_embeddings(self.x)
        assert seq_emb.shape == (self.batch_size, self.seq_len, 128)

    def test_custom_hidden_dims(self):
        """Custom hidden dims must produce correct output shape."""
        encoder = LSTMEncoder(
            input_dim=28, output_dim=64,
            hidden_dims=(128, 64, 32)
        )
        x = torch.randn(4, 60, 28)
        with torch.no_grad():
            out = encoder(x)
        assert out.shape == (4, 64)

    def test_custom_output_dim(self):
        """Custom output_dim must be respected."""
        encoder = LSTMEncoder(output_dim=128)
        x = torch.randn(4, 60, 28)
        with torch.no_grad():
            out = encoder(x)
        assert out.shape == (4, 128)

    def test_different_batch_sizes(self):
        """Forward pass must work for various batch sizes."""
        for bs in [1, 8, 32, 64, 128]:
            x = torch.randn(bs, 60, 28)
            with torch.no_grad():
                out = self.encoder(x)
            assert out.shape == (bs, 64), f"Failed for batch_size={bs}"

    # ── Value tests ───────────────────────────────────────────────────────

    def test_output_no_nan(self):
        """Output must never contain NaN values."""
        with torch.no_grad():
            out = self.encoder(self.x)
        assert not torch.isnan(out).any(), "NaN values in encoder output"

    def test_output_no_inf(self):
        """Output must never contain Inf values."""
        with torch.no_grad():
            out = self.encoder(self.x)
        assert not torch.isinf(out).any(), "Inf values in encoder output"

    def test_output_not_all_zeros(self):
        """Output must not be a zero vector (dead network check)."""
        with torch.no_grad():
            out = self.encoder(self.x)
        assert out.abs().max() > 1e-6, "Encoder output is all zeros — dead network"

    def test_layernorm_output_scale(self):
        """
        LayerNorm on output should keep values in a reasonable range.
        After LayerNorm, output should not have extreme values.
        """
        with torch.no_grad():
            out = self.encoder(self.x)
        # LayerNorm normalizes per-sample, so std across features ≈ 1
        std_per_sample = out.std(dim=-1)
        # Most samples should have std between 0.5 and 2.0
        in_range = ((std_per_sample > 0.3) & (std_per_sample < 3.0)).float().mean()
        assert in_range > 0.8, \
            f"LayerNorm not working correctly: {in_range:.1%} samples in range"

    def test_different_inputs_different_outputs(self):
        """Different inputs must produce different outputs (no constant mapping)."""
        x1 = torch.randn(8, 60, 28)
        x2 = torch.randn(8, 60, 28)
        with torch.no_grad():
            out1 = self.encoder(x1)
            out2 = self.encoder(x2)
        assert not torch.allclose(out1, out2, atol=1e-4), \
            "Encoder produces identical outputs for different inputs"

    def test_same_input_same_output(self):
        """Same input must always produce same output (deterministic in eval)."""
        x = torch.randn(8, 60, 28)
        with torch.no_grad():
            out1 = self.encoder(x)
            out2 = self.encoder(x)
        assert torch.allclose(out1, out2, atol=1e-6), \
            "Encoder is non-deterministic in eval mode"

    def test_without_attention_pooling(self):
        """Last-hidden-state pooling (use_attention=False) must also work."""
        encoder = LSTMEncoder(use_attention=False)
        x = torch.randn(16, 60, 28)
        with torch.no_grad():
            out = encoder(x)
        assert out.shape == (16, 64)
        assert not torch.isnan(out).any()

    # ── Gradient tests ────────────────────────────────────────────────────

    def test_gradients_flow(self):
        """Gradients must flow back through all layers during training."""
        encoder = LSTMEncoder().train()
        x = torch.randn(8, 60, 28, requires_grad=False)
        out = encoder(x)
        loss = out.mean()
        loss.backward()

        # Check that all parameters received gradients
        for name, param in encoder.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, \
                    f"No gradient for parameter: {name}"
                assert not torch.isnan(param.grad).any(), \
                    f"NaN gradient for parameter: {name}"

    def test_dropout_active_in_train_mode(self):
        """Dropout should make outputs stochastic in train mode."""
        encoder = LSTMEncoder(dropout=0.5).train()
        x = torch.randn(32, 60, 28)
        with torch.no_grad():
            out1 = encoder(x)
            out2 = encoder(x)
        # With 50% dropout, outputs should differ
        assert not torch.allclose(out1, out2, atol=1e-4), \
            "Dropout not active in train mode"

    def test_dropout_inactive_in_eval_mode(self):
        """Dropout must be disabled in eval mode (deterministic output)."""
        encoder = LSTMEncoder(dropout=0.5).eval()
        x = torch.randn(32, 60, 28)
        with torch.no_grad():
            out1 = encoder(x)
            out2 = encoder(x)
        assert torch.allclose(out1, out2, atol=1e-6), \
            "Non-deterministic output in eval mode — dropout not disabled"

    # ── Error handling tests ──────────────────────────────────────────────

    def test_wrong_input_dims_raises(self):
        """2D input (missing seq_len dim) must raise ValueError."""
        import pytest
        x_2d = torch.randn(64, 28)
        with pytest.raises(ValueError, match="Expected 3D input"):
            self.encoder(x_2d)

    def test_wrong_feature_dim_raises(self):
        """Wrong number of features must raise ValueError."""
        import pytest
        x_wrong = torch.randn(8, 60, 15)   # 15 features instead of 28
        with pytest.raises(ValueError, match="Input feature dim"):
            self.encoder(x_wrong)

    # ── Architecture tests ────────────────────────────────────────────────

    def test_num_lstm_layers(self):
        """Must have exactly 3 BiLSTM layers."""
        assert len(self.encoder.lstm_layers) == 3

    def test_bidirectional(self):
        """All LSTM layers must be bidirectional."""
        for lstm in self.encoder.lstm_layers:
            assert lstm.bidirectional, "LSTM layer is not bidirectional"

    def test_parameter_count_reasonable(self):
        """
        Parameter count should be in a reasonable range.
        Too few = underpowered. Too many = overfitting risk.
        Default config should be ~2–6M parameters.
        """
        n_params = self.encoder.num_parameters
        assert 500_000 < n_params < 10_000_000, \
            f"Parameter count {n_params:,} outside expected range [500K, 10M]"

    def test_attention_pool_weights_sum_to_1(self):
        """Attention weights must sum to 1.0 across time dimension."""
        assert self.encoder.use_attention and self.encoder.pool is not None, \
            "Attention pool not enabled"

        x = torch.randn(8, 60, 128)   # input to attention pool
        with torch.no_grad():
            _, weights = self.encoder.pool(x)

        weight_sums = weights.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones(8), atol=1e-5), \
            f"Attention weights don't sum to 1: {weight_sums}"

    def test_forget_gate_bias_initialized(self):
        """Forget gate bias should be initialized to 1.0."""
        for lstm in self.encoder.lstm_layers:
            # Bias layout: [input(h), forget(h), cell(h), output(h)]
            bias = lstm.bias_ih_l0
            n = bias.size(0)
            forget_bias = bias[n // 4: n // 2]
            # After initialization, forget gate should be ≈ 1.0
            assert forget_bias.mean().item() > 0.5, \
                "Forget gate bias not initialized to 1.0"

    # ── Integration test ──────────────────────────────────────────────────

    def test_full_pipeline_simulation(self):
        """
        Simulates the full Phase 2 training step:
        fusion output → LSTM encoder → loss → backward.
        """
        encoder = LSTMEncoder().train()
        optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-4)

        # Simulate a batch from fusion.py output
        # Shape: (batch=64, seq=60, features=28), all values in [-1, +1]
        x = torch.rand(64, 60, 28) * 2 - 1

        # Forward pass
        output = encoder(x)         # (64, 64)
        assert output.shape == (64, 64)

        # Simulate a supervised learning target (direction prediction)
        target = torch.randint(0, 2, (64,)).float()  # binary: up or down
        pred   = output.mean(dim=-1).sigmoid()        # (64,) in [0,1]
        loss   = F.binary_cross_entropy(pred, target)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert loss.item() > 0, "Loss should be positive"
        assert not math.isnan(loss.item()), "Loss is NaN"


# ── Run tests when file is executed directly ──────────────────────────────
if __name__ == "__main__":
    import sys
    import pytest

    # Print model summary before running tests
    encoder = LSTMEncoder()
    print(encoder.summary())
    print()

    sys.exit(pytest.main([__file__, "-v", "--tb=short", "--no-header"]))