"""
╔══════════════════════════════════════════════════════════════════════════╗
║         G.O.D.S E.Y.E — Cross-Stock Transformer Encoder                 ║
║         Project : MultiStockLSTMBot                                      ║
║         File    : models/transformer_encoder.py                          ║
║         Phase   : 2 — AI Backbone (Supervised Pre-training)             ║
║                                                                          ║
║  What this module does:                                                  ║
║    Takes the 64-dimensional temporal context vectors produced by the     ║
║    LSTMEncoder for ALL stocks simultaneously, and lets each stock        ║
║    "attend" to every other stock in the universe.                        ║
║                                                                          ║
║    This is where the system learns:                                      ║
║      • Which stocks are leading indicators for others                    ║
║        (e.g. HDFC Bank moves before smaller banking stocks)              ║
║      • Sector contagion — when IT bellwethers sell off, what follows     ║
║      • Market-wide sentiment embedded in cross-stock correlations        ║
║      • How FII-heavy stocks signal before FII data is officially out     ║
║                                                                          ║
║  Why Transformer (not just LSTM)?                                        ║
║    LSTM processes one stock at a time with no awareness of other stocks. ║
║    The Transformer processes ALL stocks simultaneously and explicitly     ║
║    models pairwise relationships via attention weights. This gives the   ║
║    model a "god's eye view" of the entire market at each step.           ║
║                                                                          ║
║  Architecture:                                                           ║
║    Input   : (batch, N_stocks, 64)  — LSTM outputs for all stocks       ║
║    ↓ Linear projection: 64 → 128 (model_dim)                            ║
║    ↓ Positional encoding (learnable stock embeddings)                    ║
║    ↓ TransformerLayer 1: 4-head MHA + FFN(512) + residual + norm        ║
║    ↓ TransformerLayer 2: 4-head MHA + FFN(512) + residual + norm        ║
║    ↓ Output projection: 128 → 128                                        ║
║    Output  : (batch, N_stocks, 128) — cross-stock enriched embeddings   ║
║                                                                          ║
║  Attention interpretation:                                               ║
║    attn_weights[i, j] = how much stock i attends to stock j             ║
║    High weight = stock j's state strongly influences prediction for i    ║
║    These weights are saved for interpretability / correlation pillar     ║
║                                                                          ║
║  Key design decisions:                                                   ║
║    • Pre-LN (LayerNorm before attention): more stable training than      ║
║      Post-LN; avoids gradient explosion in early epochs                  ║
║    • Learnable stock positional embeddings: unlike NLP where position    ║
║      matters sequentially, stocks have no inherent order — but learned  ║
║      embeddings let the model assign stable identity to each stock       ║
║    • Causal masking DISABLED: we attend across all stocks equally        ║
║      (no temporal ordering between stocks, unlike token sequences)       ║
║    • FFN dim 512: wider than model_dim (128) for expressive power       ║
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


# ── Module-level constants ────────────────────────────────────────────────
DEFAULT_INPUT_DIM  = 64     # LSTMEncoder output dimension
DEFAULT_MODEL_DIM  = 128    # Internal Transformer dimension
DEFAULT_OUTPUT_DIM = 128    # Final output per stock
DEFAULT_N_HEADS    = 4      # Multi-head attention heads
DEFAULT_N_LAYERS   = 2      # Transformer encoder layers
DEFAULT_FFN_DIM    = 512    # Feed-forward network hidden dimension
DEFAULT_DROPOUT    = 0.1    # Transformer dropout (lower than LSTM — less data)
DEFAULT_MAX_STOCKS = 500    # Maximum stocks in universe (Nifty 500)


# ══════════════════════════════════════════════════════════════════════════
#  LEARNABLE STOCK POSITIONAL EMBEDDING
# ══════════════════════════════════════════════════════════════════════════

class StockPositionalEmbedding(nn.Module):
    """
    Learnable positional embedding for stocks in the universe.

    Unlike NLP where positional embeddings encode token ORDER in a sequence,
    stocks have no inherent sequential order. Instead, this gives each stock
    a learnable identity embedding that the model can use to distinguish
    HDFC Bank from Reliance even when their feature vectors look similar.

    During training the model learns that certain positions (stocks) have
    predictable roles in the cross-attention graph — e.g. index heavyweights
    like Reliance, TCS, HDFC always influence many other stocks.

    Args:
        max_stocks : Maximum number of stocks (Nifty 500 = 500)
        model_dim  : Dimension of the embedding (matches Transformer model_dim)
    """

    def __init__(self, max_stocks: int = DEFAULT_MAX_STOCKS, model_dim: int = DEFAULT_MODEL_DIM):
        super().__init__()
        # Each stock gets a unique learnable embedding vector
        self.embedding = nn.Embedding(max_stocks, model_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, stock_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Adds positional embeddings to stock feature vectors.

        Args:
            x             : (batch, n_stocks, model_dim) — projected LSTM outputs
            stock_indices : (n_stocks,) — integer indices identifying each stock
                            If None, uses [0, 1, 2, ..., n_stocks-1]

        Returns:
            x + positional_embedding : (batch, n_stocks, model_dim)
        """
        n_stocks = x.size(1)

        if stock_indices is None:
            stock_indices = torch.arange(n_stocks, device=x.device)

        # (n_stocks, model_dim) → (1, n_stocks, model_dim) → broadcast
        pos_emb = self.embedding(stock_indices).unsqueeze(0)
        return x + pos_emb


# ══════════════════════════════════════════════════════════════════════════
#  SINGLE TRANSFORMER LAYER (Pre-LN variant)
# ══════════════════════════════════════════════════════════════════════════

class TransformerEncoderLayer(nn.Module):
    """
    Single Pre-LN Transformer encoder layer.

    Pre-LN applies LayerNorm BEFORE the attention/FFN sublayers
    (unlike the original "Attention is All You Need" Post-LN).
    Pre-LN is more stable for training deep transformers and is
    used in GPT-2, GPT-3, and most modern LLMs.

    Structure per layer:
        x = x + MultiHeadAttention(LayerNorm(x))   ← self-attention sublayer
        x = x + FFN(LayerNorm(x))                  ← feed-forward sublayer

    Args:
        model_dim  : Embedding dimension (default: 128)
        n_heads    : Number of attention heads (default: 4)
        ffn_dim    : Feed-forward hidden dimension (default: 512)
        dropout    : Dropout rate (default: 0.1)
    """

    def __init__(
        self,
        model_dim : int   = DEFAULT_MODEL_DIM,
        n_heads   : int   = DEFAULT_N_HEADS,
        ffn_dim   : int   = DEFAULT_FFN_DIM,
        dropout   : float = DEFAULT_DROPOUT,
    ):
        super().__init__()

        assert model_dim % n_heads == 0, (
            f"model_dim ({model_dim}) must be divisible by n_heads ({n_heads})"
        )

        self.model_dim = model_dim
        self.n_heads   = n_heads

        # ── Pre-LN normalization layers ───────────────────────────────────
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

        # ── Multi-head self-attention ─────────────────────────────────────
        # PyTorch's built-in MHA handles Q, K, V projections internally
        self.self_attn = nn.MultiheadAttention(
            embed_dim   = model_dim,
            num_heads   = n_heads,
            dropout     = dropout,
            batch_first = True,    # input: (batch, seq, dim) not (seq, batch, dim)
        )

        # ── Feed-forward network ──────────────────────────────────────────
        # FFN: model_dim → ffn_dim → model_dim with GELU activation
        # GELU is smoother than ReLU and works better for financial data
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, model_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x             : torch.Tensor,
        attn_mask     : Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through one Transformer encoder layer.

        Args:
            x                : (batch, n_stocks, model_dim)
            attn_mask        : Optional attention mask (e.g. to mask certain stocks)
            key_padding_mask : (batch, n_stocks) — True for positions to ignore
                               Used when batch has variable number of stocks
            return_weights   : If True, return attention weights for visualization

        Returns:
            output  : (batch, n_stocks, model_dim)
            weights : (batch, n_stocks, n_stocks) or None
                      weights[b, i, j] = attention stock i pays to stock j in batch b
        """
        # ── Self-attention sublayer (Pre-LN) ──────────────────────────────
        residual = x
        x_norm   = self.norm1(x)

        attn_out, attn_weights = self.self_attn(
            query              = x_norm,
            key                = x_norm,
            value              = x_norm,
            attn_mask          = attn_mask,
            key_padding_mask   = key_padding_mask,
            need_weights       = return_weights,
            average_attn_weights=True,   # average over heads for interpretability
        )

        x = residual + self.dropout(attn_out)

        # ── Feed-forward sublayer (Pre-LN) ────────────────────────────────
        residual = x
        x_norm   = self.norm2(x)
        x        = residual + self.dropout(self.ffn(x_norm))

        return x, attn_weights if return_weights else None


# ══════════════════════════════════════════════════════════════════════════
#  TRANSFORMER ENCODER (FULL)
# ══════════════════════════════════════════════════════════════════════════

class TransformerEncoder(nn.Module):
    """
    Multi-layer cross-stock Transformer Encoder.

    Processes all stocks simultaneously, enriching each stock's embedding
    with information from the entire market universe via self-attention.

    After this module, each stock's 128-dim vector contains not just its
    own temporal patterns (from LSTM) but also context from all other
    stocks — enabling the model to detect sector contagion, lead-lag
    relationships, and market-wide regime shifts.

    Args:
        input_dim  : Dimension of LSTM encoder output (default: 64)
        model_dim  : Internal Transformer dimension (default: 128)
        output_dim : Final output dimension per stock (default: 128)
        n_heads    : Number of attention heads (default: 4)
        n_layers   : Number of Transformer layers (default: 2)
        ffn_dim    : Feed-forward hidden dimension (default: 512)
        dropout    : Dropout rate (default: 0.1)
        max_stocks : Maximum stocks for positional embedding (default: 500)

    Input shape  : (batch, n_stocks, input_dim=64)
    Output shape : (batch, n_stocks, output_dim=128)

    Example:
        transformer = TransformerEncoder()

        # Simulate LSTM output for 64 batch × 500 stocks × 64 dims
        lstm_out = torch.randn(64, 500, 64)
        out = transformer(lstm_out)     # → (64, 500, 128)

        # Or with fewer stocks (e.g. filtered universe)
        lstm_out = torch.randn(64, 100, 64)
        out = transformer(lstm_out)     # → (64, 100, 128)
    """

    def __init__(
        self,
        input_dim  : int   = DEFAULT_INPUT_DIM,
        model_dim  : int   = DEFAULT_MODEL_DIM,
        output_dim : int   = DEFAULT_OUTPUT_DIM,
        n_heads    : int   = DEFAULT_N_HEADS,
        n_layers   : int   = DEFAULT_N_LAYERS,
        ffn_dim    : int   = DEFAULT_FFN_DIM,
        dropout    : float = DEFAULT_DROPOUT,
        max_stocks : int   = DEFAULT_MAX_STOCKS,
    ):
        super().__init__()

        self.input_dim  = input_dim
        self.model_dim  = model_dim
        self.output_dim = output_dim
        self.n_heads    = n_heads
        self.n_layers   = n_layers

        # ── Input projection: LSTM dim → Transformer dim ──────────────────
        # 64 (LSTM output) → 128 (Transformer model_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.LayerNorm(model_dim),
        )

        # ── Learnable stock positional embeddings ─────────────────────────
        self.pos_embedding = StockPositionalEmbedding(max_stocks, model_dim)

        # ── Transformer encoder layers ────────────────────────────────────
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                model_dim = model_dim,
                n_heads   = n_heads,
                ffn_dim   = ffn_dim,
                dropout   = dropout,
            )
            for _ in range(n_layers)
        ])

        # ── Final LayerNorm (Post-LN on full stack output) ────────────────
        self.final_norm = nn.LayerNorm(model_dim)

        # ── Output projection: model_dim → output_dim ────────────────────
        # If model_dim == output_dim this is an identity-like projection
        # but still provides a learnable linear transformation
        if model_dim != output_dim:
            self.output_proj = nn.Linear(model_dim, output_dim)
        else:
            self.output_proj = nn.Identity()

        # ── Weight initialization ─────────────────────────────────────────
        self._init_weights()

    def _init_weights(self):
        """Xavier uniform for linear layers, small normal for embeddings."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        x             : torch.Tensor,
        stock_indices : Optional[torch.Tensor] = None,
        padding_mask  : Optional[torch.Tensor] = None,
        return_weights: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, list]:
        """
        Forward pass through the full cross-stock Transformer.

        Args:
            x              : (batch, n_stocks, input_dim=64) — LSTM outputs
            stock_indices  : (n_stocks,) — integer IDs for positional embedding
                             If None, uses sequential [0, 1, ..., n_stocks-1]
            padding_mask   : (batch, n_stocks) — True for padded/missing stocks
                             Used when batch has stocks with no data that day
            return_weights : If True, returns attention weights from all layers
                             for visualization and interpretability

        Returns:
            If return_weights=False:
                output : (batch, n_stocks, output_dim=128)

            If return_weights=True:
                output  : (batch, n_stocks, output_dim=128)
                weights : list of (batch, n_stocks, n_stocks) per layer
                          weights[layer][b, i, j] = attention stock i → stock j

        Raises:
            ValueError : If input shape doesn't match expected dimensions
        """
        if x.dim() != 3:
            raise ValueError(
                f"Expected 3D input (batch, n_stocks, features), got {x.dim()}D. "
                f"LSTMEncoder output must be stacked across stocks before passing here."
            )

        batch_size, n_stocks, feat_dim = x.shape

        if feat_dim != self.input_dim:
            raise ValueError(
                f"Input dim {feat_dim} doesn't match expected {self.input_dim}. "
                f"LSTMEncoder output_dim must equal TransformerEncoder input_dim."
            )

        # ── Project LSTM outputs to Transformer dimension ─────────────────
        # (batch, n_stocks, 64) → (batch, n_stocks, 128)
        x = self.input_proj(x)

        # ── Add learnable stock identity embeddings ───────────────────────
        x = self.pos_embedding(x, stock_indices)

        # ── Pass through Transformer layers ──────────────────────────────
        all_attn_weights = []

        for layer in self.layers:
            x, weights = layer(
                x,
                key_padding_mask = padding_mask,
                return_weights   = return_weights,
            )
            if return_weights and weights is not None:
                all_attn_weights.append(weights)

        # ── Final normalization ───────────────────────────────────────────
        x = self.final_norm(x)

        # ── Output projection ─────────────────────────────────────────────
        # (batch, n_stocks, 128) → (batch, n_stocks, 128)
        x = self.output_proj(x)

        if return_weights:
            return x, all_attn_weights

        return x

    def get_attention_map(
        self,
        x            : torch.Tensor,
        stock_indices: Optional[torch.Tensor] = None,
        layer_idx    : int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the attention weight map for a given layer.
        Useful for understanding which stocks influence which others.

        Args:
            x             : (batch, n_stocks, input_dim)
            stock_indices : Optional stock index tensor
            layer_idx     : Which layer's attention to return (-1 = last layer)

        Returns:
            output  : (batch, n_stocks, output_dim)
            weights : (batch, n_stocks, n_stocks)
                      Entry [b, i, j] = how much stock i attended to stock j
        """
        output, all_weights = self.forward(
            x, stock_indices=stock_indices, return_weights=True
        )
        if not all_weights:
            return output, torch.zeros(x.size(0), x.size(1), x.size(1))

        return output, all_weights[layer_idx]

    @property
    def num_parameters(self) -> int:
        """Returns total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self) -> str:
        """Returns a human-readable model summary string."""
        head_dim = self.model_dim // self.n_heads
        lines = [
            "╔══════════════════════════════════════════╗",
            "║     TransformerEncoder Architecture      ║",
            "╚══════════════════════════════════════════╝",
            f"  Input      : (batch, N_stocks, {self.input_dim})",
            f"  Input proj : Linear({self.input_dim} → {self.model_dim}) + LayerNorm",
            f"  Pos embed  : Learnable ({DEFAULT_MAX_STOCKS} stocks × {self.model_dim})",
        ]
        for i in range(self.n_layers):
            lines.append(
                f"  Layer {i+1}    : MHA({self.n_heads} heads, dim={head_dim}) "
                f"+ FFN({self.model_dim}→{DEFAULT_FFN_DIM}→{self.model_dim})"
            )
        lines.append(f"  Final norm : LayerNorm({self.model_dim})")
        lines.append(f"  Output proj: Linear({self.model_dim} → {self.output_dim})")
        lines.append(f"  Output     : (batch, N_stocks, {self.output_dim})")
        lines.append(f"  Parameters : {self.num_parameters:,}")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════
#  FACTORY FUNCTION
# ══════════════════════════════════════════════════════════════════════════

def build_transformer_encoder(config: Optional[dict] = None) -> TransformerEncoder:
    """
    Builds a TransformerEncoder from a config dict (from model_config.yaml).

    Args:
        config : Optional dict with keys matching TransformerEncoder __init__ params.
                 If None, uses all defaults (suitable for Phase 2 training).

    Returns:
        TransformerEncoder moved to GPU if available.

    Example:
        encoder = build_transformer_encoder()
        print(encoder.summary())
    """
    defaults = {
        "input_dim"  : DEFAULT_INPUT_DIM,
        "model_dim"  : DEFAULT_MODEL_DIM,
        "output_dim" : DEFAULT_OUTPUT_DIM,
        "n_heads"    : DEFAULT_N_HEADS,
        "n_layers"   : DEFAULT_N_LAYERS,
        "ffn_dim"    : DEFAULT_FFN_DIM,
        "dropout"    : DEFAULT_DROPOUT,
        "max_stocks" : DEFAULT_MAX_STOCKS,
    }

    if config:
        defaults.update(config)

    encoder = TransformerEncoder(**defaults)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return encoder.to(device)


# ══════════════════════════════════════════════════════════════════════════
#  UNIT TESTS
#  Run with: python -m pytest models/transformer_encoder.py -v
# ══════════════════════════════════════════════════════════════════════════

class TestTransformerEncoder:
    """Unit tests for TransformerEncoder and its subcomponents."""

    def setup_method(self):
        torch.manual_seed(42)
        self.device      = torch.device("cpu")
        self.transformer = TransformerEncoder().to(self.device)
        self.transformer.eval()

        self.batch_size = 8
        self.n_stocks   = 50    # use 50 stocks in tests (not full 500 — slow on CPU)
        self.input_dim  = 64
        self.x = torch.randn(self.batch_size, self.n_stocks, self.input_dim)

    # ── Shape tests ───────────────────────────────────────────────────────

    def test_output_shape_default(self):
        """Standard forward must return (batch, n_stocks, 128)."""
        with torch.no_grad():
            out = self.transformer(self.x)
        assert out.shape == (self.batch_size, self.n_stocks, DEFAULT_OUTPUT_DIM), \
            f"Expected {(self.batch_size, self.n_stocks, DEFAULT_OUTPUT_DIM)}, got {out.shape}"

    def test_output_shape_single_stock(self):
        """Must handle n_stocks=1 (edge case for single-stock inference)."""
        x = torch.randn(4, 1, self.input_dim)
        with torch.no_grad():
            out = self.transformer(x)
        assert out.shape == (4, 1, DEFAULT_OUTPUT_DIM)

    def test_output_shape_full_universe(self):
        """Must handle full 500-stock universe."""
        x = torch.randn(2, 500, self.input_dim)
        with torch.no_grad():
            out = self.transformer(x)
        assert out.shape == (2, 500, DEFAULT_OUTPUT_DIM)

    def test_output_shape_variable_stocks(self):
        """Must work for any number of stocks up to max_stocks."""
        for n in [10, 50, 100, 200, 499]:
            x = torch.randn(2, n, self.input_dim)
            with torch.no_grad():
                out = self.transformer(x)
            assert out.shape == (2, n, DEFAULT_OUTPUT_DIM), \
                f"Shape mismatch for n_stocks={n}: {out.shape}"

    def test_return_weights_shapes(self):
        """return_weights=True must return output + list of weight tensors."""
        with torch.no_grad():
            out, weights = self.transformer(self.x, return_weights=True)

        assert out.shape == (self.batch_size, self.n_stocks, DEFAULT_OUTPUT_DIM)
        assert len(weights) == DEFAULT_N_LAYERS, \
            f"Expected {DEFAULT_N_LAYERS} weight tensors, got {len(weights)}"
        for w in weights:
            assert w.shape == (self.batch_size, self.n_stocks, self.n_stocks), \
                f"Weight shape wrong: {w.shape}"

    def test_get_attention_map_shape(self):
        """get_attention_map must return (batch, n_stocks, n_stocks) weights."""
        with torch.no_grad():
            out, weights = self.transformer.get_attention_map(self.x)
        assert out.shape     == (self.batch_size, self.n_stocks, DEFAULT_OUTPUT_DIM)
        assert weights.shape == (self.batch_size, self.n_stocks, self.n_stocks)

    # ── Value tests ───────────────────────────────────────────────────────

    def test_output_no_nan(self):
        """Output must never contain NaN."""
        with torch.no_grad():
            out = self.transformer(self.x)
        assert not torch.isnan(out).any(), "NaN in Transformer output"

    def test_output_no_inf(self):
        """Output must never contain Inf."""
        with torch.no_grad():
            out = self.transformer(self.x)
        assert not torch.isinf(out).any(), "Inf in Transformer output"

    def test_output_not_all_zeros(self):
        """Output must not be a zero tensor."""
        with torch.no_grad():
            out = self.transformer(self.x)
        assert out.abs().max() > 1e-6, "Transformer output is all zeros"

    def test_different_inputs_different_outputs(self):
        """Different inputs must produce different outputs."""
        x1 = torch.randn(4, self.n_stocks, self.input_dim)
        x2 = torch.randn(4, self.n_stocks, self.input_dim)
        with torch.no_grad():
            out1 = self.transformer(x1)
            out2 = self.transformer(x2)
        assert not torch.allclose(out1, out2, atol=1e-4)

    def test_deterministic_in_eval(self):
        """Same input must give same output in eval mode."""
        x = torch.randn(4, self.n_stocks, self.input_dim)
        with torch.no_grad():
            out1 = self.transformer(x)
            out2 = self.transformer(x)
        assert torch.allclose(out1, out2, atol=1e-6), \
            "Non-deterministic in eval mode"

    def test_attention_weights_sum_to_one(self):
        """
        Attention weights across the key dimension must sum to 1.0
        (they are softmax outputs over all stocks).
        """
        with torch.no_grad():
            _, weights = self.transformer(self.x, return_weights=True)

        for layer_idx, w in enumerate(weights):
            # w: (batch, n_stocks, n_stocks)
            # For each query stock i, weights over all key stocks j sum to 1
            row_sums = w.sum(dim=-1)   # (batch, n_stocks)
            assert torch.allclose(
                row_sums,
                torch.ones_like(row_sums),
                atol=1e-4
            ), f"Layer {layer_idx} attention weights don't sum to 1: {row_sums[0, :3]}"

    def test_attention_weights_non_negative(self):
        """Attention weights must all be non-negative (softmax output)."""
        with torch.no_grad():
            _, weights = self.transformer(self.x, return_weights=True)
        for w in weights:
            assert (w >= 0).all(), "Negative attention weights found"

    # ── Positional embedding tests ────────────────────────────────────────

    def test_positional_embedding_changes_output(self):
        """
        Different stock_indices must produce different outputs
        (positional embedding has effect).
        """
        x = torch.randn(2, 5, self.input_dim)
        idx1 = torch.tensor([0, 1, 2, 3, 4])
        idx2 = torch.tensor([10, 20, 30, 40, 50])
        with torch.no_grad():
            out1 = self.transformer(x, stock_indices=idx1)
            out2 = self.transformer(x, stock_indices=idx2)
        assert not torch.allclose(out1, out2, atol=1e-4), \
            "Positional embedding has no effect on output"

    def test_positional_embedding_shape(self):
        """Positional embedding must not change tensor shape."""
        x = torch.randn(4, self.n_stocks, DEFAULT_MODEL_DIM)
        pos_emb = StockPositionalEmbedding(500, DEFAULT_MODEL_DIM)
        out = pos_emb(x)
        assert out.shape == x.shape

    def test_default_stock_indices_sequential(self):
        """
        Calling with stock_indices=None must produce same result as
        calling with [0, 1, 2, ..., n_stocks-1].
        """
        x = torch.randn(2, 10, self.input_dim)
        explicit_idx = torch.arange(10)
        with torch.no_grad():
            out_implicit = self.transformer(x, stock_indices=None)
            out_explicit = self.transformer(x, stock_indices=explicit_idx)
        assert torch.allclose(out_implicit, out_explicit, atol=1e-6)

    # ── Padding mask tests ────────────────────────────────────────────────

    def test_padding_mask_accepted(self):
        """
        Padding mask must be accepted without errors.
        (Simulates a batch where some stocks have no data today)
        """
        x = torch.randn(4, self.n_stocks, self.input_dim)
        # Mark last 5 stocks as padded (no data)
        mask = torch.zeros(4, self.n_stocks, dtype=torch.bool)
        mask[:, -5:] = True

        with torch.no_grad():
            out = self.transformer(x, padding_mask=mask)
        assert out.shape == (4, self.n_stocks, DEFAULT_OUTPUT_DIM)
        assert not torch.isnan(out).any()

    # ── Architecture tests ────────────────────────────────────────────────

    def test_num_layers(self):
        """Must have exactly n_layers TransformerEncoderLayer instances."""
        assert len(self.transformer.layers) == DEFAULT_N_LAYERS

    def test_n_heads_per_layer(self):
        """Each layer must have the correct number of attention heads."""
        for layer in self.transformer.layers:
            assert layer.self_attn.num_heads == DEFAULT_N_HEADS

    def test_model_dim_divisible_by_heads(self):
        """model_dim must be divisible by n_heads."""
        assert DEFAULT_MODEL_DIM % DEFAULT_N_HEADS == 0, \
            f"{DEFAULT_MODEL_DIM} not divisible by {DEFAULT_N_HEADS}"

    def test_invalid_model_dim_raises(self):
        """model_dim not divisible by n_heads must raise AssertionError."""
        import pytest
        with pytest.raises(AssertionError):
            TransformerEncoderLayer(model_dim=100, n_heads=3)  # 100 / 3 not integer

    def test_parameter_count_reasonable(self):
        """Parameter count should be in reasonable range (~1–5M)."""
        n = self.transformer.num_parameters
        assert 200_000 < n < 10_000_000, \
            f"Parameter count {n:,} outside expected range"

    # ── Gradient tests ────────────────────────────────────────────────────

    def test_gradients_flow(self):
        """Gradients must flow back through all Transformer layers."""
        transformer = TransformerEncoder().train()
        x = torch.randn(4, self.n_stocks, self.input_dim)

        out  = transformer(x)
        loss = out.mean()
        loss.backward()

        for name, param in transformer.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient: {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient: {name}"

    def test_dropout_active_in_train(self):
        """Dropout must be stochastic in train mode."""
        transformer = TransformerEncoder(dropout=0.5).train()
        x = torch.randn(8, self.n_stocks, self.input_dim)
        with torch.no_grad():
            out1 = transformer(x)
            out2 = transformer(x)
        assert not torch.allclose(out1, out2, atol=1e-4), \
            "Dropout not active in train mode"

    # ── Error handling tests ──────────────────────────────────────────────

    def test_wrong_input_dim_raises(self):
        """2D input must raise ValueError."""
        import pytest
        x_2d = torch.randn(8, 64)
        with pytest.raises(ValueError, match="Expected 3D input"):
            self.transformer(x_2d)

    def test_wrong_feature_dim_raises(self):
        """Wrong feature dimension must raise ValueError."""
        import pytest
        x_wrong = torch.randn(4, self.n_stocks, 32)  # 32 instead of 64
        with pytest.raises(ValueError, match="Input dim"):
            self.transformer(x_wrong)

    # ── Integration test ──────────────────────────────────────────────────

    def test_lstm_to_transformer_pipeline(self):
        """
        Full pipeline: LSTMEncoder → TransformerEncoder in one pass.
        This simulates exactly how backbone.py will connect the two.
        """
        from models.lstm_encoder import LSTMEncoder

        lstm        = LSTMEncoder().eval()
        transformer = TransformerEncoder().eval()

        # Simulate 8 stocks, each with 60 days of 28 features
        n_stocks = 8
        x_stock  = torch.randn(n_stocks, 60, 28)   # (stocks, seq, features)

        with torch.no_grad():
            # Step 1: LSTM encodes each stock independently
            lstm_out = lstm(x_stock)               # (n_stocks, 64)

            # Step 2: Stack for Transformer (add batch dimension)
            # In real training: batch_size stocks processed together
            transformer_in = lstm_out.unsqueeze(0) # (1, n_stocks, 64)

            # Step 3: Transformer cross-stock attention
            transformer_out = transformer(transformer_in)  # (1, n_stocks, 128)

        assert transformer_out.shape == (1, n_stocks, 128)
        assert not torch.isnan(transformer_out).any()
        assert not torch.isinf(transformer_out).any()


# ── Run when executed directly ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import pytest

    transformer = TransformerEncoder()
    print(transformer.summary())
    print()

    sys.exit(pytest.main([__file__, "-v", "--tb=short", "--no-header"]))