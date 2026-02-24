import torch
import torch.nn as nn
import torch.nn.functional as F


class ZodicSeq2Seq(nn.Module):
    """
    Encoder-Decoder Seq2Seq for 5-day price path prediction.

    Input  : (batch, 45, 16)  — 45 days × 16 features
    Output : (batch,  5)      — predicted 5-day cumulative return path
    """

    def __init__(
        self,
        n_features : int   = 16,
        enc_hidden : int   = 64,
        dec_hidden : int   = 64,
        enc_layers : int   = 2,
        dec_layers : int   = 1,
        dropout    : float = 0.25,
        horizon    : int   = 5,
    ):
        super().__init__()
        self.horizon    = horizon
        self.dec_hidden = dec_hidden
        self.dec_layers = dec_layers

        # LayerNorm across features at each timestep
        self.input_norm = nn.LayerNorm(n_features)

        # ── ENCODER ──────────────────────────────────────────────────
        # 2-layer GRU: layer 1 captures short patterns (5-10d),
        #              layer 2 captures medium patterns (20-45d)
        self.encoder = nn.GRU(
            input_size  = n_features,
            hidden_size = enc_hidden,
            num_layers  = enc_layers,
            batch_first = True,
            dropout     = dropout if enc_layers > 1 else 0.0,
        )

        # ── CONTEXT PROJECTION ───────────────────────────────────────
        # enc hidden -> dec hidden initialisation
        self.context_proj = nn.Sequential(
            nn.Linear(enc_hidden, dec_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── DECODER ──────────────────────────────────────────────────
        # At each step: input = prev_prediction (1) + context (enc_hidden)
        self.decoder = nn.GRU(
            input_size  = 1 + enc_hidden,
            hidden_size = dec_hidden,
            num_layers  = dec_layers,
            batch_first = True,
            dropout     = 0.0,
        )

        # ── OUTPUT PROJECTION ────────────────────────────────────────
        self.out_proj = nn.Sequential(
            nn.LayerNorm(dec_hidden),
            nn.Linear(dec_hidden, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)       # prevents vanishing gradients in RNNs
            elif "bias" in name:
                nn.init.zeros_(param)
            elif "weight" in name and param.dim() == 2:
                nn.init.xavier_uniform_(param)

    def encode(self, x: torch.Tensor):
        """
        Args:
            x : (batch, 45, n_features)
        Returns:
            enc_out : (batch, 45, enc_hidden)   all timestep outputs
            context : (batch, enc_hidden)        top-layer final hidden state
        """
        x       = self.input_norm(x)
        enc_out, h_n = self.encoder(x)      # h_n: (enc_layers, batch, enc_hidden)
        context = h_n[-1]                   # (batch, enc_hidden) — top layer only
        return enc_out, context

    def decode(self, context: torch.Tensor) -> torch.Tensor:
        """
        Auto-regressive decoding — 5 steps.

        At each step t:
          decoder input = [prev_prediction | context_vector]
          decoder output -> predicted cumulative return at t+1

        Args:
            context : (batch, enc_hidden)
        Returns:
            path    : (batch, 5)
        """
        batch = context.size(0)

        # Initialise decoder hidden from projected context
        h = self.context_proj(context)                          # (batch, dec_hidden)
        h = h.unsqueeze(0).repeat(self.dec_layers, 1, 1)       # (dec_layers, batch, dec_hidden)

        preds  = []
        prev_y = torch.zeros(batch, 1, device=context.device)  # start token = 0

        for _ in range(self.horizon):
            dec_in     = torch.cat(
                [prev_y.unsqueeze(1), context.unsqueeze(1)], dim=-1
            )                                                   # (batch, 1, 1+enc_hidden)
            dec_out, h = self.decoder(dec_in, h)               # (batch, 1, dec_hidden)
            y          = self.out_proj(dec_out.squeeze(1))     # (batch, 1)
            preds.append(y)
            prev_y = y.detach()                                 # no teacher forcing

        return torch.cat(preds, dim=1)                         # (batch, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x    : (batch, 45, 16)
        Returns:
            path : (batch, 5)
        """
        _, context = self.encode(x)
        return self.decode(context)
