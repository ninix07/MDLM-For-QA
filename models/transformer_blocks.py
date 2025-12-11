"""
Transformer blocks for the conditional denoiser.
"""

from typing import Optional
import torch
import torch.nn as nn


class ConditionalTransformerBlock(nn.Module):
    """
    Transformer block with cross-attention for conditioning.

    Structure:
    1. Self-attention on noisy latent
    2. Cross-attention with condition (Q + C)
    3. Feed-forward network

    All with residual connections, layer norm, and AdaLN timestep modulation.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.self_attn_norm = nn.LayerNorm(d_model)

        # Cross-attention with condition
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn_norm = nn.LayerNorm(d_model)

        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )
        self.ff_norm = nn.LayerNorm(d_model)

        # Timestep modulation (AdaLN-style): scale, shift for 3 norms
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, d_model * 6),
        )

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        time_emb: torch.Tensor,
        x_mask: Optional[torch.Tensor] = None,
        condition_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Noisy latent [batch, seq_len, d_model]
            condition: Condition embeddings [batch, cond_len, d_model]
            time_emb: Timestep embedding [batch, d_model]
            x_mask: Key padding mask for x (True = ignore)
            condition_mask: Key padding mask for condition (True = ignore)
        Returns:
            output: [batch, seq_len, d_model]
        """
        # Get modulation parameters from timestep
        time_params = self.time_mlp(time_emb)
        scale1, shift1, scale2, shift2, scale3, shift3 = time_params.chunk(6, dim=-1)

        # Self-attention with AdaLN
        x_norm = self.self_attn_norm(x)
        x_norm = x_norm * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, key_padding_mask=x_mask)
        x = x + attn_out

        # Cross-attention with condition
        x_norm = self.cross_attn_norm(x)
        x_norm = x_norm * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        cross_out, _ = self.cross_attn(
            x_norm, condition, condition, key_padding_mask=condition_mask
        )
        x = x + cross_out

        # Feed-forward with AdaLN
        x_norm = self.ff_norm(x)
        x_norm = x_norm * (1 + scale3.unsqueeze(1)) + shift3.unsqueeze(1)
        x = x + self.ff(x_norm)

        return x


class SelfAttentionBlock(nn.Module):
    """Simple self-attention block without cross-attention."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, key_padding_mask=mask)
        x = x + attn_out

        # Feed-forward
        x = x + self.ff(self.norm2(x))
        return x
