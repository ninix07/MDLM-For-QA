"""
Transformer blocks for the conditional denoiser.
"""

from typing import Optional
import torch
import torch.nn as nn

class ConditionalTransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        
        # 1. Standard Transformer Components
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.self_attn_norm = nn.LayerNorm(d_model, elementwise_affine=False) # Norm logic handled by AdaLN
        
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout)
        )
        self.ff_norm = nn.LayerNorm(d_model, elementwise_affine=False)

        # 2. AdaLN-Zero Timestep MLP
        # Generates: [shift1, scale1, gate1, shift2, scale2, gate2] per attention/FF stage
        # We need 3 sets (Self-Attn, Cross-Attn, Feed-Forward) -> 9 parameters
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, d_model * 9) 
        )
        
        # INITIALIZATION: Zero out the final layer to enforce Identity at start
        nn.init.zeros_(self.time_mlp[1].weight)
        nn.init.zeros_(self.time_mlp[1].bias)

    def modulate(self, x, shift, scale):
        # Apply shift and scale to normalized input
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(self, x, condition, time_emb, x_mask=None, condition_mask=None):
        # A. Get AdaLN parameters for this timestep
        t_out = self.time_mlp(time_emb)
        sft1, scl1, gt1, sft2, scl2, gt2, sft3, scl3, gt3 = t_out.chunk(9, dim=-1)

        # B. Self-Attention (Gated Residual)
        h = self.modulate(self.self_attn_norm(x), sft1, scl1)
        
        # SAFETY CHECK: Ensure x_mask is never all-zero
        if x_mask is not None:
            # key_padding_mask: True means IGNORE. If all are True, it's all-zero.
            all_masked = x_mask.all(dim=1)
            if all_masked.any():
                x_mask = x_mask.clone()
                x_mask[all_masked, 0] = False # Unmask first token
                
        h, _ = self.self_attn(h, h, h, key_padding_mask=x_mask)
        x = x + gt1.unsqueeze(1) * h
        
        # C. Cross-Attention (Gated Residual)
        h = self.modulate(self.cross_attn_norm(x), sft2, scl2)
        
        # SAFETY CHECK: Ensure condition_mask is never all-zero
        if condition_mask is not None:
            all_masked = condition_mask.all(dim=1)
            if all_masked.any():
                condition_mask = condition_mask.clone()
                condition_mask[all_masked, 0] = False # Unmask first token
                
        h, _ = self.cross_attn(h, condition, condition, key_padding_mask=condition_mask)
        x = x + gt2.unsqueeze(1) * h

        # D. Feed-Forward (Gated Residual)
        h = self.modulate(self.ff_norm(x), sft3, scl3)
        x = x + gt3.unsqueeze(1) * self.ff(h)

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
