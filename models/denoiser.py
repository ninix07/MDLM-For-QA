"""
Conditional Denoising Transformer for Latent Diffusion.
Predicts noise given noisy latent z_t, timestep t, and condition (Q + C).
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
from transformers import AutoModel

from .embeddings import SinusoidalTimestepEmbedding, TimestepMLP
from .transformer_blocks import ConditionalTransformerBlock


class ConditionalDenoiser(nn.Module):
    """Main Denoising Network for Latent Diffusion."""

    def __init__(
        self,
        latent_dim: int = 768,
        d_model: int = 768,
        num_layers: int = 6,
        num_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 64,
        condition_encoder: str = "xlm-roberta-base",
        freeze_condition_encoder: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.d_model = d_model

        # Use AutoModel for flexibility with different encoders (BERT, XLM-R, etc.)
        self.condition_encoder = AutoModel.from_pretrained(condition_encoder)
        if freeze_condition_encoder:
            for param in self.condition_encoder.parameters():
                param.requires_grad = False

        cond_dim = self.condition_encoder.config.hidden_size
        self.condition_proj = (
            nn.Linear(cond_dim, d_model) if cond_dim != d_model else nn.Identity()
        )

        self.time_embed = SinusoidalTimestepEmbedding(d_model)
        self.time_mlp = TimestepMLP(d_model, d_model * 4, d_model)
        self.input_proj = (
            nn.Linear(latent_dim, d_model) if latent_dim != d_model else nn.Identity()
        )
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)

        self.blocks = nn.ModuleList(
            [
                ConditionalTransformerBlock(d_model, num_heads, ff_dim, dropout)
                for _ in range(num_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, latent_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def encode_condition(
        self,
        context_ids: torch.Tensor,
        context_mask: torch.Tensor,
        question_ids: torch.Tensor,
        question_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode question and context using frozen XLM-R."""
        with torch.no_grad():
            context_out = self.condition_encoder(
                input_ids=context_ids, attention_mask=context_mask
            ).last_hidden_state
            question_out = self.condition_encoder(
                input_ids=question_ids, attention_mask=question_mask
            ).last_hidden_state

        context_emb = self.condition_proj(context_out)
        question_emb = self.condition_proj(question_out)
        condition = torch.cat([question_emb, context_emb], dim=1)
        condition_mask = torch.cat([question_mask, context_mask], dim=1)
        condition_mask = ~condition_mask.bool()
        return condition, condition_mask

    def forward(
        self,
        z_t: torch.Tensor,
        timesteps: torch.Tensor,
        context_ids: torch.Tensor,
        context_mask: torch.Tensor,
        question_ids: torch.Tensor,
        question_mask: torch.Tensor,
        z_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict noise from noisy latent."""
        batch_size, seq_len, _ = z_t.shape

        condition, condition_mask = self.encode_condition(
            context_ids, context_mask, question_ids, question_mask
        )

        t_emb = self.time_embed(timesteps)
        t_emb = self.time_mlp(t_emb)

        x = self.input_proj(z_t)
        x = x + self.pos_encoding[:, :seq_len, :]

        z_key_mask = ~z_mask.bool() if z_mask is not None else None

        for block in self.blocks:
            x = block(
                x, condition, t_emb, x_mask=z_key_mask, condition_mask=condition_mask
            )

        x = self.final_norm(x)
        noise_pred = self.output_proj(x)
        return noise_pred

    def forward_with_cached_condition(
        self,
        z_t: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor,
        condition_mask: torch.Tensor,
        z_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with pre-computed condition (for faster inference)."""
        batch_size, seq_len, _ = z_t.shape

        t_emb = self.time_embed(timesteps)
        t_emb = self.time_mlp(t_emb)

        x = self.input_proj(z_t)
        x = x + self.pos_encoding[:, :seq_len, :]

        z_key_mask = ~z_mask.bool() if z_mask is not None else None

        for block in self.blocks:
            x = block(
                x, condition, t_emb, x_mask=z_key_mask, condition_mask=condition_mask
            )

        x = self.final_norm(x)
        noise_pred = self.output_proj(x)
        return noise_pred
