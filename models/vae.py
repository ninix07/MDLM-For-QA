"""
Variational Autoencoder (VAE) for text latent space.

This module provides two approaches:
- Approach A: Simple embedding bridge using frozen XLM-R embeddings
- Approach B: Full VAE with encoder-decoder for smoother latent space
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig


class EmbeddingBridge(nn.Module):
    """
    Approach A: Simple embedding bridge.

    Uses frozen XLM-RoBERTa embeddings to convert tokens to continuous vectors.
    The latent space is simply the embedding space.
    """

    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        freeze_embeddings: bool = True,
    ):
        super().__init__()

        # Load model for embeddings (supports any HuggingFace model)
        self.model = AutoModel.from_pretrained(model_name)
        self.embedding_dim = self.model.config.hidden_size

        if freeze_embeddings:
            for param in self.model.embeddings.parameters():
                param.requires_grad = False

    def get_embeddings(self) -> nn.Embedding:
        """Get the word embedding layer."""
        return self.model.embeddings.word_embeddings

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode token IDs to embeddings.

        Args:
            input_ids: [batch_size, seq_len]

        Returns:
            embeddings: [batch_size, seq_len, embedding_dim]
        """
        return self.model.embeddings.word_embeddings(input_ids)

    def decode(
        self,
        latent: torch.Tensor,
        return_logits: bool = False,
    ) -> torch.Tensor:
        """
        Decode latent vectors back to token IDs via nearest neighbor.

        Args:
            latent: [batch_size, seq_len, embedding_dim]
            return_logits: If True, return similarity scores instead of token IDs

        Returns:
            token_ids: [batch_size, seq_len] or logits [batch_size, seq_len, vocab_size]
        """
        # Get embedding matrix
        embedding_weight = (
            self.model.embeddings.word_embeddings.weight
        )  # [vocab_size, dim]

        # Normalize for cosine similarity
        latent_norm = F.normalize(latent, p=2, dim=-1)
        embed_norm = F.normalize(embedding_weight, p=2, dim=-1)

        # Compute similarity: [batch, seq_len, vocab_size]
        similarity = torch.matmul(latent_norm, embed_norm.T)

        if return_logits:
            return similarity

        # Get nearest neighbor
        token_ids = similarity.argmax(dim=-1)
        return token_ids




class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        return x + self.pe[:, : x.size(1), :]


class SequenceVAE(nn.Module):
    """
    Sequence-level VAE that preserves sequence structure in latent space.

    Unlike TextVAE which compresses to a single vector, this maintains
    the sequence dimension, making it more suitable for diffusion.

    Latent shape: [batch_size, seq_len, latent_dim]
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 768,
        latent_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        pretrained_embeddings: Optional[nn.Embedding] = None,
        pad_token_id: int = 0,
        latent_seq_len: int = 8,
    ):
        super().__init__()
       
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.pad_token_id = pad_token_id
        self.latent_seq_len = latent_seq_len

        # Embeddings
        # IMPORTANT: Create our own embedding layer and COPY weights, don't share reference
        # Sharing reference causes dtype mismatch with AMP (bfloat16 vs float32)
        # because the pretrained embeddings belong to the frozen condition encoder
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        if pretrained_embeddings is not None:
            with torch.no_grad():
                # Handle vocab size mismatch (e.g., when special tokens like <NULL_ANS> are added)
                pretrained_vocab_size = pretrained_embeddings.weight.shape[0]
                copy_size = min(vocab_size, pretrained_vocab_size)
                self.embeddings.weight[:copy_size].copy_(pretrained_embeddings.weight[:copy_size])
                # Any new tokens (e.g., <NULL_ANS>) keep their random initialization

        # Encoder: embedding_dim -> latent_dim
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)

        # Latent projections (per-position)
        self.to_mean = nn.Linear(embedding_dim, latent_dim)
        self.to_logvar = nn.Linear(embedding_dim, latent_dim)

        # Decoder: latent_dim -> embedding_dim
        self.from_latent = nn.Linear(latent_dim, embedding_dim)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)

        # No separate output projection - use embedding weights (weight tying)
        # This saves ~200M parameters for XLM-R vocab
        
        # Add output normalization to fix semantic mismatch
        self.output_norm = nn.LayerNorm(embedding_dim)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Encode to sequence of latent vectors.

        Returns:
            z: [batch, latent_seq_len, latent_dim]
            mean: [batch, latent_seq_len, latent_dim]
            logvar: [batch, latent_seq_len, latent_dim]
            latent_mask: [batch, latent_seq_len] or None
        """
        # Get embeddings
        x = self.embeddings(input_ids)

        # Encode
        if attention_mask is not None:
            src_key_padding_mask = ~attention_mask.bool()
        else:
            src_key_padding_mask = None

        encoded = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        # Get latent distribution (per position)
        # Apply structural bottleneck: average pool sequence across time
        # [batch, 64, hidden] -> [batch, 8, hidden]
        encoded_pooled = F.adaptive_avg_pool1d(encoded.transpose(1, 2), self.latent_seq_len).transpose(1, 2)
        
        # 2. Project to latent distribution
        mean = self.to_mean(encoded_pooled)
        logvar = self.to_logvar(encoded_pooled)
        
        # Clamp logvar to prevent instability
        # Relaxed to [-5, 2] to force higher variance (std min 0.08)
        logvar = torch.clamp(logvar, -5, 2)

        # Reparameterization
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std

        # 3. Downsample mask if provided [batch, 64] -> [batch, 8]
        # Use max pooling: if any token in the pool is NOT padding, the latent is valid.
        latent_mask = None
        if attention_mask is not None:
            # attention_mask: 1 for valid, 0 for pad
            # F.adaptive_max_pool1d expects B, C, L
            latent_mask = F.adaptive_max_pool1d(attention_mask.unsqueeze(1).float(), self.latent_seq_len).squeeze(1)
            # Threshold to keep it binary (max pool on float might be slightly > 0)
            latent_mask = (latent_mask > 0.5).float()

        # Requirement 1: VAE Robustness & Local Isometry
        # Inject additional noise during training to bridge gap with diffusion
        if self.training:
            # Increased noise to 0.2 to further break identity mapping
            noise_injection = torch.randn_like(z) * 0.2
            z = z + noise_injection

        return z, mean, logvar, latent_mask

    def decode(self, z: torch.Tensor, target_len: int = 64) -> torch.Tensor:
        """
        Decode latent sequence to hidden states.

        Args:
            z: [batch, latent_seq_len, latent_dim]
            target_len: Target sequence length to expand to (default 64)

        Returns:
            hidden: [batch, target_len, embedding_dim]
        """
        # 1. Project to embedding space [batch, 8, latent_dim] -> [batch, 8, embedding_dim]
        x = self.from_latent(z)
        
        # 2. Expand sequence length [batch, 8, E] -> [batch, 64, E]
        # This breaks the 1:1 token mapping
        if x.shape[1] != target_len:
            x = F.interpolate(x.transpose(1, 2), size=target_len, mode='linear', align_corners=False).transpose(1, 2)
            
        # 3. Decode with bidirectional attention
        decoded = self.decoder(x)
        # Apply output normalization to fix semantic mismatch
        decoded = self.output_norm(decoded)
        return decoded

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        z, mean, logvar, latent_mask = self.encode(input_ids, attention_mask)
        logits = self.decode(z)

        return {
            "logits": logits,
            "z": z,
            "mean": mean,
            "logvar": logvar,
            "latent_mask": latent_mask,
        }

    def loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kl_weight: float = 0.1,
    ) -> Dict[str, torch.Tensor]:
        z, mean, logvar, latent_mask = self.encode(input_ids.contiguous(), attention_mask.contiguous())
        decoded = self.decode(z)

        # Memory-efficient reconstruction loss using chunked computation
        # Instead of computing full [batch, seq, vocab] logits, compute loss per position
        batch_size, seq_len, hidden_dim = decoded.shape
        embed_weight = self.embeddings.weight  # [vocab_size, embedding_dim]

        # Compute loss in chunks to save memory
        recon_loss = 0.0
        chunk_size = 8  # Process 8 positions at a time
        num_chunks = (seq_len + chunk_size - 1) // chunk_size

        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, seq_len)

            # Get chunk of decoded hidden states [batch, chunk, hidden]
            chunk_hidden = decoded[:, start_idx:end_idx, :]
            chunk_targets = input_ids[:, start_idx:end_idx]

            # Compute logits for this chunk: [batch, chunk, vocab]
            chunk_logits = torch.matmul(chunk_hidden, embed_weight.T)

            # Compute loss for this chunk
            chunk_loss = F.cross_entropy(
                chunk_logits.reshape(-1, self.vocab_size),
                chunk_targets.reshape(-1),
                reduction="sum",
                ignore_index=self.pad_token_id
            )
            recon_loss = recon_loss + chunk_loss

        # Average over valid tokens only (not padding)
        num_valid_tokens = (input_ids != self.pad_token_id).sum()
        recon_loss = recon_loss / num_valid_tokens.clamp(min=1.0)

        # KL loss (per position, then averaged)
        # REMOVED Free Bits clamp to see true KL collapse and force model to work harder
        kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())

        total_loss = recon_loss + kl_weight * kl_loss

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "z": z,
            "mean": mean,
            "logvar": logvar,
        }

    def get_latent(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_mean: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get latent for diffusion (sequence-level)."""
        z, mean, logvar, latent_mask = self.encode(input_ids, attention_mask)
        return (mean, latent_mask) if use_mean else (z, latent_mask)
