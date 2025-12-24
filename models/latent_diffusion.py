"""
Main Latent Diffusion Model combining VAE, Denoiser, and Diffusion.
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import XLMRobertaTokenizer

from .vae import EmbeddingBridge, SequenceVAE
from .denoiser import ConditionalDenoiser
from .diffusion import NoiseScheduler, GaussianDiffusion
from .sampler import DDIMSampler, CachedDDIMSampler
from .scaler import LatentScaler


class LatentDiffusionQA(nn.Module):
    """
    Latent Diffusion Model for Question Answering (SQuAD 2.0).

    Combines:
    - VAE/Embedding bridge for text-to-latent conversion
    - Conditional Denoiser for noise prediction
    - Diffusion process for training and sampling
    - Null answer detection for unanswerable questions
    """

    def __init__(
        self,
        tokenizer: XLMRobertaTokenizer,
        latent_dim: int = 256,
        d_model: int = 768,
        num_layers: int = 6,
        num_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        max_answer_len: int = 64,
        num_train_timesteps: int = 1000,
        num_inference_timesteps: int = 50,
        schedule_type: str = "cosine",
        use_vae: bool = True,
        null_ans_token: str = "<NULL_ANS>",
        base_encoder: str = "xlm-roberta-base",
        false_negative_penalty_weight: float = 1.0,
        scaler: Optional[LatentScaler] = None,
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.latent_dim = latent_dim
        self.max_answer_len = max_answer_len
        self.null_ans_token = null_ans_token
        self.null_ans_token = null_ans_token
        self.use_vae = use_vae
        self.false_negative_penalty_weight = false_negative_penalty_weight
        self.scaler = scaler

        # Ensure null answer token exists
        if null_ans_token not in tokenizer.get_vocab():
            tokenizer.add_special_tokens(
                {"additional_special_tokens": [null_ans_token]}
            )
        self.null_ans_token_id = tokenizer.convert_tokens_to_ids(null_ans_token)

        vocab_size = len(tokenizer)
        embedding_dim = 768  # XLM-R base hidden size

        # VAE or Embedding Bridge
        if use_vae:
            self.vae = SequenceVAE(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                latent_dim=latent_dim,
                num_layers=4,
                num_heads=8,
                dropout=dropout,
            )
            actual_latent_dim = latent_dim
        else:
            self.vae = EmbeddingBridge(model_name=base_encoder)
            actual_latent_dim = embedding_dim
            self.latent_dim = embedding_dim

        # Denoiser
        self.denoiser = ConditionalDenoiser(
            latent_dim=actual_latent_dim,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            max_seq_len=max_answer_len,
            condition_encoder=base_encoder,
        )

        # Noise scheduler and diffusion
        self.scheduler = NoiseScheduler(
            num_timesteps=num_train_timesteps,
            schedule_type=schedule_type,
        )
        self.diffusion = GaussianDiffusion(self.scheduler)

        # Sampler for inference
        self.sampler = CachedDDIMSampler(
            self.scheduler,
            num_inference_steps=num_inference_timesteps,
        )

        # Cache null answer embedding for threshold comparison
        self._null_ans_embedding = None

    def get_null_ans_embedding(self, device: torch.device) -> torch.Tensor:
        """Get or compute the null answer embedding."""
        if (
            self._null_ans_embedding is None
            or self._null_ans_embedding.device != device
        ):
            null_ids = torch.tensor([[self.null_ans_token_id]], device=device)
            if self.use_vae:
                with torch.no_grad():
                    null_emb = self.vae.get_latent(null_ids, use_mean=True)
            else:
                null_emb = self.vae.encode(null_ids)
            self._null_ans_embedding = null_emb.squeeze(0)
        return self._null_ans_embedding

    def encode_answer(
        self,
        answer_ids: torch.Tensor,
        answer_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode answer tokens to latent space."""
        if self.use_vae:
            return self.vae.get_latent(answer_ids, answer_mask, use_mean=True)
        else:
            return self.vae.encode(answer_ids)

    def decode_latent(
        self,
        z: torch.Tensor,
        return_tokens: bool = True,
    ) -> torch.Tensor:
        """Decode latent back to tokens."""
        if self.use_vae:
            hidden = self.vae.decode(z)
            # Project hidden to logits using embedding weight tying
            embed_weight = self.vae.embeddings.weight
            logits = torch.matmul(hidden, embed_weight.T)
            
            if return_tokens:
                return logits.argmax(dim=-1)
            return logits
        else:
            return self.vae.decode(z, return_logits=not return_tokens)

    def vae_reconstruct(
        self,
        answer_ids: torch.Tensor,
        answer_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Reconstruct answers using only the VAE."""
        # Encode to latent
        z = self.encode_answer(answer_ids, answer_mask)
        
        # Decode latent to tokens
        tokens = self.decode_latent(z, return_tokens=True)
        
        # Identify null answers (first token is null_ans_token_id)
        is_null = (tokens[:, 0] == self.null_ans_token_id)
        
        return {
            "tokens": tokens,
            "is_null": is_null,
        }

    def forward(
        self,
        context_ids: torch.Tensor,
        context_mask: torch.Tensor,
        question_ids: torch.Tensor,
        question_mask: torch.Tensor,
        answer_ids: torch.Tensor,
        answer_mask: torch.Tensor,
        train_vae_only: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass.

        Returns dict with:
            - loss: Total training loss
            - diffusion_loss: MSE loss on noise prediction
            - vae_loss: VAE reconstruction + KL loss (if using VAE)
        """
        vae_loss = torch.tensor(0.0, device=answer_ids.device)
        if train_vae_only:
            vae_output = self.vae.loss(answer_ids, answer_mask, kl_weight=0.1)
            vae_loss = vae_output["loss"]
            return {
                "loss": vae_loss,
                "diffusion_loss": torch.tensor(0.0, device=answer_ids.device),
                "vae_loss": vae_loss,
                "penalty": torch.tensor(0.0, device=answer_ids.device),
            }

        # Encode answer to latent
        z_0 = self.encode_answer(answer_ids, answer_mask)

        # Requirement 2: Latent Calibration
        # Normalize latent if scaler is provided
        if self.scaler is not None:
            z_0 = self.scaler.transform(z_0)

        # Diffusion training loss
        condition_kwargs = {
            "context_ids": context_ids,
            "context_mask": context_mask,
            "question_ids": question_ids,
            "question_mask": question_mask,
        }

        # Requirement 3: SQuAD 2.0 'Null-Sink' Logic
        # For unanswerable questions, target x_0 is the VAE-encoded <NULL_ANS> sequence
        if self.use_vae:
            # Get null answer embedding (latent)
            null_ids = torch.tensor([[self.null_ans_token_id]], device=answer_ids.device)
            # We need a sequence of null latents matching the answer length
            # But wait, the VAE produces a sequence.
            # If the answer is unanswerable, the input `answer_ids` should already be [NULL_ANS, PAD, PAD...]
            # So `z_0` computed from `answer_ids` is ALREADY the null latent sequence!
            # The requirement says: "Ensure that for unanswerable questions, the Diffusion model's target x_0 is the VAE-encoded representation of the padded <NULL_ANS> sequence."
            # Since the data loader prepares unanswerable questions as [NULL_ANS] + [PAD]..., 
            # z_0 is already correct.
            # However, we need to ensure we are NOT using the penalty heuristic.
            pass

        # Identify answerable questions (for logging or other logic if needed, but not for penalty)
        is_unanswerable = (answer_ids[:, 0] == self.null_ans_token_id)
        is_answerable = ~is_unanswerable

        # Remove penalty logic
        # We pass is_answerable just for info if needed, but diffusion loss handles it naturally
        
        diff_output = self.diffusion.training_loss(
            self.denoiser, 
            z_0, 
            condition_kwargs, 
            mask=answer_mask,
            # null_embedding=null_emb,  # Removed
            # is_answerable=is_answerable, # Removed
            # penalty_weight=... # Removed
        )
        diffusion_loss = diff_output["loss"]
        # penalty_loss = diff_output.get("penalty", torch.tensor(0.0)) # Removed
        penalty_loss = torch.tensor(0.0, device=diffusion_loss.device)

        if self.use_vae:
            total_loss = diffusion_loss + 0.1 * vae_loss
        else:
            total_loss = diffusion_loss

        return {
            "loss": total_loss,
            "diffusion_loss": diffusion_loss,
            "vae_loss": vae_loss,
            "penalty": penalty_loss
        }

    @torch.no_grad()
    def generate(
        self,
        context_ids: torch.Tensor,
        context_mask: torch.Tensor,
        question_ids: torch.Tensor,
        question_mask: torch.Tensor,
        null_threshold: float = 0.7,
        show_progress: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate answer for given context and question.

        Returns:
            - tokens: Predicted token IDs [batch, seq_len]
            - is_null: Boolean indicating unanswerable [batch]
            - null_similarity: Cosine similarity to null answer [batch]
            - latent: Generated latent representation
        """
        batch_size = context_ids.shape[0]
        device = context_ids.device

        # Sample from diffusion
        shape = (batch_size, self.max_answer_len, self.latent_dim)
        z_0 = self.sampler.sample(
            self.denoiser,
            shape,
            context_ids,
            context_mask,
            question_ids,
            question_mask,
            device,
            show_progress=show_progress,
        )

        # Requirement 2: Latent Calibration
        # Denormalize latent if scaler is provided
        if self.scaler is not None:
            z_0 = self.scaler.inverse_transform(z_0)

        # Decode to tokens
        tokens = self.decode_latent(z_0, return_tokens=True)

        # Check for null answer
        null_emb = self.get_null_ans_embedding(device)

        # Pool latent for comparison (mean over sequence)
        z_sink = z_0[:, 0, :] 
        null_sink = null_emb[0, :] if null_emb.dim() > 1 else null_emb

        z_norm = F.normalize(z_sink, p=2, dim=-1)
        null_norm = F.normalize(null_sink.unsqueeze(0), p=2, dim=-1)
        null_similarity = (z_norm * null_norm).sum(dim=-1)

        is_null = null_similarity > null_threshold

        return {
            "tokens": tokens,
            "is_null": is_null,
            "null_similarity": null_similarity,
            "latent": z_0,
        }

    def decode_tokens_to_text(
        self,
        tokens: torch.Tensor,
        is_null: torch.Tensor,
    ) -> list:
        """Convert token IDs to text strings."""
        texts = []
        for i in range(tokens.shape[0]):
            if is_null[i]:
                texts.append("")
            else:
                token_ids = tokens[i].tolist()
                # Remove padding and special tokens
                text = self.tokenizer.decode(
                    token_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                texts.append(text.strip())
        return texts
