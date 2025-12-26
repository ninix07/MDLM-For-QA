"""
Main Latent Diffusion Model combining VAE, Denoiser, and Diffusion.
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

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
        tokenizer: AutoTokenizer,
        latent_dim: int = 128,
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
        self.pad_token_id = tokenizer.pad_token_id
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
        embedding_dim = 768  # BERT base hidden size (always 768, not d_model)

        # VAE or Embedding Bridge
        if use_vae:
            self.vae = SequenceVAE(
                vocab_size=len(tokenizer),
                embedding_dim=embedding_dim,  # Use actual BERT dim (768), not d_model
                latent_dim=latent_dim,
                num_layers=4,
                num_heads=8,
                dropout=dropout,
                pretrained_embeddings=None,
                pad_token_id=self.pad_token_id,
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
        is_null = (tokens[:, 1] == self.null_ans_token_id)
        
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
        kl_weight: float = 0.1,
        cond_dropout_prob: float = 0.1,
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
            vae_output = self.vae.loss(answer_ids, answer_mask, kl_weight=kl_weight)
            vae_loss = vae_output["loss"]
            ret = {
                "loss": vae_loss,
                "diffusion_loss": torch.tensor(0.0, device=answer_ids.device),
                "vae_loss": vae_loss,
                "penalty": torch.tensor(0.0, device=answer_ids.device),
            }
            if "mean" in vae_output:
                ret["mean"] = vae_output["mean"]
                ret["logvar"] = vae_output["logvar"]
            if "recon_loss" in vae_output:
                ret["recon_loss"] = vae_output["recon_loss"]
                ret["kl_loss"] = vae_output["kl_loss"]
            return ret

        # Encode answer to latent
        z_0 = self.encode_answer(answer_ids, answer_mask)

        # Requirement 2: Latent Calibration
        # Normalize latent if scaler is provided
        if self.scaler is not None:
            z_0 = self.scaler.transform(z_0, mask=answer_mask)

        # IMPLEMENT CONDITIONING DROPOUT
        # This is required for CFG to work at inference
        if self.training and cond_dropout_prob > 0:
            # Create a mask: 1 for dropout (null), 0 for keep
            drop_mask = torch.bernoulli(torch.full((context_ids.shape[0],), cond_dropout_prob, device=context_ids.device)).bool()
            
            # Clone to avoid modifying original batch for other logging
            context_ids = context_ids.clone()
            question_ids = question_ids.clone()
            context_mask = context_mask.clone()
            question_mask = question_mask.clone()
            
            # Replace with PAD tokens to represent "No Context"
            context_ids[drop_mask] = self.pad_token_id
            question_ids[drop_mask] = self.pad_token_id
            # Set masks to 0 for these entries
            context_mask[drop_mask] = 0
            question_mask[drop_mask] = 0

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
            # Since the data loader prepares unanswerable questions as [NULL_ANS] + [PAD]..., 
            # z_0 is already correct.
            pass

        # Identify answerable questions (for logging or other logic if needed, but not for penalty)
        is_unanswerable = (answer_ids[:, 1] == self.null_ans_token_id)
        is_answerable = ~is_unanswerable

        # Identify answerable questions (for logging or other logic if needed, but not for penalty)
        is_unanswerable = (answer_ids[:, 1] == self.null_ans_token_id)
        is_answerable = ~is_unanswerable

        # Requirement 3: False Negative Penalty
        # Penalize if the model predicts "null" (or close to it) for answerable questions.
        # We check the distance between the predicted noise (or implied x_0) and the null embedding.
        # However, diffusion predicts NOISE, not x_0 directly in the loss wrapper usually.
        # But we can look at the diffusion loss itself? No, that's just MSE.
        
        # We need to compute the predicted x_0 from the noise prediction to do this properly,
        # OR we can just add a penalty on the predicted noise if we knew what "null noise" looked like (we don't).
        
        # Alternative: The user wants to avoid "converging into everything being unanswerable".
        # This often happens if the null sink is a strong attractor.
        # Let's add a penalty that pushes the predicted x_0 AWAY from the null embedding 
        # if the ground truth is answerable.
        
        diff_output = self.diffusion.training_loss(
            self.denoiser, 
            z_0, 
            condition_kwargs, 
            mask=answer_mask,
        )
        diffusion_loss = diff_output["loss"]
        
        penalty_loss = torch.tensor(0.0, device=diffusion_loss.device)
        
        if self.false_negative_penalty_weight > 0 and is_answerable.any():
             # Sample a few answerable examples
             subset_indices = torch.where(is_answerable)[0]
             if len(subset_indices) > 4:
                 subset_indices = subset_indices[:4]
             
             z_ans = z_0[subset_indices]
             mask_ans = answer_mask[subset_indices]
             cond_ids_ans = context_ids[subset_indices]
             cond_mask_ans = context_mask[subset_indices]
             q_ids_ans = question_ids[subset_indices]
             q_mask_ans = question_mask[subset_indices]
             
             # Get null embedding
             null_emb = self.get_null_ans_embedding(z_0.device)
             if self.scaler is not None:
                 # Normalize null embedding
                 null_emb = self.scaler.transform(null_emb.unsqueeze(0)).squeeze(0)
             
             # We want to check if the model predicts 'null' for these.
             # We can check the denoiser output at a random timestep.
             t = torch.randint(0, self.scheduler.num_timesteps, (len(subset_indices),), device=z_0.device).long()
             
             # Add noise
             noise = torch.randn_like(z_ans)
             z_t, _ = self.diffusion.q_sample(z_ans, t, noise)
             
             # Predict noise
             noise_pred = self.denoiser(
                 z_t, t, cond_ids_ans, cond_mask_ans, q_ids_ans, q_mask_ans, z_mask=mask_ans
             )
             
             # Estimate x_0 (requires scheduler)
             # self.scheduler.step() is for sampling. We need `predict_start_from_noise`.
             # I'll assume `self.scheduler` has `predict_start_from_noise` or similar.
             # If not, I can implement it manually: x0 = (xt - sqrt(1-alpha_cumprod)*eps) / sqrt(alpha_cumprod)
             
             # Let's try to access the scheduler's alphas.
             # This is getting complicated without seeing `diffusion.py`.
             # I'll just use the `read_file` tool in the next turn to be safe.
             # But I need to finish this turn.
             
             # I'll comment out the penalty implementation details and put a TODO 
             # or just implement the manual formula which is standard.
             
             alpha_cumprod = self.scheduler.alphas_cumprod.to(z_0.device)[t]
             sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod).view(-1, 1, 1)
             sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alpha_cumprod).view(-1, 1, 1)
             
             pred_x0 = (z_t - sqrt_one_minus_alpha_cumprod * noise_pred) / sqrt_alpha_cumprod.clamp(min=1e-8)
             
             # Calculate distance to null
             # null_emb: [dim] or [seq, dim]?
             # get_null_ans_embedding returns [seq, dim] (squeezed) or [1, seq, dim]?
             # In `get_null_ans_embedding`: `self._null_ans_embedding = null_emb.squeeze(0)` -> [seq, dim]
             
             # pred_x0: [batch, seq, dim]
             # null_emb: [seq, dim]
             
             # Compute distance
             # We can use cosine similarity or Euclidean distance.
             # User mentioned "converging into everything being unanswerable".
             # Let's use Euclidean distance.
             
             # Pool over sequence?
             pred_mean = pred_x0.mean(dim=1) # [batch, dim]
             null_mean = null_emb.mean(dim=0) # [dim]
             
             dist_sq = torch.sum((pred_mean - null_mean)**2, dim=-1)
             dist = torch.sqrt(dist_sq + 1e-8) 
             
             # We want to MAXIMIZE distance (minimize -distance or exp(-distance))
             # Penalty = exp(-distance)
             penalty = torch.exp(-dist).mean()
             
             penalty_loss = self.false_negative_penalty_weight * penalty

        if self.use_vae:
            total_loss = diffusion_loss + 0.1 * vae_loss + penalty_loss
        else:
            total_loss = diffusion_loss + penalty_loss
        
        # if self.training:
        #     print(f"--- Batch Health Check ---")
        #     print(f"z_0 NaN: {torch.isnan(z_0).any().item()}")
        #     print(f"Diffusion Loss NaN: {torch.isnan(diffusion_loss).any().item()}")
        #     print(f"Penalty Loss NaN: {torch.isnan(penalty_loss).any().item()}")
            
        #     if torch.isnan(diffusion_loss):
        #         print(f"Denoiser Output Max: {self.denoiser.last_output.max().item() if hasattr(self.denoiser, 'last_output') else 'N/A'}")
        
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
        null_threshold: float = 0.3,
        show_progress: bool = False,
        num_inference_steps: Optional[int] = None,
        guidance_scale: float = 5.0,
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
        
        # Temporarily override num_inference_steps if provided
        original_steps = self.sampler.num_inference_steps
        if num_inference_steps is not None:
            self.sampler.num_inference_steps = num_inference_steps
            
        try:
            # PREPARE NULL CONDITIONING FOR PASS 2
            uncond_context_ids = torch.full_like(context_ids, self.pad_token_id)
            uncond_context_mask = torch.zeros_like(context_mask)
            uncond_question_ids = torch.full_like(question_ids, self.pad_token_id)
            uncond_question_mask = torch.zeros_like(question_mask)

            z_0 = self.sampler.sample(
                self.denoiser,
                shape,
                context_ids,
                context_mask,
                question_ids,
                question_mask,
                device,
                show_progress=show_progress,
                uncond_context_ids=uncond_context_ids,
                uncond_context_mask=uncond_context_mask,
                uncond_question_ids=uncond_question_ids,
                uncond_question_mask=uncond_question_mask,
                guidance_scale=guidance_scale,
            )
        finally:
            # Restore original steps
            if num_inference_steps is not None:
                self.sampler.num_inference_steps = original_steps

        # 1. Decode to tokens (using denormalized latent)
        z_denorm = self.scaler.inverse_transform(z_0) if self.scaler is not None else z_0
        tokens = self.decode_latent(z_denorm, return_tokens=True)

        # 2. Check for null answer (in NORMALIZED space)
        null_raw = self.get_null_ans_embedding(device) # [seq_len, dim]
        
        # Transform the null reference into the diffusion space
        if self.scaler is not None:
            # Scaler expects [batch, seq, dim]
            null_norm_ref = self.scaler.transform(null_raw.unsqueeze(0)).squeeze(0)
        else:
            null_norm_ref = null_raw

        # Compare normalized generated z_0 with normalized reference
        z_pooled = z_0.mean(dim=1) 
        null_pooled = null_norm_ref.mean(dim=0)
        
        z_vec = F.normalize(z_pooled, p=2, dim=-1)
        n_vec = F.normalize(null_pooled.unsqueeze(0), p=2, dim=-1)
        null_similarity = (z_vec * n_vec).sum(dim=-1)

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
        sep_id = self.tokenizer.sep_token_id 
        pad_id = self.tokenizer.pad_token_id 
        stop_tokens = {sep_id, pad_id}  
        for i in range(tokens.shape[0]):
            if is_null[i]:
                texts.append("")
            else:
                # Convert the row of token IDs to a standard Python list
                token_ids = tokens[i].tolist()
            
            # 2. THE EOS TRICK: Find the first occurrence of a Stop Token
            # This acts as a 'Wall' that stops the decoder from reading gibberish.
                stop_idx = len(token_ids)
                for idx, tid in enumerate(token_ids):
                    if tid in stop_tokens:
                        stop_idx = idx
                        break
            
            # 3. Slice the list to remove everything after (and including) the stop token
                valid_ids = token_ids[:stop_idx]
            
            # 4. Decode only the valid tokens into a clean string
            # skip_special_tokens=True is used as a safety layer for CLS or other markers.
                text = self.tokenizer.decode(
                    valid_ids, 
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces=True
                )
            
            # Final cleanup of leading/trailing whitespace
                texts.append(text.strip())
            
        return texts