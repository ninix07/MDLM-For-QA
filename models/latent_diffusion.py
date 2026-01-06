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
        false_negative_penalty_margin: float = 0.3,
        scaler: Optional[LatentScaler] = None,
        prediction_type: str = "epsilon",
        latent_seq_len: int = 16,  # BUG #9 FIX: Configurable, default 16 (was hardcoded 8)
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.latent_dim = latent_dim
        self.max_answer_len = max_answer_len
        self.null_ans_token = null_ans_token
        self.use_vae = use_vae
        self.false_negative_penalty_weight = false_negative_penalty_weight
        self.false_negative_penalty_margin = false_negative_penalty_margin
        self.scaler = scaler
        self.prediction_type = prediction_type

        # Ensure null answer token exists
        if null_ans_token not in tokenizer.get_vocab():
            tokenizer.add_special_tokens({"additional_special_tokens": [null_ans_token]})
        self.null_ans_token_id = tokenizer.convert_tokens_to_ids(null_ans_token)

        vocab_size = len(tokenizer)
        embedding_dim = 768  # BERT base hidden size (always 768, not d_model)

        # Determine actual latent dimension
        actual_latent_dim = latent_dim if use_vae else embedding_dim

        # BUG #9 FIX: Use configurable latent_seq_len instead of hardcoded 8
        self._latent_seq_len = latent_seq_len  # From parameter, default 16

        # Denoiser (initialized with default, will match VAE)
        self.denoiser = ConditionalDenoiser(
            latent_dim=actual_latent_dim,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            max_seq_len=self._latent_seq_len,  # Use internal variable
            condition_encoder=base_encoder,
        )

        # Extract pretrained embeddings from denoiser's condition encoder
        pretrained_embeddings = None
        if hasattr(self.denoiser.condition_encoder, "embeddings"):
            # BERT/RoBERTa style
            pretrained_embeddings = self.denoiser.condition_encoder.embeddings.word_embeddings
        elif hasattr(self.denoiser.condition_encoder, "model"):
            # XLM-R style (sometimes wrapped)
            pretrained_embeddings = self.denoiser.condition_encoder.model.embeddings.word_embeddings
        else:
            print("Warning: Could not extract pretrained embeddings from denoiser")

        # VAE or Embedding Bridge
        if use_vae:
            self.vae = SequenceVAE(
                vocab_size=len(tokenizer),
                embedding_dim=embedding_dim,  # Use actual BERT dim (768), not d_model
                latent_dim=latent_dim,
                num_layers=6,
                num_heads=12,
                dropout=dropout,
                pretrained_embeddings=pretrained_embeddings,  # CRITICAL: Use pretrained embeddings!
                pad_token_id=self.pad_token_id,
                latent_seq_len=self._latent_seq_len,
                max_seq_len=max_answer_len,
            )
            # FIX: Update latent_seq_len from VAE to ensure consistency
            self._latent_seq_len = self.vae.latent_seq_len
        else:
            self.vae = EmbeddingBridge(model_name=base_encoder)
            self.latent_dim = embedding_dim

        # Noise scheduler and diffusion
        self.scheduler = NoiseScheduler(
            num_timesteps=num_train_timesteps,
            schedule_type=schedule_type,
            device=None,  # Will be set when model is moved to device
        )
        self.diffusion = GaussianDiffusion(self.scheduler, prediction_type=prediction_type)

        # Sampler for inference
        self.sampler = CachedDDIMSampler(
            self.scheduler,
            num_inference_steps=num_inference_timesteps,
            prediction_type=prediction_type,
        )

        # Cache null answer embedding for threshold comparison
        self._null_ans_embedding = None

    def get_null_ans_embedding(self, device: torch.device) -> torch.Tensor:
        """Get or compute the null answer embedding."""
        if self._null_ans_embedding is None or self._null_ans_embedding.device != device:
            # BUG #29 FIX: Create properly padded null sequence instead of single token
            # The null answer in training data is [<s>, <NULL_ANS>, </s>, <pad>...]
            # We must encode the same structure to get a matching latent
            null_ids = torch.full((1, self.max_answer_len), self.pad_token_id, device=device)

            # Get special token IDs (handle different tokenizer conventions)
            bos_id = getattr(self.tokenizer, "bos_token_id", None) or getattr(
                self.tokenizer, "cls_token_id", None
            )
            eos_id = getattr(self.tokenizer, "eos_token_id", None) or getattr(
                self.tokenizer, "sep_token_id", None
            )

            # Build: [BOS/CLS, NULL_ANS, EOS/SEP, PAD, PAD, ...]
            if bos_id is not None:
                null_ids[0, 0] = bos_id
            null_ids[0, 1] = self.null_ans_token_id
            if eos_id is not None:
                null_ids[0, 2] = eos_id

            # Create attention mask: 1 for valid tokens, 0 for padding
            null_mask = torch.zeros((1, self.max_answer_len), device=device)
            null_mask[0, :3] = 1  # First 3 tokens are valid (BOS, NULL_ANS, EOS)

            if self.use_vae:
                with torch.no_grad():
                    null_emb, _ = self.vae.get_latent(null_ids, null_mask, use_mean=True)
            else:
                # EmbeddingBridge.encode returns 3D tensor directly
                null_emb = self.vae.encode(null_ids)
            # FIX: Ensure null embedding is float32 to avoid dtype issues
            self._null_ans_embedding = null_emb.squeeze(0).float()
        return self._null_ans_embedding

    def encode_answer(
        self, answer_ids: torch.Tensor, answer_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Encode answer tokens to latent space."""
        if self.use_vae:
            return self.vae.get_latent(answer_ids, answer_mask, use_mean=True)
        else:
            return self.vae.encode(answer_ids), answer_mask

    def decode_latent(
        self,
        z: torch.Tensor,
        return_tokens: bool = True,
    ) -> torch.Tensor:
        """Decode latent back to tokens."""
        if self.use_vae:
            # FIX: Convert to float32 to avoid dtype mismatch with model weights
            # (z may be in bfloat16 from mixed precision training/inference)
            z = z.float()

            # Use max_answer_len for proper sequence length
            hidden = self.vae.decode(z, target_len=self.max_answer_len)
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
        z, _ = self.encode_answer(answer_ids, answer_mask)

        # Decode latent to tokens
        tokens = self.decode_latent(z, return_tokens=True)

        # FIX Bug 2: Compare with ground truth answer_ids for null detection
        # The decoded tokens structure may differ, so we check original answer_ids
        is_null = answer_ids[:, 1] == self.null_ans_token_id

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
        recon_loss = torch.tensor(0.0, device=answer_ids.device)
        kl_loss = torch.tensor(0.0, device=answer_ids.device)
        vae_output = {}

        if self.use_vae:
            # Optimize: Skip reconstruction during diffusion training (when VAE is frozen)
            # train_vae_only=True -> Phase 1 (Warmup) -> compute_recon=True
            # train_vae_only=False -> Phase 2 (Diffusion) -> compute_recon=False
            compute_recon = train_vae_only

            vae_output = self.vae.loss(
                answer_ids, answer_mask, kl_weight=kl_weight, compute_recon=compute_recon
            )
            vae_loss = vae_output["loss"]
            recon_loss = vae_output.get("recon_loss", torch.tensor(0.0, device=answer_ids.device))
            kl_loss = vae_output.get("kl_loss", torch.tensor(0.0, device=answer_ids.device))

            if train_vae_only:
                ret = {
                    "loss": vae_loss,
                    "diffusion_loss": torch.tensor(0.0, device=answer_ids.device),
                    "vae_loss": vae_loss,
                    "recon_loss": recon_loss,
                    "kl_loss": kl_loss,
                    "penalty": torch.tensor(0.0, device=answer_ids.device),
                }
                if "mean" in vae_output:
                    ret["mean"] = vae_output["mean"]
                    ret["logvar"] = vae_output["logvar"]
                if "z" in vae_output:
                    ret["z"] = vae_output["z"]
                if "latent_mask" in vae_output:
                    ret["latent_mask"] = vae_output["latent_mask"]
                return ret

        # FIX: Reuse mean from VAE loss computation instead of re-encoding
        # This ensures consistency and avoids redundant computation
        if self.use_vae and "mean" in vae_output:
            # BUG #40 FIX: Add small noise to mean during training to match VAE distribution
            # The VAE decoder was trained on z = mean + eps*std (sampled), not just mean.
            # Training diffusion on pure mean creates a distribution mismatch.
            # We add scaled noise to approximate the VAE posterior sampling.
            z_0 = vae_output["mean"]
            if self.training and "logvar" in vae_output:
                # Add small noise proportional to VAE's learned variance
                # Scale factor 0.1 keeps noise small but present
                std = torch.exp(0.5 * vae_output["logvar"])
                noise = torch.randn_like(z_0)
                z_0 = z_0 + 0.1 * std * noise

            # Recompute latent_mask from answer_mask (downsample to latent_seq_len)
            latent_mask = F.adaptive_max_pool1d(
                answer_mask.unsqueeze(1).float(), self._latent_seq_len
            ).squeeze(1)
            latent_mask = (latent_mask > 0.5).float()
        else:
            # Fallback for non-VAE case
            z_0, latent_mask = self.encode_answer(answer_ids, answer_mask)

        # Requirement 2: Latent Calibration
        # Normalize latent if scaler is provided
        if self.scaler is not None:
            z_0 = self.scaler.transform(z_0, mask=latent_mask)

        # IMPLEMENT CONDITIONING DROPOUT
        # This is required for CFG to work at inference
        if self.training and cond_dropout_prob > 0:
            # Create a mask: 1 for dropout (null), 0 for keep
            drop_mask = torch.bernoulli(
                torch.full((context_ids.shape[0],), cond_dropout_prob, device=context_ids.device)
            ).bool()

            # FIX: Clone tensors before modifying to avoid mutating dataloader tensors
            context_ids = context_ids.clone()
            context_mask = context_mask.clone()
            question_ids = question_ids.clone()
            question_mask = question_mask.clone()

            # CRITICAL FIX: Must match inference unconditional signal!
            # During inference, uncond uses ALL PAD tokens with first token mask.
            # During training, we must do the SAME to avoid train/inference mismatch.
            #
            # The old approach (only zeroing mask) still let BERT see original tokens
            # via self-attention, causing the "unconditional" output to still
            # contain input-specific information.

            # 1. Replace ALL token IDs with PAD for dropped samples
            context_ids[drop_mask] = self.pad_token_id
            question_ids[drop_mask] = self.pad_token_id

            # 2. Set mask to 0 everywhere for dropped samples
            context_mask[drop_mask] = 0
            question_mask[drop_mask] = 0

            # 3. Set mask to 1 only for the first token to prevent NaN in attention
            context_mask[drop_mask, 0] = 1
            question_mask[drop_mask, 0] = 1

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
        # FIX: Check correct index for unanswerable token
        # XLM-R: [<s>, <NULL_ANS>, </s>, <pad>...] -> Index 1 is NULL_ANS
        # BERT: [CLS, <NULL_ANS>, SEP, <pad>...] -> Index 1 is NULL_ANS
        is_unanswerable = answer_ids[:, 1] == self.null_ans_token_id
        is_answerable = ~is_unanswerable

        # BUG #27 FIX: Force latent_mask to 1 for unanswerable questions
        # Without this, the model only learns to predict the first NULL token, leaving the
        # remaining latent positions (which should be PAD latents) completely unconstrained.
        # This causes garbage in the tail, corrupting null_similarity computation.
        # By forcing mask=1, the model learns to predict "silence" (PAD latents) for the entire sequence.
        if is_unanswerable.any():
            latent_mask = latent_mask.clone()  # Avoid mutating original
            latent_mask[is_unanswerable] = 1.0

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
            mask=latent_mask,
        )
        diffusion_loss = diff_output["loss"]

        penalty_loss = torch.tensor(0.0, device=diffusion_loss.device)

        if self.false_negative_penalty_weight > 0 and is_answerable.any():
            # Apply penalty to ALL answerable examples, not just a subset
            subset_indices = torch.where(is_answerable)[0]

            # Sample randomly if too many (for memory efficiency)
            if len(subset_indices) > 8:
                perm = torch.randperm(len(subset_indices))[:8]
                subset_indices = subset_indices[perm]

            # Retrieve necessary tensors for the subset from diff_output
            # We reuse the noise and t sampled in training_loss to avoid extra forward passes
            z_ans = z_0[subset_indices]
            t_ans = diff_output["t"][subset_indices]
            noise_ans = diff_output["noise"][subset_indices]
            model_output_ans = diff_output["model_output"][subset_indices]

            # BUG #33 FIX: Only compute penalty for low-t samples where pred_x0 is reliable
            # At high t (e.g., t=900), z_t is almost pure noise and pred_x0 is garbage
            # This adds noise to the gradient and prevents effective learning
            # BUG #39 FIX: Increased from 200 to 500 (25% of 2000 timesteps)
            # With 200, only 10% of samples qualified, often resulting in zero penalty samples
            low_t_threshold = 500  # Use samples with t < 500 (25% of timesteps)
            low_t_mask = t_ans < low_t_threshold

            if low_t_mask.sum() > 0:
                # Filter to only low-t samples
                z_ans = z_ans[low_t_mask]
                t_ans = t_ans[low_t_mask]
                noise_ans = noise_ans[low_t_mask]
                model_output_ans = model_output_ans[low_t_mask]

                # Reconstruct z_t (using the same noise/t as in training_loss)
                z_t_ans, _ = self.diffusion.q_sample(z_ans, t_ans, noise_ans)

                # Predict x_0 using the unified method
                pred_x0 = self.diffusion.predict_x0(z_t_ans, t_ans, model_output_ans)

                # STABLE NORMALIZATION: Clip pred_x0 to prevent inf values
                # Latents are typically unit variance, so [-20, 20] is a very safe bound
                # that prevents overflow during norm calculation.
                pred_x0 = torch.clamp(pred_x0, -20.0, 20.0)

                # Get null embedding
                null_emb = self.get_null_ans_embedding(z_0.device)
                if self.scaler is not None:
                    null_emb = self.scaler.transform(null_emb.unsqueeze(0)).squeeze(0)

                # FIX: Use Cosine Similarity on FLATTENED latents for stricter structural matching
                # Mean pooling destroys temporal information (e.g. [A, B] vs [B, A])
                # Flatten: [batch, seq, dim] -> [batch, seq*dim]
                pred_flat = pred_x0.reshape(pred_x0.shape[0], -1)
                null_flat = null_emb.flatten().unsqueeze(0)  # [1, seq*dim]

                # Normalize for cosine similarity
                pred_norm = F.normalize(pred_flat, p=2, dim=-1)
                null_norm = F.normalize(null_flat, p=2, dim=-1)

                # Cosine similarity: high value means prediction is structurally close to null
                cos_sim = (pred_norm * null_norm).sum(dim=-1)  # [batch]

                # Penalty: we want to push AWAY from null for answerable questions
                # Use hinge loss: Penalty = max(0, cos_sim - margin)
                # If cos_sim > margin (too close to null), apply penalty
                margin = self.false_negative_penalty_margin
                penalty = F.relu(cos_sim - margin).mean()

                penalty_loss = self.false_negative_penalty_weight * penalty

                # FINAL SAFETY CHECK: Zero out if NaN or Inf
                if torch.isnan(penalty_loss) or torch.isinf(penalty_loss):
                    penalty_loss = torch.tensor(0.0, device=diffusion_loss.device)

        # Combine losses (penalty already scaled by false_negative_penalty_weight in config)
        # FIX: Only add VAE loss during VAE warmup phase (train_vae_only=True)
        # During diffusion training (train_vae_only=False), VAE is frozen and vae_loss
        # only contains KL which would add noise to diffusion gradients
        if self.use_vae and train_vae_only:
            total_loss = diffusion_loss + (vae_loss * 0.1) + penalty_loss
        else:
            total_loss = diffusion_loss + penalty_loss

        ret = {
            "loss": total_loss,
            "diffusion_loss": diffusion_loss,
            "vae_loss": vae_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "penalty": penalty_loss,
        }

        if "z" in vae_output:
            ret["z"] = vae_output["z"]
        if "mean" in vae_output:
            ret["mean"] = vae_output["mean"]
            ret["logvar"] = vae_output["logvar"]

        # Ensure latent_mask is always returned if available
        # (It is computed as 'latent_mask' variable in this function)
        ret["latent_mask"] = latent_mask

        return ret

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

        # Sample from diffusion - use latent_seq_len property
        shape = (batch_size, self._latent_seq_len, self.latent_dim)

        # Temporarily override num_inference_steps if provided
        original_steps = self.sampler.num_inference_steps
        if num_inference_steps is not None:
            self.sampler.num_inference_steps = num_inference_steps

        try:
            # PREPARE NULL CONDITIONING FOR PASS 2
            uncond_context_ids = torch.full_like(context_ids, self.pad_token_id)
            # Match training logic: Mask all except first token
            uncond_context_mask = torch.zeros_like(context_mask)
            uncond_context_mask[:, 0] = 1

            uncond_question_ids = torch.full_like(question_ids, self.pad_token_id)
            uncond_question_mask = torch.zeros_like(question_mask)
            uncond_question_mask[:, 0] = 1

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
        null_raw = self.get_null_ans_embedding(device)  # [seq_len, dim]

        # Transform the null reference into the diffusion space
        if self.scaler is not None:
            # Scaler expects [batch, seq, dim]
            null_norm_ref = self.scaler.transform(null_raw.unsqueeze(0)).squeeze(0)
        else:
            null_norm_ref = null_raw

        # Compare normalized generated z_0 with normalized reference
        # FIX: Use FLATTENED cosine similarity to match training penalty logic
        # z_0: [batch, seq, dim] -> [batch, seq*dim]
        # null_norm_ref: [seq, dim] -> [1, seq*dim]

        batch_size = z_0.shape[0]
        # FIX: Convert to float32 for consistent comparison (null_norm_ref is float32)
        z_flat = z_0.float().reshape(batch_size, -1)

        # null_norm_ref comes from get_null_ans_embedding which returns [seq, dim]
        # We flatten it and unsqueeze for broadcasting
        null_flat = null_norm_ref.flatten().unsqueeze(0)

        z_vec = F.normalize(z_flat, p=2, dim=-1)
        n_vec = F.normalize(null_flat, p=2, dim=-1)

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
        batch_size = tokens.shape[0]
        texts = [""] * batch_size

        # Identify non-null indices
        active_indices = torch.where(~is_null)[0]
        if len(active_indices) == 0:
            return texts

        active_tokens = tokens[active_indices]

        # 2. THE EOS TRICK: Find the first occurrence of a Stop Token
        # This acts as a 'Wall' that stops the decoder from reading gibberish.
        sep_id = self.tokenizer.sep_token_id
        pad_id = self.tokenizer.pad_token_id

        # Convert to list for easier processing if needed, but batch_decode is better
        # We still need to truncate at the first stop token for each sequence
        # FIX Bug 11: Skip BOS/CLS token at the start (index 0)
        truncated_tokens = []
        for i in range(len(active_tokens)):
            row = active_tokens[i].tolist()
            # Skip the first token (BOS/CLS)
            row = row[1:] if len(row) > 1 else row
            stop_idx = len(row)
            for idx, tid in enumerate(row):
                # BUG #28 FIX: Add null_ans_token_id as a stop token
                # This prevents garbage after <NULL_ANS> from being decoded
                if tid == sep_id or tid == pad_id or tid == self.null_ans_token_id:
                    stop_idx = idx
                    break
            truncated_tokens.append(row[:stop_idx])

        # 4. Decode only the valid tokens into clean strings
        decoded_texts = self.tokenizer.batch_decode(
            truncated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        # Fill in the results
        for idx, text in zip(active_indices.tolist(), decoded_texts):
            texts[idx] = text.strip()

        return texts

    def to(self, device: torch.device):
        """Move model and scheduler to device with optimization."""
        super().to(device)
        # Move scheduler tensors to device efficiently
        self.scheduler.to(device)

        if self.scaler is not None:
            self.scaler.to(device)
        return self
