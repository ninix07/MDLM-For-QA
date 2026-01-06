"""
Latent Scaler for normalizing VAE latents to N(0, 1).
"""

import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Optional


class LatentScaler:
    """
    Scales latents to have zero mean and unit variance.
    Required for Gaussian Diffusion which assumes N(0, 1) data distribution.

    Uses two-stage normalization:
    1. Per-dimension normalization (zero mean, unit variance per dim)
    2. Global scaling factor to ensure overall std ≈ 1.0
    """

    def __init__(self):
        self.mean = None
        self.std = None
        self.global_scale = None  # Additional global scaling factor
        self.device = None
        self._warned_unfitted = False

    @property
    def is_fitted(self) -> bool:
        """Check if scaler has been fitted."""
        return self.mean is not None and self.std is not None and self.global_scale is not None

    def fit(self, dataloader, vae_model, device):
        """
        Compute global mean and std of latents from dataloader.
        Optimized to keep data on GPU and reduce CPU-GPU transfers.
        """
        print("Fitting LatentScaler...")
        vae_model.eval()
        self.device = device

        # Use online statistics to avoid storing all latents in memory
        running_mean = torch.zeros(vae_model.latent_dim, device=device)
        running_var = torch.zeros(vae_model.latent_dim, device=device)
        total_count = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Computing latent stats"):
                answer_ids = batch["answer_input_ids"].to(device, non_blocking=True)
                answer_mask = batch["answer_attention_mask"].to(device, non_blocking=True)

                # Get mean latent (deterministic) and the downsampled latent mask
                z, latent_mask = vae_model.get_latent(answer_ids, answer_mask, use_mean=True)

                # Mask out padding and compute statistics online
                mask = latent_mask.bool()
                z_masked = z[mask]  # [valid_latents, latent_dim]

                # Online mean and variance calculation
                batch_count = z_masked.shape[0]
                batch_mean = z_masked.mean(dim=0)
                batch_var = ((z_masked - batch_mean) ** 2).mean(dim=0)

                # Update running statistics
                delta = batch_mean - running_mean
                new_count = total_count + batch_count

                running_mean = running_mean + delta * batch_count / new_count

                # Update variance using Welford's algorithm
                m_a = running_var * total_count
                m_b = batch_var * batch_count
                M2 = m_a + m_b + delta**2 * total_count * batch_count / new_count
                running_var = M2 / new_count

                total_count = new_count

        # Final statistics
        self.mean = running_mean
        self.std = torch.sqrt(running_var)

        # Avoid division by zero and handle collapsed dimensions
        self.std = torch.clamp(self.std, min=1e-3)

        # CRITICAL FIX: Compute global scaling factor
        # After per-dim normalization, compute the actual global std
        # to ensure the final output has std ≈ 1.0
        print("Computing global scaling factor...")
        global_std_sum = 0.0
        global_count = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Computing global scale"):
                answer_ids = batch["answer_input_ids"].to(device, non_blocking=True)
                answer_mask = batch["answer_attention_mask"].to(device, non_blocking=True)

                z, latent_mask = vae_model.get_latent(answer_ids, answer_mask, use_mean=True)

                # Apply per-dim normalization
                z_norm = (z - self.mean) / self.std

                # Only count valid (non-padded) tokens
                mask = latent_mask.bool()
                z_valid = z_norm[mask]

                if z_valid.numel() > 0:
                    # Accumulate variance (not std, to avoid sqrt issues)
                    global_std_sum += (z_valid**2).sum().item()
                    global_count += z_valid.numel()

        # Compute global std after per-dim normalization
        if global_count > 0:
            global_variance = global_std_sum / global_count
            self.global_scale = torch.tensor(global_variance**0.5, device=device)
        else:
            self.global_scale = torch.tensor(1.0, device=device)

        # Clamp to reasonable range
        self.global_scale = torch.clamp(self.global_scale, min=0.1, max=10.0)

        print(
            f"LatentScaler fitted: mean_norm={self.mean.norm().item():.4f}, "
            f"std_norm={self.std.norm().item():.4f}, global_scale={self.global_scale.item():.4f}"
        )

    def transform(self, z, mask: Optional[torch.Tensor] = None):
        """
        Normalize z to N(0, 1) using two-stage normalization:
        1. Per-dimension: (z - mean) / std
        2. Global: divide by global_scale to ensure std ≈ 1.0

        Note: We do NOT zero out padded positions here because:
        1. The normalized mean is 0, so padding naturally centers at 0
        2. The denoiser's attention mask properly ignores padded positions
        3. Zeroing would create an artificial attractor at z=0 for short sequences
        """
        if self.mean is None:
            if not self._warned_unfitted:
                print(
                    "Warning: LatentScaler.transform() called before fit(). Returning unchanged z."
                )
                self._warned_unfitted = True
            return z

        # FIX: Cast to match input dtype (handles bfloat16 from mixed precision)
        mean = self.mean.to(dtype=z.dtype)
        std = self.std.to(dtype=z.dtype)
        global_scale = self.global_scale.to(dtype=z.dtype)

        # Two-stage normalization
        z_norm = (z - mean) / std  # Per-dim normalization
        z_norm = z_norm / global_scale  # Global scaling to ensure std ≈ 1.0

        # Clipping to prevent extreme outliers from destabilizing diffusion
        z_norm = torch.clamp(z_norm, min=-5.0, max=5.0)

        return z_norm

    def inverse_transform(self, z_norm, mask: Optional[torch.Tensor] = None):
        """
        Denormalize z: reverse the two-stage normalization.
        1. Multiply by global_scale
        2. Multiply by std and add mean

        Note: We do NOT zero out padded positions here either.
        The VAE decoder and tokenizer handle padding appropriately.
        """
        if self.mean is None:
            return z_norm

        # FIX: Cast to match input dtype (handles bfloat16 from mixed precision)
        mean = self.mean.to(dtype=z_norm.dtype)
        std = self.std.to(dtype=z_norm.dtype)
        global_scale = self.global_scale.to(dtype=z_norm.dtype)

        # Reverse two-stage normalization
        z = z_norm * global_scale  # Reverse global scaling
        z = z * std + mean  # Reverse per-dim normalization

        return z

    def to(self, device):
        """Move stats to device."""
        self.device = device
        if self.mean is not None:
            self.mean = self.mean.to(device)
        if self.std is not None:
            self.std = self.std.to(device)
        if self.global_scale is not None:
            self.global_scale = self.global_scale.to(device)
        return self
