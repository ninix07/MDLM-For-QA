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

    def fit(self, dataloader, vae_model, device, noise_scale: float = 0.1):
        """
        Compute global mean and std of latents from dataloader in a SINGLE PASS.
        
        ARCHITECTURAL FIX: Previously used two passes over the dataset.
        Now computes per-dimension stats AND global scale simultaneously using
        an extended Welford's algorithm.

        Args:
            dataloader: DataLoader with training data
            vae_model: The VAE model to encode latents
            device: Device to run on
            noise_scale: Scale of noise to add (matches training distribution)
        """
        print("Fitting LatentScaler (single-pass)...")
        vae_model.eval()
        self.device = device

        # Per-dimension online statistics (Welford's algorithm)
        running_mean = torch.zeros(vae_model.latent_dim, device=device)
        running_M2 = torch.zeros(vae_model.latent_dim, device=device)  # Sum of squared deviations
        total_count = 0

        # Global online statistics (for normalized values - estimated from per-dim stats)
        # We'll compute global_scale from the per-dim variance distribution
        all_batch_global_vars = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Fitting scaler"):
                answer_ids = batch["answer_input_ids"].to(device, non_blocking=True)
                answer_mask = batch["answer_attention_mask"].to(device, non_blocking=True)

                # Encode with noise to match training distribution
                z_sampled, mean, logvar, latent_mask = vae_model.encode(answer_ids, answer_mask)

                if noise_scale > 0:
                    std = torch.exp(0.5 * logvar)
                    noise = torch.randn_like(mean)
                    z = mean + noise_scale * std * noise
                else:
                    z = mean

                # Mask out padding
                mask = latent_mask.bool()
                z_masked = z[mask]  # [valid_latents, latent_dim]

                if z_masked.shape[0] == 0:
                    continue

                batch_count = z_masked.shape[0]

                # Welford's online algorithm for mean and variance
                for i in range(batch_count):
                    total_count += 1
                    delta = z_masked[i] - running_mean
                    running_mean = running_mean + delta / total_count
                    delta2 = z_masked[i] - running_mean
                    running_M2 = running_M2 + delta * delta2

                # Track global variance estimate for this batch
                # (After per-dim normalization, what's the overall std?)
                batch_mean = z_masked.mean(dim=0)
                batch_var = z_masked.var(dim=0)
                # Estimate: after normalizing by running stats, what's the global var?
                if total_count > 100:  # Only after we have stable estimates
                    current_std = torch.sqrt(running_M2 / total_count).clamp(min=1e-3)
                    z_norm_est = (z_masked - running_mean) / current_std
                    all_batch_global_vars.append(z_norm_est.var().item())

        # Finalize per-dimension statistics
        self.mean = running_mean
        variance = running_M2 / max(total_count, 1)
        self.std = torch.sqrt(variance).clamp(min=1e-3)

        # Compute global scale from collected estimates
        if all_batch_global_vars:
            # Use median for robustness to outliers
            sorted_vars = sorted(all_batch_global_vars)
            median_var = sorted_vars[len(sorted_vars) // 2]
            self.global_scale = torch.tensor(median_var ** 0.5, device=device)
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
