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
    """

    def __init__(self):
        self.mean = None
        self.std = None
        self.device = None

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
                
                # Get mean latent (deterministic)
                z = vae_model.get_latent(answer_ids, answer_mask, use_mean=True)
                
                # Mask out padding and compute statistics online
                mask = answer_mask.bool()
                z_masked = z[mask]  # [valid_tokens, latent_dim]
                
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
        
        print(f"LatentScaler fitted: mean_norm={self.mean.norm().item():.4f}, std_norm={self.std.norm().item():.4f}")

    def transform(self, z, mask: Optional[torch.Tensor] = None):
        """Normalize z: (z - mu) / sigma, with optional zero-remasking and clipping."""
        if self.mean is None:
            return z
        
        z_norm = (z - self.mean) / self.std
        
        # Clipping to prevent extreme outliers from destabilizing diffusion
        z_norm = torch.clamp(z_norm, min=-5.0, max=5.0)
        
        if mask is not None:
            # Ensure mask is [batch, seq_len, 1] for broadcasting
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)
            z_norm = z_norm * mask.float()
            
        return z_norm

    def inverse_transform(self, z_norm, mask: Optional[torch.Tensor] = None):
        """Denormalize z: z_norm * sigma + mu, with optional zero-remasking."""
        if self.mean is None:
            return z_norm
            
        z = z_norm * self.std + self.mean
        
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)
            z = z * mask.float()
            
        return z

    def to(self, device):
        """Move stats to device."""
        self.device = device
        if self.mean is not None:
            self.mean = self.mean.to(device)
        if self.std is not None:
            self.std = self.std.to(device)
        return self
