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
        """
        print("Fitting LatentScaler...")
        vae_model.eval()
        self.device = device
        
        all_latents = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Computing latent stats"):
                answer_ids = batch["answer_input_ids"].to(device)
                answer_mask = batch["answer_attention_mask"].to(device)
                
                # Get mean latent (deterministic)
                z = vae_model.get_latent(answer_ids, answer_mask, use_mean=True)
                
                # Mask out padding
                mask = answer_mask.unsqueeze(-1).bool()
                z_masked = z[mask].view(-1, z.shape[-1])
                
                all_latents.append(z_masked.cpu())
        
        # Concatenate all latents
        all_latents = torch.cat(all_latents, dim=0)
        
        # Compute stats
        self.mean = all_latents.mean(dim=0).to(device)
        self.std = all_latents.std(dim=0).to(device)
        
        # Avoid division by zero
        self.std = torch.clamp(self.std, min=1e-5)
        
        print(f"LatentScaler fitted: mean_norm={self.mean.norm().item():.4f}, std_norm={self.std.norm().item():.4f}")

    def transform(self, z, mask: Optional[torch.Tensor] = None):
        """Normalize z: (z - mu) / sigma, with optional zero-remasking."""
        if self.mean is None:
            return z
        
        z_norm = (z - self.mean) / self.std
        
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
