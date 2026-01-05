"""
Diffusion process: noise schedules, forward process, and sampling.
"""

import math
from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class NoiseScheduler:
    """
    Noise scheduler for diffusion process.
    Supports linear, cosine, and sqrt schedules.
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        schedule_type: str = "cosine",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        cosine_s: float = 0.008,
        device: Optional[torch.device] = None,
    ):
        self.num_timesteps = num_timesteps
        self.schedule_type = schedule_type
        self.device = device

        if schedule_type == "linear":
            betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        elif schedule_type == "cosine":
            betas = self._cosine_schedule(num_timesteps, cosine_s, device)
        elif schedule_type == "sqrt":
            betas = (
                torch.linspace(beta_start**0.5, beta_end**0.5, num_timesteps, device=device) ** 2
            )
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-12)
        )
        self.posterior_mean_coef1 = (
            betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )

    def _cosine_schedule(
        self, num_timesteps: int, s: float = 0.008, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Cosine schedule as proposed in Improved DDPM."""
        steps = num_timesteps + 1
        t = torch.linspace(0, num_timesteps, steps, device=device)
        alphas_cumprod = torch.cos(((t / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0.0001, 0.9999)

    def to(self, device: torch.device):
        """Move all tensors to device - optimized batch operation."""
        if self.device == device:
            return self  # Already on correct device

        self.device = device

        # Batch move all tensors at once for efficiency
        tensors = [
            self.betas,
            self.alphas,
            self.alphas_cumprod,
            self.alphas_cumprod_prev,
            self.sqrt_alphas_cumprod,
            self.sqrt_one_minus_alphas_cumprod,
            self.sqrt_recip_alphas_cumprod,
            self.sqrt_recipm1_alphas_cumprod,
            self.posterior_variance,
            self.posterior_log_variance_clipped,
            self.posterior_mean_coef1,
            self.posterior_mean_coef2,
        ]

        # Move all tensors in a single operation
        for tensor in tensors:
            tensor.data = tensor.data.to(device)

        return self

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: Tuple) -> torch.Tensor:
        """Extract values from a at indices t, broadcast to x_shape."""
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


class GaussianDiffusion(nn.Module):
    """
    Gaussian Diffusion process for training and sampling.
    """

    def __init__(
        self,
        scheduler: NoiseScheduler,
        prediction_type: str = "epsilon",
    ):
        super().__init__()
        self.scheduler = scheduler
        self.prediction_type = prediction_type
        self.num_timesteps = scheduler.num_timesteps

    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion: add noise to x_0 at timestep t.
        z_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha = self.scheduler._extract(self.scheduler.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alpha = self.scheduler._extract(
            self.scheduler.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )

        z_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
        return z_t, noise

    def predict_x0(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        model_output: torch.Tensor,
    ) -> torch.Tensor:
        """Predict x_0 from z_t and model output (epsilon or v)."""
        sqrt_alpha = self.scheduler._extract(self.scheduler.sqrt_alphas_cumprod, t, z_t.shape)
        sqrt_one_minus_alpha = self.scheduler._extract(
            self.scheduler.sqrt_one_minus_alphas_cumprod, t, z_t.shape
        )

        if self.prediction_type == "epsilon":
            return (z_t - sqrt_one_minus_alpha * model_output) / sqrt_alpha
        elif self.prediction_type == "v":
            return sqrt_alpha * z_t - sqrt_one_minus_alpha * model_output
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")

    def predict_x0_from_noise(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Predict x_0 from z_t and predicted noise (always assumes epsilon)."""
        sqrt_recip = self.scheduler._extract(self.scheduler.sqrt_recip_alphas_cumprod, t, z_t.shape)
        sqrt_recipm1 = self.scheduler._extract(
            self.scheduler.sqrt_recipm1_alphas_cumprod, t, z_t.shape
        )
        return sqrt_recip * z_t - sqrt_recipm1 * noise

    def q_posterior(
        self,
        x_0: torch.Tensor,
        z_t: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute posterior q(z_{t-1} | z_t, x_0)."""
        coef1 = self.scheduler._extract(self.scheduler.posterior_mean_coef1, t, z_t.shape)
        coef2 = self.scheduler._extract(self.scheduler.posterior_mean_coef2, t, z_t.shape)
        posterior_mean = coef1 * x_0 + coef2 * z_t
        posterior_var = self.scheduler._extract(self.scheduler.posterior_variance, t, z_t.shape)
        return posterior_mean, posterior_var

    def training_loss(
        self,
        model: nn.Module,
        x_0: torch.Tensor,
        condition_kwargs: Dict,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss.

        Args:
            model: Denoiser model
            x_0: Clean latent [batch, seq_len, dim]
            condition_kwargs: Dict with context_ids, context_mask, question_ids, question_mask
            mask: Optional mask for valid latent positions (1=valid, 0=pad)
        """
        batch_size = x_0.shape[0]
        device = x_0.device

        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device)
        z_t, noise = self.q_sample(x_0, t)

        # FIX: Pass z_mask to denoiser so self-attention can ignore padded positions
        model_output = model(z_t, t, z_mask=mask, **condition_kwargs)

        # Determine target based on prediction type
        if self.prediction_type == "epsilon":
            target = noise
        elif self.prediction_type == "v":
            # v = sqrt(alpha_bar) * noise - sqrt(1 - alpha_bar) * x_0
            sqrt_alpha = self.scheduler._extract(self.scheduler.sqrt_alphas_cumprod, t, x_0.shape)
            sqrt_one_minus_alpha = self.scheduler._extract(
                self.scheduler.sqrt_one_minus_alphas_cumprod, t, x_0.shape
            )
            target = sqrt_alpha * noise - sqrt_one_minus_alpha * x_0
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")

        # Requirement 5: Min-SNR Weighted MSE Loss
        # SNR(t) = alpha_bar_t / (1 - alpha_bar_t)
        alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
        alpha_bar_t = alphas_cumprod[t]

        # Min-SNR weighting strategy (Hang et al. 2023)
        gamma = 5.0
        snr = alpha_bar_t / (1 - alpha_bar_t).clamp(min=1e-8)

        if self.prediction_type == "epsilon":
            mse_weight = torch.minimum(torch.ones_like(snr), gamma / snr)
        elif self.prediction_type == "v":
            # For v-prediction, the natural weight is SNR / (SNR + 1)
            # Min-SNR weighting for v-prediction: min(SNR, gamma) / (SNR + 1)
            mse_weight = torch.minimum(snr, torch.full_like(snr, gamma)) / (snr + 1)
        else:
            mse_weight = torch.ones_like(snr)

        # Expand for element-wise multiplication: [batch, 1, 1]
        mse_weight = mse_weight.view(-1, 1, 1)

        if mask is not None:
            mask = mask.unsqueeze(-1).float()
            loss = F.mse_loss(model_output, target, reduction="none")
            # Apply weight and mask
            # mask.sum() counts valid tokens, but loss sum includes latent_dim
            # We must divide by (valid_tokens * latent_dim) to get MSE per element
            num_valid_elements = mask.sum() * model_output.shape[-1]
            loss = (loss * mse_weight * mask).sum() / num_valid_elements.clamp(min=1.0)
        else:
            loss = (F.mse_loss(model_output, target, reduction="none") * mse_weight).mean()

        return {
            "loss": loss,
            "model_output": model_output,
            "noise": noise,
            "t": t,
        }
