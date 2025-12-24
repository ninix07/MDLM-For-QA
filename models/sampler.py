"""
Samplers for reverse diffusion: DDPM and DDIM.
"""

from typing import Optional, Callable
import torch
import torch.nn as nn
from tqdm import tqdm

from .diffusion import NoiseScheduler


class DDPMSampler:
    """Standard DDPM sampling (slow but high quality)."""

    def __init__(self, scheduler: NoiseScheduler):
        self.scheduler = scheduler
        self.num_timesteps = scheduler.num_timesteps

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: tuple,
        condition_kwargs: dict,
        device: torch.device,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """
        Sample from the model using DDPM.

        Args:
            model: Denoiser model
            shape: Shape of output (batch, seq_len, dim)
            condition_kwargs: Conditioning inputs
            device: Device to use
        """
        batch_size = shape[0]
        z_t = torch.randn(shape, device=device)

        timesteps = list(range(self.num_timesteps))[::-1]
        if show_progress:
            timesteps = tqdm(timesteps, desc="DDPM Sampling")

        for t in timesteps:
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

            noise_pred = model(z_t, t_batch, **condition_kwargs)

            alpha = self.scheduler.alphas[t]
            alpha_bar = self.scheduler.alphas_cumprod[t]
            alpha_bar_prev = self.scheduler.alphas_cumprod_prev[t]

            pred_x0 = (z_t - torch.sqrt(1 - alpha_bar) * noise_pred) / torch.sqrt(
                alpha_bar
            )
            

            coef1 = (
                torch.sqrt(alpha_bar_prev) * self.scheduler.betas[t] / (1 - alpha_bar)
            )
            coef2 = torch.sqrt(alpha) * (1 - alpha_bar_prev) / (1 - alpha_bar)
            mean = coef1 * pred_x0 + coef2 * z_t

            if t > 0:
                noise = torch.randn_like(z_t)
                var = self.scheduler.posterior_variance[t]
                z_t = mean + torch.sqrt(var) * noise
            else:
                z_t = mean

        return z_t


class DDIMSampler:
    """DDIM sampling for faster inference."""

    def __init__(
        self,
        scheduler: NoiseScheduler,
        num_inference_steps: int = 50,
        eta: float = 0.0,
    ):
        self.scheduler = scheduler
        self.num_inference_steps = num_inference_steps
        self.eta = eta

        step_ratio = scheduler.num_timesteps // num_inference_steps
        self.timesteps = list(range(0, scheduler.num_timesteps, step_ratio))[::-1]

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: tuple,
        condition_kwargs: dict,
        device: torch.device,
        show_progress: bool = True,
        clip_sample: bool = True,
    ) -> torch.Tensor:
        """
        Sample using DDIM (faster than DDPM).

        Args:
            model: Denoiser model
            shape: Output shape (batch, seq_len, dim)
            condition_kwargs: Conditioning inputs
            device: Device
            clip_sample: Whether to clip predicted x0
        """
        batch_size = shape[0]
        z_t = torch.randn(shape, device=device)

        timesteps = self.timesteps
        if show_progress:
            timesteps = tqdm(timesteps, desc="DDIM Sampling")

        for i, t in enumerate(timesteps):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

            noise_pred = model(z_t, t_batch, **condition_kwargs)

            alpha_bar = self.scheduler.alphas_cumprod[t]

            if i + 1 < len(self.timesteps):
                t_prev = self.timesteps[i + 1]
                alpha_bar_prev = self.scheduler.alphas_cumprod[t_prev]
            else:
                alpha_bar_prev = torch.tensor(1.0, device=device)

            pred_x0 = (z_t - torch.sqrt(1 - alpha_bar) * noise_pred) / torch.sqrt(
                alpha_bar
            )
            if clip_sample:
                pred_x0 = torch.clamp(pred_x0, -1, 1)

            sigma = self.eta * torch.sqrt(
                (1 - alpha_bar_prev)
                / (1 - alpha_bar)
                * (1 - alpha_bar / alpha_bar_prev)
            )

            dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma**2) * noise_pred

            if self.eta > 0 and i + 1 < len(self.timesteps):
                noise = torch.randn_like(z_t)
                z_t = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt + sigma * noise
            else:
                z_t = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt

        return z_t


class CachedDDIMSampler:
    """DDIM sampler with cached condition encoding for efficiency."""

    def __init__(
        self,
        scheduler: NoiseScheduler,
        num_inference_steps: int = 50,
        eta: float = 0.0,
    ):
        self.scheduler = scheduler
        self.num_inference_steps = num_inference_steps
        self.eta = eta

        step_ratio = scheduler.num_timesteps // num_inference_steps
        self.timesteps = list(range(0, scheduler.num_timesteps, step_ratio))[::-1]

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: tuple,
        context_ids: torch.Tensor,
        context_mask: torch.Tensor,
        question_ids: torch.Tensor,
        question_mask: torch.Tensor,
        device: torch.device,
        show_progress: bool = True,
        clip_sample: bool = True,
    ) -> torch.Tensor:
        """Sample with cached condition for faster inference."""
        batch_size = shape[0]
        z_t = torch.randn(shape, device=device)

        condition, condition_mask = model.encode_condition(
            context_ids, context_mask, question_ids, question_mask
        )

        timesteps = self.timesteps
        if show_progress:
            timesteps = tqdm(timesteps, desc="DDIM Sampling")

        for i, t in enumerate(timesteps):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

            noise_pred = model.forward_with_cached_condition(
                z_t, t_batch, condition, condition_mask
            )

            alpha_bar = self.scheduler.alphas_cumprod[t]

            if i + 1 < len(self.timesteps):
                t_prev = self.timesteps[i + 1]
                alpha_bar_prev = self.scheduler.alphas_cumprod[t_prev]
            else:
                # Use the first value of alphas_cumprod (clean state)
                alpha_bar_prev = self.scheduler.alphas_cumprod_prev[0]

            pred_x0 = (z_t - torch.sqrt(1 - alpha_bar) * noise_pred) / torch.sqrt(
                alpha_bar
            )
            if clip_sample:
                pred_x0 = torch.clamp(pred_x0, -1, 1)

            sigma = self.eta * torch.sqrt(
                (1 - alpha_bar_prev)
                / (1 - alpha_bar)
                * (1 - alpha_bar / alpha_bar_prev)
            )

            dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma**2) * noise_pred

            if self.eta > 0 and i + 1 < len(self.timesteps):
                noise = torch.randn_like(z_t)
                z_t = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt + sigma * noise
            else:
                z_t = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt

        return z_t
