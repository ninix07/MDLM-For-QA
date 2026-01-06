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

    def __init__(self, scheduler: NoiseScheduler, prediction_type: str = "epsilon"):
        self.scheduler = scheduler
        self.num_timesteps = scheduler.num_timesteps
        self.prediction_type = prediction_type

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
        Optimized to reduce tensor allocations.

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

        # Pre-allocate t_batch tensor to avoid repeated allocations
        t_batch = torch.empty((batch_size,), device=device, dtype=torch.long)

        for t in timesteps:
            t_batch.fill_(t)

            model_output = model(z_t, t_batch, **condition_kwargs)

            alpha_bar = self.scheduler.alphas_cumprod[t]

            # Derive pred_z0 and pred_epsilon
            if self.prediction_type == "epsilon":
                pred_epsilon = model_output
                pred_z0 = (z_t - torch.sqrt(1 - alpha_bar) * pred_epsilon) / torch.sqrt(alpha_bar)
            elif self.prediction_type == "v":
                pred_z0 = torch.sqrt(alpha_bar) * z_t - torch.sqrt(1 - alpha_bar) * model_output
                pred_epsilon = (
                    torch.sqrt(alpha_bar) * model_output + torch.sqrt(1 - alpha_bar) * z_t
                )
            else:
                raise ValueError(f"Unknown prediction type: {self.prediction_type}")

            # Stability clamping - match scaler range [-5,5]
            pred_z0 = torch.clamp(pred_z0, -5.0, 5.0)

            alpha = self.scheduler.alphas[t]
            alpha_bar_prev = self.scheduler.alphas_cumprod_prev[t]

            coef1 = torch.sqrt(alpha_bar_prev) * self.scheduler.betas[t] / (1 - alpha_bar)
            coef2 = torch.sqrt(alpha) * (1 - alpha_bar_prev) / (1 - alpha_bar)
            mean = coef1 * pred_z0 + coef2 * z_t

            if t > 0:
                # Reuse noise tensor shape to avoid allocation
                noise = torch.randn_like(z_t)
                var = self.scheduler.posterior_variance[t]
                z_t = mean + torch.sqrt(var) * noise
            else:
                z_t = mean

        return z_t


class DDIMSampler:
    """DDIM sampling for faster inference."""

    @property
    def num_inference_steps(self) -> int:
        return self._num_inference_steps

    @num_inference_steps.setter
    def num_inference_steps(self, value: int):
        self._num_inference_steps = value
        step_ratio = self.scheduler.num_timesteps // value
        self.timesteps = list(range(0, self.scheduler.num_timesteps, step_ratio))[::-1]

    def __init__(
        self,
        scheduler: NoiseScheduler,
        num_inference_steps: int = 50,
        eta: float = 0.0,
        prediction_type: str = "epsilon",
    ):
        self.scheduler = scheduler
        self._num_inference_steps = num_inference_steps
        self.eta = eta
        self.prediction_type = prediction_type

        # Initialize timesteps via setter
        self.num_inference_steps = num_inference_steps

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
        Optimized to reduce tensor allocations.

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

        # Pre-allocate t_batch tensor to avoid repeated allocations
        t_batch = torch.empty((batch_size,), device=device, dtype=torch.long)

        for i, t in enumerate(timesteps):
            t_batch.fill_(t)

            model_output = model(z_t, t_batch, **condition_kwargs)

            alpha_bar = self.scheduler.alphas_cumprod[t]

            if i + 1 < len(self.timesteps):
                t_prev = self.timesteps[i + 1]
                alpha_bar_prev = self.scheduler.alphas_cumprod[t_prev]
            else:
                alpha_bar_prev = torch.tensor(1.0, device=device)

            # Derive pred_z0 and pred_epsilon
            if self.prediction_type == "epsilon":
                pred_epsilon = model_output
                pred_z0 = (z_t - torch.sqrt(1 - alpha_bar) * pred_epsilon) / torch.sqrt(alpha_bar)
            elif self.prediction_type == "v":
                pred_z0 = torch.sqrt(alpha_bar) * z_t - torch.sqrt(1 - alpha_bar) * model_output
                pred_epsilon = (
                    torch.sqrt(alpha_bar) * model_output + torch.sqrt(1 - alpha_bar) * z_t
                )
            else:
                raise ValueError(f"Unknown prediction type: {self.prediction_type}")

            if clip_sample:
                pred_z0 = torch.clamp(pred_z0, -5.0, 5.0)  # Match the scaler!

            sigma = self.eta * torch.sqrt(
                (1 - alpha_bar_prev) / (1 - alpha_bar) * (1 - alpha_bar / alpha_bar_prev)
            )

            dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma**2) * pred_epsilon

            if self.eta > 0 and i + 1 < len(self.timesteps):
                # Reuse noise tensor shape to avoid allocation
                noise = torch.randn_like(z_t)
                z_t = torch.sqrt(alpha_bar_prev) * pred_z0 + dir_xt + sigma * noise
            else:
                z_t = torch.sqrt(alpha_bar_prev) * pred_z0 + dir_xt

        return z_t


class CachedDDIMSampler:
    """DDIM sampler with cached condition encoding for efficiency."""

    def __init__(
        self,
        scheduler: NoiseScheduler,
        num_inference_steps: int = 50,
        eta: float = 0.0,
        prediction_type: str = "epsilon",
    ):
        self.scheduler = scheduler
        self.num_inference_steps = num_inference_steps
        self.eta = eta
        self.prediction_type = prediction_type

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
        uncond_context_ids: Optional[torch.Tensor] = None,
        uncond_context_mask: Optional[torch.Tensor] = None,
        uncond_question_ids: Optional[torch.Tensor] = None,
        uncond_question_mask: Optional[torch.Tensor] = None,
        guidance_scale: float = 5.0,
    ) -> torch.Tensor:
        """Sample with cached condition for faster inference."""
        batch_size = shape[0]
        z_t = torch.randn(shape, device=device)

        # Encode condition
        condition, condition_mask = model.encode_condition(
            context_ids, context_mask, question_ids, question_mask
        )

        # CFG Prep: Encode unconditional if needed
        # BUG #24 FIX: Always recompute unconditional to avoid stale cache issues
        is_cfg = guidance_scale != 1.0 and uncond_context_ids is not None
        if is_cfg:
            uncond_condition, uncond_condition_mask = model.encode_condition(
                uncond_context_ids, uncond_context_mask, uncond_question_ids, uncond_question_mask
            )

            # Concatenate for batch processing [2*B, seq, dim]
            condition = torch.cat([condition, uncond_condition])
            condition_mask = torch.cat([condition_mask, uncond_condition_mask])

        timesteps = self.timesteps
        if show_progress:
            timesteps = tqdm(timesteps, desc="DDIM Sampling")

        # BUG #22 FIX: Create z_mask for inference (all positions valid, no padding)
        # This ensures consistent behavior between training (with mask) and inference
        z_mask = torch.ones(batch_size, shape[1], device=device)

        for i, t in enumerate(timesteps):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

            if is_cfg:
                # Expand z and t for dual-pass [2*B, ...]
                z_in = torch.cat([z_t, z_t])
                t_in = torch.cat([t_batch, t_batch])
                # BUG #22 FIX: Expand z_mask for CFG
                z_mask_in = torch.cat([z_mask, z_mask])

                model_output_all = model.forward_with_cached_condition(
                    z_in, t_in, condition, condition_mask, z_mask=z_mask_in
                )

                # Split and apply CFG formula to raw model output
                model_output_cond, model_output_uncond = model_output_all.chunk(2)
                model_output = model_output_uncond + guidance_scale * (
                    model_output_cond - model_output_uncond
                )
            else:
                model_output = model.forward_with_cached_condition(
                    z_t, t_batch, condition, condition_mask, z_mask=z_mask
                )

            alpha_bar = self.scheduler.alphas_cumprod[t]

            if i + 1 < len(self.timesteps):
                t_prev = self.timesteps[i + 1]
                alpha_bar_prev = self.scheduler.alphas_cumprod[t_prev]
            else:
                # FIX Bug 6: Use 1.0 directly for final step (consistent with DDIMSampler)
                alpha_bar_prev = torch.tensor(1.0, device=device)

            # Derive pred_z0 and pred_epsilon
            if self.prediction_type == "epsilon":
                pred_epsilon = model_output
                pred_z0 = (z_t - torch.sqrt(1 - alpha_bar) * pred_epsilon) / torch.sqrt(
                    alpha_bar.clamp(min=1e-8)
                )
            elif self.prediction_type == "v":
                pred_z0 = torch.sqrt(alpha_bar) * z_t - torch.sqrt(1 - alpha_bar) * model_output
                pred_epsilon = (
                    torch.sqrt(alpha_bar) * model_output + torch.sqrt(1 - alpha_bar) * z_t
                )
            else:
                raise ValueError(f"Unknown prediction type: {self.prediction_type}")

            if clip_sample:
                pred_z0 = torch.clamp(pred_z0, -5.0, 5.0)  # Match the scaler!

            sigma = self.eta * torch.sqrt(
                (1 - alpha_bar_prev) / (1 - alpha_bar) * (1 - alpha_bar / alpha_bar_prev)
            )

            dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma**2) * pred_epsilon

            if self.eta > 0 and i + 1 < len(self.timesteps):
                noise = torch.randn_like(z_t)
                z_t = torch.sqrt(alpha_bar_prev) * pred_z0 + dir_xt + sigma * noise
            else:
                z_t = torch.sqrt(alpha_bar_prev) * pred_z0 + dir_xt

        return z_t
