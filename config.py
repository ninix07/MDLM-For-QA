"""
Configuration for Multilingual Latent Diffusion Model for SQuAD 2.0
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    # Base encoder for multilingual support
    base_encoder: str = "bert-base-uncased"

    # Latent dimensions (VAE compresses to this)
    latent_dim: int = 256  # VAE latent dimension (smaller than embedding_dim)

    # Denoiser Transformer config
    denoiser_layers: int = 12  # Increased to 12 for deep cross-attention on 512-token contexts
    denoiser_heads: int = 12  # At least 12 heads for complex inference over 512 tokens
    denoiser_dim: int = 768  # Increased to 768 to eliminate bottleneck
    denoiser_ff_dim: int = 1024  # Reduced from 2048 for memory
    dropout: float = 0.1

    # VAE config (if using Approach B)
    use_vae: bool = True
    vae_latent_dim: int = 256

    # BUG #9 FIX: Sequence compression ratio - increased from 8 to 32 for better reconstruction
    # 64/2 = 32 latent tokens (2x compression)
    latent_seq_len: int = 32  # Increased to 32 for higher resolution reconstruction

    # Max sequence lengths
    max_context_length: int = 512  # Reverted to 512 (BERT limit)
    max_question_length: int = 64  # Increased from 64 64 is good
    max_answer_length: int = 64  # Increased from 64

    # Special tokens
    null_ans_token: str = "<NULL_ANS>"


@dataclass
class DiffusionConfig:
    """Diffusion process configuration."""

    # Number of diffusion steps
    num_train_timesteps: int = 2000
    num_inference_timesteps: int = 50  # DDIM sampling

    # Noise schedule type: 'linear', 'cosine', 'sqrt'
    schedule_type: str = "cosine"

    # Beta schedule bounds (for linear schedule)
    beta_start: float = 0.0001
    beta_end: float = 0.02

    # Cosine schedule parameters
    cosine_s: float = 0.008  # Small offset to prevent singularity

    # Prediction type: 'epsilon' (noise) or 'v' (velocity)
    prediction_type: str = "v"


@dataclass
class TrainingConfig:
    """Training configuration."""

    # Batch size (increased for stability)
    batch_size: int = 16
    gradient_accumulation_steps: int = 4  # Effective batch size = 16 * 4 = 64

    # Learning rate
    learning_rate: float = 1e-5  # Smaller LR for large model convergence
    weight_decay: float = 0.1
    warmup_steps: int = 2000
    max_grad_norm: float = 0.5  # Reduced from 1.0 to clip spikes aggressively

    # Training duration
    num_epochs: int = 10
    max_steps: Optional[int] = None

    # Logging and saving
    log_every: int = 100
    save_every: int = 1000
    eval_every: int = 500

    # Mixed precision
    use_amp: bool = True


    # False negative penalty (penalize predicting no answer when answerable)
    # False negative penalty (penalize predicting no answer when answerable)
    # BUG #34 FIX: Increased to 1.0 to combat modal collapse (predicting NULL for everything)
    false_negative_penalty_weight: float = 1.0
    false_negative_penalty_margin: float = 0.4  # Increased margin to push away from null

    # Auxiliary token loss (Option A fix for VAE-Diffusion alignment)
    # This provides direct token-level signal to the diffusion model
    aux_token_loss_weight: float = 1.0  # Increased to 1.0 to force learning
    aux_token_loss_low_t_threshold: int = 1000  # Apply to first 50% of steps

    # Checkpointing
    output_dir: str = "./checkpoints"
    resume_from: Optional[str] = None

    # VAE Warmup
    vae_warmup_epochs: int = 40
    vae_patience: int = 5
    # BUG #36 FIX: Increased target KL from 0.01 to 0.1 to force tighter latent distribution
    # High KL (~15) means posterior is far from N(0,1), making diffusion hard
    target_kl: float = 0.1


@dataclass
class InferenceConfig:
    """Inference configuration."""

    # DDIM sampling steps
    num_inference_steps: int = 50

    # Guidance scale for classifier-free guidance (if implemented)
    # BUG #36 FIX: Reduced from 3.0 to 1.5 - high guidance pushes latents out of VAE distribution
    guidance_scale: float = 1.5

    # Null answer threshold
    null_ans_threshold: float = 0.5  # Require higher certainty before abstaining

    # Temperature for sampling
    temperature: float = 1.0

    # Beam search (for token decoding)
    use_beam_search: bool = False
    beam_width: int = 5


@dataclass
class WandbConfig:
    """Weights & Biases configuration."""

    project: str = "mdlm-squad"
    entity: Optional[str] = None
    name: Optional[str] = None
    mode: str = "online"  # "online", "offline", "disabled"


@dataclass
class Config:
    """Main configuration combining all sub-configs."""

    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    # Device
    device: str = "cuda"
    seed: int = 42

    # Data paths
    train_file: str = "data/train-v2.0.json"
    dev_file: str = "data/dev-v2.0.json"


def get_config() -> Config:
    """Get default configuration."""
    return Config()
