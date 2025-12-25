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
    denoiser_layers: int = 4  # Reduced from 6 for memory
    denoiser_heads: int = 8
    denoiser_dim: int = 512  # Reduced from 768 for memory
    denoiser_ff_dim: int = 1024  # Reduced from 2048 for memory
    dropout: float = 0.1

    # VAE config (if using Approach B)
    use_vae: bool = True
    vae_latent_dim: int = 256
    vae_hidden_dim: int = 512

    # Max sequence lengths
    max_context_length: int = 512  # Increased from 384
    max_question_length: int = 100  # Increased from 64
    max_answer_length: int = 100  # Increased from 64

    # Special tokens
    null_ans_token: str = "<NULL_ANS>"
    pad_token: str = "<pad>"


@dataclass
class DiffusionConfig:
    """Diffusion process configuration."""

    # Number of diffusion steps
    num_train_timesteps: int = 1000
    num_inference_timesteps: int = 50  # DDIM sampling

    # Noise schedule type: 'linear', 'cosine', 'sqrt'
    schedule_type: str = "cosine"

    # Beta schedule bounds (for linear schedule)
    beta_start: float = 0.0001
    beta_end: float = 0.02

    # Cosine schedule parameters
    cosine_s: float = 0.008  # Small offset to prevent singularity

    # Prediction type: 'epsilon' (noise) or 'v' (velocity)
    prediction_type: str = "epsilon"

    # Clipping
    clip_sample: bool = True
    clip_sample_range: float = 1.0


@dataclass
class TrainingConfig:
    """Training configuration."""

    # Batch size (reduced for memory - use gradient accumulation for effective larger batch)
    batch_size: int = 8
    gradient_accumulation_steps: int = 4  # Effective batch size = 32

    # Learning rate
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0

    # Training duration
    num_epochs: int = 10
    max_steps: Optional[int] = None

    # Logging and saving
    log_every: int = 100
    save_every: int = 1000
    eval_every: int = 500

    # Mixed precision
    use_amp: bool = True

    # Balanced batching for SQuAD 2.0
    answerable_ratio: float = 0.5  # 50% answerable, 50% unanswerable

    # Auxiliary loss to keep latents close to valid embeddings
    use_embedding_loss: bool = True
    embedding_loss_weight: float = 0.1

    # False negative penalty (penalize predicting no answer when answerable)
    false_negative_penalty_weight: float = 1.0

    # Checkpointing
    output_dir: str = "./checkpoints"
    resume_from: Optional[str] = None

    # VAE Warmup
    vae_warmup_epochs: int = 20
    vae_patience: int = 5


@dataclass
class InferenceConfig:
    """Inference configuration."""

    # DDIM sampling steps
    num_inference_steps: int = 50

    # Guidance scale for classifier-free guidance (if implemented)
    guidance_scale: float = 1.0

    # Null answer threshold
    null_ans_threshold: float = 0.7  # Cosine similarity threshold

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
