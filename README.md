# Multilingual Latent Diffusion Model for SQuAD 2.0

A Latent Diffusion Model (LDM) for Question Answering on SQuAD 2.0, with support for unanswerable questions.

## Overview

This implementation follows the outlined approach for text diffusion:

1. **Latent Space**: Operates in continuous latent space using a VAE or embedding bridge
2. **Conditional Denoising**: Transformer-based denoiser conditioned on question + context
3. **Null Answer Detection**: Threshold-based detection for unanswerable questions

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Latent Diffusion QA                       │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────┐  │
│  │   VAE /     │    │  Conditional │    │    DDIM       │  │
│  │  Embedding  │───▶│   Denoiser   │───▶│   Sampler     │  │
│  │   Bridge    │    │ (Transformer)│    │               │  │
│  └─────────────┘    └──────────────┘    └───────────────┘  │
│         ▲                  ▲                               │
│         │                  │                               │
│    Answer Text      Question + Context                     │
│                    (Frozen XLM-R)                          │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
MDLM for Squad/
├── config.py              # Configuration dataclasses
├── train.py               # Training script
├── evaluate.py            # Evaluation with SQuAD metrics
├── inference.py           # Interactive inference
├── requirements.txt       # Dependencies
├── data/
│   ├── __init__.py
│   └── dataset.py         # SQuAD 2.0 dataset with balanced sampling
└── models/
    ├── __init__.py
    ├── embeddings.py      # Timestep and positional embeddings
    ├── transformer_blocks.py  # Conditional transformer blocks
    ├── vae.py             # VAE and embedding bridge
    ├── denoiser.py        # Conditional denoiser network
    ├── diffusion.py       # Noise scheduler and diffusion process
    ├── sampler.py         # DDPM and DDIM samplers
    └── latent_diffusion.py    # Main model combining all components
```

## Installation

Using [uv](https://github.com/astral-sh/uv) (recommended):

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .

# Or install with dev dependencies
uv pip install -e ".[dev]"
```

Using pip:

```bash
pip install -r requirements.txt
```

## Data Preparation

Download SQuAD 2.0:

```bash
mkdir -p data
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json -O data/train-v2.0.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json -O data/dev-v2.0.json
```

## Training

```bash
python train.py \
    --train_file data/train-v2.0.json \
    --dev_file data/dev-v2.0.json \
    --output_dir ./checkpoints \
    --batch_size 32 \
    --epochs 10 \
    --lr 1e-4 \
    --use_vae
```

### Key Training Features

- **Balanced Batching**: 50% answerable, 50% unanswerable questions per batch
- **Cosine Noise Schedule**: Better for text than linear schedule
- **Mixed Precision**: Automatic mixed precision for faster training
- **VAE Regularization**: Optional VAE for smoother latent space

## Evaluation

```bash
python evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --data_file data/dev-v2.0.json \
    --null_threshold 0.3
```

Outputs:

- Exact Match (EM)
- F1 Score
- HasAnswer EM/F1
- NoAnswer Accuracy

## Inference

### Single Question

```bash
python inference.py \
    --checkpoint checkpoints/best_model.pt \
    --context "The Eiffel Tower is located in Paris, France." \
    --question "Where is the Eiffel Tower?"
```

### Interactive Mode

```bash
python inference.py --checkpoint checkpoints/best_model.pt --interactive
```

### Python API

```python
from inference import QAInference

qa = QAInference("checkpoints/best_model.pt")
result = qa.answer(
    context="The Eiffel Tower is located in Paris, France.",
    question="Where is the Eiffel Tower?"
)
print(result['answer'])  # "Paris, France"
print(result['is_unanswerable'])  # False
```

## Key Components

### 1. VAE / Embedding Bridge (`models/vae.py`)

- **EmbeddingBridge**: Simple approach using frozen XLM-R embeddings
- **SequenceVAE**: Full VAE preserving sequence structure for smoother latent space

### 2. Conditional Denoiser (`models/denoiser.py`)

- Transformer with cross-attention to question + context
- AdaLN (Adaptive Layer Norm) for timestep conditioning
- Frozen XLM-R encoder for multilingual support

### 3. Diffusion Process (`models/diffusion.py`)

- Cosine noise schedule (recommended for text)
- Forward process: q(z_t | z_0)
- Training loss: MSE on predicted noise

### 4. Sampling (`models/sampler.py`)

- **DDPM**: Standard sampling (1000 steps)
- **DDIM**: Fast sampling (50 steps)
- Cached condition encoding for efficiency

### 5. Null Answer Detection

- Compares generated latent to `<NULL_ANS>` embedding
- Cosine similarity threshold (default: 0.7)
- Returns empty string if above threshold

## Hyperparameters

| Component              | Default Value    |
| ---------------------- | ---------------- |
| Base Encoder           | xlm-roberta-base |
| Latent Dim (VAE)       | 256              |
| Denoiser Layers        | 6                |
| Denoiser Heads         | 8                |
| Training Steps         | 1000             |
| Inference Steps (DDIM) | 50               |
| Noise Schedule         | Cosine           |
| Batch Size             | 32               |
| Learning Rate          | 1e-4             |
| Null Threshold         | 0.7              |

## Troubleshooting

### "Word Salad" Outputs

- Increase VAE KL weight to regularize latent space
- Add auxiliary embedding loss during training
- Check that latent space is well-clustered

### Model Ignores Context

- Verify cross-attention is working (check attention weights)
- Ensure context is not being truncated
- Try unfreezing some XLM-R layers

### Slow Inference

- Use DDIM sampling (50 steps vs 1000)
- Enable cached condition encoding
- Reduce max answer length

### Poor Null Answer Detection

- Tune `null_threshold` on validation set
- Ensure balanced training (50/50 split)
- Check `<NULL_ANS>` embedding is learned correctly

## Citation

If you use this code, please cite:

```bibtex
@misc{mdlm-squad,
  title={Multilingual Latent Diffusion for SQuAD 2.0},
  year={2024},
}
```

## License

MIT License
