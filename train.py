"""
Training script for Multilingual Latent Diffusion Model on SQuAD 2.0.
"""

import os
import json
import random
import argparse
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast
from transformers import XLMRobertaTokenizer
from tqdm import tqdm

from config import Config, get_config
from data import create_dataloader
from models import LatentDiffusionQA


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_step(
    model, batch, optimizer, scaler, device, use_amp, accumulation_steps=1, step_idx=0
):
    """Single training step with gradient accumulation support."""
    context_ids = batch["context_input_ids"].to(device)
    context_mask = batch["context_attention_mask"].to(device)
    question_ids = batch["question_input_ids"].to(device)
    question_mask = batch["question_attention_mask"].to(device)
    answer_ids = batch["answer_input_ids"].to(device)
    answer_mask = batch["answer_attention_mask"].to(device)

    # Only zero gradients at the start of accumulation
    if step_idx % accumulation_steps == 0:
        optimizer.zero_grad()

    if use_amp and scaler is not None:
        with autocast(device_type=device.type):
            outputs = model(
                context_ids,
                context_mask,
                question_ids,
                question_mask,
                answer_ids,
                answer_mask,
            )
            loss = outputs["loss"] / accumulation_steps
        scaler.scale(loss).backward()

        # Only step optimizer after accumulation
        if (step_idx + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
    else:
        outputs = model(
            context_ids,
            context_mask,
            question_ids,
            question_mask,
            answer_ids,
            answer_mask,
        )
        loss = outputs["loss"] / accumulation_steps
        loss.backward()

        # Only step optimizer after accumulation
        if (step_idx + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    return {
        "loss": loss.item() * accumulation_steps,  # Return unscaled loss for logging
        "diff_loss": outputs["diffusion_loss"].item(),
        "vae_loss": outputs["vae_loss"].item(),
    }


@torch.no_grad()
def validate(model, val_loader, device):
    """Validation loop."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(val_loader, desc="Validating"):
        context_ids = batch["context_input_ids"].to(device)
        context_mask = batch["context_attention_mask"].to(device)
        question_ids = batch["question_input_ids"].to(device)
        question_mask = batch["question_attention_mask"].to(device)
        answer_ids = batch["answer_input_ids"].to(device)
        answer_mask = batch["answer_attention_mask"].to(device)

        outputs = model(
            context_ids,
            context_mask,
            question_ids,
            question_mask,
            answer_ids,
            answer_mask,
        )
        total_loss += outputs["loss"].item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def save_checkpoint(model, optimizer, scheduler, epoch, step, loss, path):
    """Save model checkpoint."""
    torch.save(
        {
            "epoch": epoch,
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "loss": loss,
        },
        path,
    )


def load_checkpoint(model, optimizer, scheduler, path, device):
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and checkpoint["scheduler_state_dict"]:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint["epoch"], checkpoint["step"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default="data/train-v2.0.json")
    parser.add_argument("--dev_file", type=str, default="data/dev-v2.0.json")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_vae", action="store_true", default=True)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    config = get_config()
    config.training.output_dir = args.output_dir
    config.training.batch_size = args.batch_size
    config.training.num_epochs = args.epochs
    config.training.learning_rate = args.lr

    # Device selection: CUDA > MPS > CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        amp_device = "cuda"
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        amp_device = "cpu"  # AMP not fully supported on MPS
    else:
        device = torch.device("cpu")
        amp_device = "cpu"
    print(f"Using device: {device}")

    # Tokenizer
    tokenizer = XLMRobertaTokenizer.from_pretrained(config.model.base_encoder)

    # Data loaders
    print("Loading training data...")
    train_loader, train_dataset = create_dataloader(
        args.train_file,
        tokenizer,
        args.batch_size,
        max_context_length=config.model.max_context_length,
        max_question_length=config.model.max_question_length,
        max_answer_length=config.model.max_answer_length,
        answerable_ratio=config.training.answerable_ratio,
    )

    print("Loading validation data...")
    val_loader, val_dataset = create_dataloader(
        args.dev_file,
        tokenizer,
        args.batch_size,
        max_context_length=config.model.max_context_length,
        max_question_length=config.model.max_question_length,
        max_answer_length=config.model.max_answer_length,
        use_balanced_sampler=False,
        shuffle=False,
    )

    # Model
    print("Initializing model...")
    model = LatentDiffusionQA(
        tokenizer=tokenizer,
        latent_dim=config.model.vae_latent_dim,
        d_model=config.model.denoiser_dim,
        num_layers=config.model.denoiser_layers,
        num_heads=config.model.denoiser_heads,
        ff_dim=config.model.denoiser_ff_dim,
        dropout=config.model.dropout,
        max_answer_len=config.model.max_answer_length,
        num_train_timesteps=config.diffusion.num_train_timesteps,
        num_inference_timesteps=config.diffusion.num_inference_timesteps,
        schedule_type=config.diffusion.schedule_type,
        use_vae=args.use_vae,
        base_encoder=config.model.base_encoder,
    )
    model = model.to(device)
    model.scheduler.to(device)

    print(f"Trainable parameters: {count_parameters(model):,}")

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader) * args.epochs)
    # Only use AMP on CUDA
    use_amp = config.training.use_amp and device.type == "cuda"
    scaler = GradScaler(amp_device) if use_amp else None

    # Resume from checkpoint
    start_epoch = 0
    global_step = 0
    if args.resume:
        start_epoch, global_step = load_checkpoint(
            model, optimizer, scheduler, args.resume, device
        )
        print(f"Resumed from epoch {start_epoch}, step {global_step}")

    os.makedirs(args.output_dir, exist_ok=True)
    best_val_loss = float("inf")
    accumulation_steps = config.training.gradient_accumulation_steps

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_idx, batch in enumerate(pbar):
            metrics = train_step(
                model,
                batch,
                optimizer,
                scaler,
                device,
                use_amp,
                accumulation_steps=accumulation_steps,
                step_idx=batch_idx,
            )
            # Only step scheduler after full accumulation
            if (batch_idx + 1) % accumulation_steps == 0:
                scheduler.step()

            epoch_loss += metrics["loss"]
            global_step += 1

            pbar.set_postfix(
                {
                    "loss": f"{metrics['loss']:.4f}",
                    "diff": f"{metrics['diff_loss']:.4f}",
                    "vae": f"{metrics['vae_loss']:.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                }
            )

            if global_step % config.training.save_every == 0:
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    global_step,
                    metrics["loss"],
                    os.path.join(args.output_dir, f"checkpoint-{global_step}.pt"),
                )

        avg_train_loss = epoch_loss / len(train_loader)
        val_loss = validate(model, val_loader, device)

        print(
            f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                global_step,
                val_loss,
                os.path.join(args.output_dir, "best_model.pt"),
            )
            print(f"Saved best model with val loss: {val_loss:.4f}")

    print("Training complete!")


if __name__ == "__main__":
    main()
