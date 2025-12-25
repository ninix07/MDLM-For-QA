"""
Training script for Multilingual Latent Diffusion Model on SQuAD 2.0.
"""

import os
import json
import random
import argparse
from datetime import datetime
import wandb

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast
from transformers import AutoTokenizer
from tqdm import tqdm

from config import Config, get_config
from data import create_dataloader
from models import LatentDiffusionQA
from models.scaler import LatentScaler
from models.scaler import LatentScaler
from metrics import compute_metrics


def get_kl_weight(current_step: int, total_steps: int, target_kl: float = 0.1, cycles: int = 4) -> float:
    """
    Calculate KL weight using cyclic linear annealing.
    """
    if total_steps == 0:
        return target_kl
        
    cycle_len = total_steps // cycles
    current_cycle_step = current_step % cycle_len
    
    # Anneal for 50% of the cycle, then hold constant
    warmup_steps = int(cycle_len * 0.5)
    
    if current_cycle_step >= warmup_steps:
        return target_kl
        
    return target_kl * (current_cycle_step / max(1, warmup_steps))



def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def debug_dimensions(model, batch, device, epoch_num):
    """Debug function to verify VAE 768→256 transformation at epoch start."""
    model.eval()
    with torch.no_grad():
        answer_ids = batch["answer_input_ids"][:1].to(device)
        answer_mask = batch["answer_attention_mask"][:1].to(device)
        
        # Decode actual answer text for transparency
        answer_text = model.tokenizer.decode(answer_ids[0], skip_special_tokens=True)
        
        # Get BERT embeddings
        embeddings = model.vae.embeddings(answer_ids)
        
        # Get VAE latent
        z, mean, logvar = model.vae.encode(answer_ids, answer_mask)
        
        print(f"\n[DEBUG Epoch {epoch_num}] Dimension Verification:")
        print(f"    Sample Answer: '{answer_text[:50]}...' " if len(answer_text) > 50 else f"    Sample Answer: '{answer_text}'")
        print(f"    BERT Embeddings: {embeddings.shape} (expected: [1, seq, 768])")
        print(f"    VAE Latent (z):  {z.shape} (expected: [1, seq, 128])")
        print(f"    Transformation: {embeddings.shape[-1]} → {z.shape[-1]} ✓" if z.shape[-1] == 128 else f"    ❌ ERROR: Expected 128, got {z.shape[-1]}")
    model.train()



def train_step(
    model, batch, optimizer, grad_scaler, device, use_amp, accumulation_steps=1, step_idx=0, train_vae_only=False, kl_weight=0.1
):
    """Single training step with gradient accumulation support."""
    context_ids = batch["context_input_ids"].to(device)
    context_mask = batch["context_attention_mask"].to(device)
    question_ids = batch["question_input_ids"].to(device)
    question_mask = batch["question_attention_mask"].to(device)
    answer_ids = batch["answer_input_ids"].to(device)
    answer_mask = batch["answer_attention_mask"].to(device)

    # Zero gradients at the start of accumulation
    if step_idx % accumulation_steps == 0:
        optimizer.zero_grad(set_to_none=True)

    if use_amp and grad_scaler is not None:
        with autocast(device_type=device.type):
            outputs = model(
                context_ids,
                context_mask,
                question_ids,
                question_mask,
                answer_ids,
                answer_mask,
                train_vae_only=train_vae_only,
                kl_weight=kl_weight,
            )
            loss = outputs["loss"] / accumulation_steps
        grad_scaler.scale(loss).backward()

        # Only step optimizer after accumulation
        if (step_idx + 1) % accumulation_steps == 0:
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            grad_scaler.step(optimizer)
            grad_scaler.update()
    else:
        outputs = model(
            context_ids,
            context_mask,
            question_ids,
            question_mask,
            answer_ids,
            answer_mask,
            train_vae_only=train_vae_only,
            kl_weight=kl_weight,
        )
        loss = outputs["loss"] / accumulation_steps
        loss.backward()

        # Only step optimizer after accumulation
        if (step_idx + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    # Compute latent stats if available
    mean_norm = 0.0
    std_mean = 0.0
    if train_vae_only and "mean" in outputs:
        # outputs["mean"] is [batch, seq, dim] or [batch, dim]
        mean_val = outputs["mean"]
        logvar_val = outputs["logvar"]
        std_val = torch.exp(0.5 * logvar_val)
        
        mean_norm = mean_val.norm(dim=-1).mean().item()
        std_mean = std_val.mean().item()

    return {
        "loss": loss.item() * accumulation_steps,  # Return unscaled loss for logging
        "diff_loss": outputs["diffusion_loss"].item(),
        "vae_loss": outputs["vae_loss"].item(),
        "penalty": outputs.get("penalty", torch.tensor(0.0)).item(),
        "mean_norm": mean_norm,
        "std_mean": std_mean,
    }


@torch.no_grad()
def validate(model, val_loader, device, train_vae_only=False, max_metric_batches=50, kl_weight=0.1):
    """
    Validation loop with F1 and EM metrics.
    
    Args:
        max_metric_batches: Number of batches to compute expensive generation metrics for.
                            Loss is computed for all batches.
    """
    model.eval()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    num_batches = 0
    
    all_predictions = []
    all_references = []

    for batch in tqdm(val_loader, desc="Validating"):
        context_ids = batch["context_input_ids"].to(device)
        context_mask = batch["context_attention_mask"].to(device)
        question_ids = batch["question_input_ids"].to(device)
        question_mask = batch["question_attention_mask"].to(device)
        answer_ids = batch["answer_input_ids"].to(device)
        answer_mask = batch["answer_attention_mask"].to(device)

        # 1. Compute Loss
        outputs = model(
            context_ids,
            context_mask,
            question_ids,
            question_mask,
            answer_ids,
            answer_mask,
            train_vae_only=train_vae_only,
            kl_weight=kl_weight,
        )
        total_loss += outputs["loss"].item()
        if "recon_loss" in outputs:
            total_recon_loss += outputs["recon_loss"].item()
            total_kl_loss += outputs["kl_loss"].item()
        num_batches += 1
        
        # 2. Generate/Reconstruct Answers (for metrics)
        # Only for the first N batches to save time
        if num_batches <= max_metric_batches:
            if train_vae_only:
                gen_outputs = model.vae_reconstruct(answer_ids, answer_mask)
            else:
                # Use reduced inference steps for speed (e.g., 20)
                
                gen_outputs = model.generate(
                    context_ids,
                    context_mask,
                    question_ids,
                    question_mask,
                    show_progress=False,
                    num_inference_steps=50
                )
            
            # Decode predictions
            pred_texts = model.decode_tokens_to_text(gen_outputs["tokens"], gen_outputs["is_null"])
            
            # Decode ground truth
            # Identify null answers in ground truth
            null_token_id = model.null_ans_token_id
            is_null_ref = (answer_ids[:, 0] == null_token_id)
            ref_texts = model.decode_tokens_to_text(answer_ids, is_null_ref)
            
            all_predictions.extend(pred_texts)
            all_references.extend(ref_texts)
            
            # Show debug for a random batch (not always batch 0)
            import random
            if num_batches == 1:
                debug_batch_idx = random.randint(1, min(max_metric_batches, 10))
            if num_batches == debug_batch_idx:
                sample_idx = random.randint(0, len(pred_texts) - 1)
                print(f"\n[DEBUG] Random Batch {num_batches}, Sample {sample_idx}:")
                print(f"Answer IDs: {answer_ids[sample_idx, :10].tolist()}")
                print(f"Gen Tokens: {gen_outputs['tokens'][sample_idx, :10].tolist()}")
                print(f"Pred Text: '{pred_texts[sample_idx]}'")
                print(f"Ref Text: '{ref_texts[sample_idx]}'")
                print(f"Is Null Ref: {is_null_ref[sample_idx]}")
                print(f"Is Null Pred: {gen_outputs['is_null'][sample_idx]}")
                # Add latent dimension info for diffusion phase
                if not train_vae_only and 'latent' in gen_outputs:
                    latent = gen_outputs['latent']
                    print(f"Generated Latent Shape: {latent.shape} (expected: [batch, 100, 128])")
                    print(f"Latent Mean: {latent.mean().item():.4f}, Std: {latent.std().item():.4f}")

    avg_loss = total_loss / max(num_batches, 1)
    
    # Compute metrics
    metrics = compute_metrics(all_predictions, all_references)
    metrics["loss"] = avg_loss
    if total_recon_loss > 0:
        metrics["recon_loss"] = total_recon_loss / max(num_batches, 1)
        metrics["kl_loss"] = total_kl_loss / max(num_batches, 1)
    
    # Memory management
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
    
    return metrics


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
            "scaler_mean": model.scaler.mean,
            "scaler_std": model.scaler.std,
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
        
    # Load scaler stats if available
    if "scaler_mean" in checkpoint and checkpoint["scaler_mean"] is not None:
        print("Loading latent scaler stats from checkpoint...")
        model.scaler.mean = checkpoint["scaler_mean"].to(device)
        model.scaler.std = checkpoint["scaler_std"].to(device)
        model.scaler.to(device)
        
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
    parser.add_argument("--vae_warmup_epochs", type=int, default=None)
    parser.add_argument("--vae_patience", type=int, default=3)
    parser.add_argument("--force_vae_warmup", action="store_true", help="Force VAE warmup even if checkpoint exists")
    args = parser.parse_args()

    set_seed(args.seed)
    config = get_config()
    config.training.output_dir = args.output_dir
    config.training.batch_size = args.batch_size
    config.training.num_epochs = args.epochs
    config.training.learning_rate = args.lr
    if args.vae_warmup_epochs is not None:
        config.training.vae_warmup_epochs = args.vae_warmup_epochs
    if args.vae_patience is not None:
        config.training.vae_patience = args.vae_patience

    # Initialize WandB
    wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity,
        name=config.wandb.name or f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        mode=config.wandb.mode,
        config=vars(args),
    )

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
    tokenizer = AutoTokenizer.from_pretrained(config.model.base_encoder)
    config.model.pad_token_id = 0
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
    # Requirement 2: Latent Calibration
    latent_scaler = LatentScaler()
    
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
        false_negative_penalty_weight=config.training.false_negative_penalty_weight,
        scaler=latent_scaler,
    )
    model = model.to(device)
    model.scheduler.to(device)
    
    # Watch model gradients and topology
    wandb.watch(model, log="all", log_freq=100)

    print(f"Trainable parameters: {count_parameters(model):,}")

    # Compute effective steps per epoch
    accumulation_steps = config.training.gradient_accumulation_steps
    steps_per_epoch = len(train_loader) // accumulation_steps
    total_steps = steps_per_epoch * args.epochs

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
    # Only use AMP on CUDA
    use_amp = config.training.use_amp and device.type == "cuda"
    grad_scaler = GradScaler(amp_device) if use_amp else None

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

    # --- Phase 1: VAE Warmup ---
    vae_checkpoint_path = os.path.join(args.output_dir, "vae_warmup_best.pt")
    skip_warmup = False
    
    if os.path.exists(vae_checkpoint_path) and args.use_vae and not args.force_vae_warmup:
        print(f"\n=== Found pre-trained VAE at {vae_checkpoint_path}. Skipping Warmup. ===")
        model.load_state_dict(torch.load(vae_checkpoint_path, map_location=device))
        skip_warmup = True
    
    if not skip_warmup and config.training.vae_warmup_epochs > 0 and args.use_vae:
        print(f"\n=== Starting VAE Warmup Phase ({config.training.vae_warmup_epochs} epochs) ===")
        best_vae_loss = float("inf")
        patience_counter = 0
        
        for epoch in range(config.training.vae_warmup_epochs):
            # Debug dimensions at start of first epoch with random batch
            if epoch == 0:
                # Get a random batch (skip random number of batches)
                import random
                skip_count = random.randint(0, min(50, len(train_loader) - 1))
                random_batch = None
                for i, batch in enumerate(train_loader):
                    if i == skip_count:
                        random_batch = batch
                        break
                debug_dimensions(model, random_batch, device, epoch + 1)
            
            model.train()
            epoch_vae_loss = 0.0
            
            pbar = tqdm(train_loader, desc=f"Warmup Epoch {epoch+1}/{config.training.vae_warmup_epochs}")
            
            # Calculate total warmup steps
            total_warmup_steps = config.training.vae_warmup_epochs * len(train_loader)
            
            for batch_idx, batch in enumerate(pbar):
                # Calculate KL weight for this step
                current_step = epoch * len(train_loader) + batch_idx
                current_kl = get_kl_weight(current_step, total_warmup_steps, target_kl=0.1, cycles=4)
                
                metrics = train_step(
                    model,
                    batch,
                    optimizer,
                    grad_scaler,
                    device,
                    use_amp,
                    accumulation_steps=accumulation_steps,
                    step_idx=batch_idx,
                    train_vae_only=True,
                    kl_weight=current_kl,
                )
                
                # Only step scheduler after full accumulation
                if (batch_idx + 1) % accumulation_steps == 0:
                    scheduler.step()
                
                epoch_vae_loss += metrics["vae_loss"]
                
                pbar.set_postfix(
                    {
                        "vae_loss": f"{metrics['vae_loss']:.4f}",
                        "vae_loss": f"{metrics['vae_loss']:.4f}",
                        "kl_w": f"{current_kl:.4f}",
                        "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                    }
                )
                # Increment global step for VAE warmup too, or keep separate?
                # User wants tracking on both. Let's log a step for VAE.
                # We'll use a separate counter or the same global_step?
                # If we use the same global_step, it might be confusing if we resume?
                # Let's just log 'warmup/step' for now to be safe and explicit.
                
                wandb.log(
                    {
                        "warmup/vae_loss": metrics["vae_loss"],
                        "warmup/kl_weight": current_kl,
                        "warmup/lr": scheduler.get_last_lr()[0],
                        "warmup/epoch": epoch,
                        "warmup/step": batch_idx + epoch * len(train_loader),
                    }
                )
            
            avg_vae_loss = epoch_vae_loss / len(train_loader)
            
            # Validate VAE
            val_metrics = validate(model, val_loader, device, train_vae_only=True, max_metric_batches=50)
            val_vae_loss = val_metrics["loss"]
            val_f1 = val_metrics["f1"]
            val_em = val_metrics["em"]
            val_recon = val_metrics.get("recon_loss", 0.0)
            val_kl = val_metrics.get("kl_loss", 0.0)
            
            print(f"Warmup Epoch {epoch+1}: Train VAE Loss = {avg_vae_loss:.4f}, Val VAE Loss = {val_vae_loss:.4f} (Recon={val_recon:.4f}, KL={val_kl:.4f}), F1 = {val_f1:.2f}, EM = {val_em:.2f}")
            
            wandb.log({
                "warmup/val_vae_loss": val_vae_loss,
                "warmup/val_recon_loss": val_recon,
                "warmup/val_kl_loss": val_kl,
                "warmup/val_f1": val_f1,
                "warmup/val_em": val_em,
                "warmup/epoch": epoch
            })
            
            # Check for convergence/early stopping based on VALIDATION loss
            if val_vae_loss < best_vae_loss:
                best_vae_loss = val_vae_loss
                patience_counter = 0
                # Save warmup checkpoint
                torch.save(model.state_dict(), os.path.join(args.output_dir, "vae_warmup_best.pt"))
                wandb.save(os.path.join(args.output_dir, "vae_warmup_best.pt"))
            else:
                patience_counter += 1
            
            print(f"Patience: {patience_counter}/{config.training.vae_patience} (Best Val Loss: {best_vae_loss:.4f})")
                
            if patience_counter >= config.training.vae_patience:
                print(f"VAE converged after {epoch+1} epochs (patience reached).")
                break
        
        if not skip_warmup:
            # Load best VAE weights
            print("Loading best VAE weights from warmup...")
            model.load_state_dict(torch.load(os.path.join(args.output_dir, "vae_warmup_best.pt")))

    # Requirement 2: Latent Calibration
    # Fit scaler on training data using the trained VAE
    if args.use_vae:
        print("\n=== Fitting Latent Scaler ===")
        latent_scaler.fit(train_loader, model.vae, device)

    # --- Phase 2: Diffusion Training (Frozen VAE) ---
    print("\n=== Starting Diffusion Training Phase ===")
    
    # Freeze VAE if using it
    if args.use_vae:
        print("Freezing VAE parameters...")
        for p in model.vae.parameters():
            p.requires_grad = False
            
        # Re-initialize optimizer for diffusion only
        # We need to filter out frozen parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
        
        # Re-init scheduler for diffusion phase with its own T_max
        # Use a small warmup for the diffusion phase to stabilize the denoiser
        diffusion_steps = steps_per_epoch * args.epochs
        scheduler = CosineAnnealingLR(optimizer, T_max=diffusion_steps)
        
        print(f"Trainable parameters (Diffusion only): {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Reset best val loss for diffusion phase
    best_val_loss = float("inf")

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        # Debug dimensions at start of first diffusion epoch with random batch
        if epoch == start_epoch:
            import random
            skip_count = random.randint(0, min(50, len(train_loader) - 1))
            random_batch = None
            for i, batch in enumerate(train_loader):
                if i == skip_count:
                    random_batch = batch
                    break
            debug_dimensions(model, random_batch, device, f"Diffusion-{epoch + 1}")
        
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_idx, batch in enumerate(pbar):
            metrics = train_step(
                model,
                batch,
                optimizer,
                grad_scaler,
                device,
                use_amp,
                accumulation_steps=accumulation_steps,
                step_idx=batch_idx,
                train_vae_only=False,
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
                    "pen": f"{metrics['penalty']:.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                }
            )
            wandb.log(
                {
                    "train/loss": metrics["loss"],
                    "train/diffusion_loss": metrics["diff_loss"],
                    "train/vae_loss": metrics["vae_loss"],
                    "train/penalty": metrics["penalty"],
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/epoch": epoch,
                    "train/global_step": global_step,
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
                    os.path.join(args.output_dir, "checkpoint-last.pt"),
                )

        avg_train_loss = epoch_loss / len(train_loader)
        val_metrics = validate(model, val_loader, device, max_metric_batches=50)
        val_loss = val_metrics["loss"]
        val_f1 = val_metrics["f1"]
        val_em = val_metrics["em"]

        print(
            f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}, F1 = {val_f1:.2f}, EM = {val_em:.2f}"
        )
        wandb.log({
            "val/loss": val_loss, 
            "val/f1": val_f1,
            "val/em": val_em,
            "epoch": epoch
        })

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
            # Upload to WandB
            wandb.save(os.path.join(args.output_dir, "best_model.pt"))

    print("Training complete!")
    wandb.finish()


if __name__ == "__main__":
    main()
