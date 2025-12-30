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
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm

from config import Config, get_config
from data import create_dataloader
from models import LatentDiffusionQA
from models.scaler import LatentScaler
from models.scaler import LatentScaler
from metrics import compute_metrics


def get_kl_weight(current_step: int, total_steps: int, target_kl: float = 0.01, cycles: int = 4) -> float:
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
    
    # Start from 10% of target, not 0, to prevent early collapse
    min_kl = target_kl * 0.1
    return min_kl + (target_kl - min_kl) * (current_cycle_step / max(1, warmup_steps))



def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_grad_norm(model: nn.Module) -> float:
    """Calculate L2 norm of all gradients combined."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def log_vital_signs(model, batch, vae_output, diffusion_output, step: int, epoch: int, grad_norm: Optional[float] = None):
    """
    Log comprehensive vital signs for VAE-Diffusion system health monitoring.
    
    Args:
        model: The LatentDiffusionQA model
        batch: Training batch with inputs and targets
        vae_output: Output from VAE forward pass
        diffusion_output: Output from diffusion forward pass
        step: Current training step
        epoch: Current epoch number
        grad_norm: Unscaled gradient norm (if available from optimizer step)
    """
    device = next(model.parameters()).device
    
    # 1. VAE Heartbeat Metrics
    answer_ids = batch["answer_input_ids"].to(device)
    valid_tokens = (answer_ids != model.pad_token_id).sum()
    total_tokens = answer_ids.numel()
    
    # Properly normalized reconstruction loss (per valid token)
    recon_loss_per_valid = vae_output["recon_loss"].item()
    
    # KL divergence metrics
    kl_loss = vae_output.get("kl_loss", torch.tensor(0.0)).item()
    
    # Active latent dimensions (where variance > 0.1)
    z = vae_output.get("z", None)
    if z is not None:
        latent_variance = z.var(dim=0)  # Variance per dimension
        active_dims = (latent_variance > 0.1).sum().item()
        latent_mean = z.mean().item()
        latent_std = z.std().item()
    else:
        active_dims = 0
        latent_mean = 0.0
        latent_std = 0.0
    
    # 2. Latent Distribution Health
    if model.scaler is not None and z is not None:
        # Get scaled latents (what diffusion model sees)
        z_scaled = model.scaler.transform(z)
        scaled_mean = z_scaled.mean().item()
        scaled_std = z_scaled.std().item()
    else:
        scaled_mean = latent_mean
        scaled_std = latent_std
    
    # 3. Diffusion Dynamics
    diff_loss = diffusion_output.get("loss", torch.tensor(0.0)).item()
    
    # Prediction magnitude (norm of model output)
    pred_noise = diffusion_output.get("pred_noise", None)
    if pred_noise is not None:
        pred_magnitude = pred_noise.norm().item()
    else:
        pred_magnitude = 0.0
    
    # 4. Task-Specific Metrics (Null Prediction Health)
    # Calculate null prediction rate (expensive, so sample)
    if step % 100 == 0:  # Every 100 steps to save compute
        with torch.no_grad():
            # Quick inference on a small sample
            sample_size = min(4, len(batch["answer_input_ids"]))
            sample_answer_ids = batch["answer_input_ids"][:sample_size].to(device)
            sample_answer_mask = batch["answer_attention_mask"][:sample_size].to(device)
            sample_context_ids = batch["context_input_ids"][:sample_size].to(device)
            sample_context_mask = batch["context_attention_mask"][:sample_size].to(device)
            sample_question_ids = batch["question_input_ids"][:sample_size].to(device)
            sample_question_mask = batch["question_attention_mask"][:sample_size].to(device)
            
            try:
                # Use model.generate for proper null prediction check
                # We use a reduced number of steps for speed
                sample_outputs = model.generate(
                    sample_context_ids, sample_context_mask,
                    sample_question_ids, sample_question_mask,
                    num_inference_steps=10,  # Fast check
                    show_progress=False
                )
                
                # Null prediction rate
                is_null = sample_outputs.get("is_null", None)
                if is_null is not None:
                    null_prediction_rate = is_null.float().mean().item()
                else:
                    null_prediction_rate = 0.0
                
                # Null cosine similarity
                null_similarity = sample_outputs.get("null_similarity", None)
                if null_similarity is not None:
                    avg_null_sim = null_similarity.mean().item()
                else:
                    avg_null_sim = 0.0
                    
            except Exception as e:
                # Fallback if inference fails
                print(f"Vital signs inference failed: {e}")
                null_prediction_rate = 0.0
                avg_null_sim = 0.0
    else:
        null_prediction_rate = None
        avg_null_sim = None
    
    # 5. Gradient Health
    # If grad_norm is not provided (e.g. accumulation step), use a placeholder or previous?
    # Better to just log what we have. If 0.0, it means no update.
    if grad_norm is None:
        grad_norm = 0.0
    
    # Compile vital signs log
    vital_logs = {
        # VAE Heartbeat
        "vital/vae_recon_per_valid": recon_loss_per_valid,
        "vital/vae_kl_divergence": kl_loss,
        "vital/vae_active_latent_dims": active_dims,
        
        # Latent Distribution
        "vital/latent_mean": latent_mean,
        "vital/latent_std": latent_std,
        "vital/scaled_mean": scaled_mean,
        "vital/scaled_std": scaled_std,
        
        # Diffusion Dynamics
        "vital/diff_mse_loss": diff_loss,
        "vital/diff_pred_magnitude": pred_magnitude,
        
        # Gradient Health
        "vital/grad_norm": grad_norm,
        
        # Basic stats
        "vital/valid_tokens": valid_tokens.item(),
        "vital/total_tokens": total_tokens,
        "vital/valid_token_ratio": valid_tokens.float() / total_tokens,
    }
    
    # Add task-specific metrics if available
    if null_prediction_rate is not None:
        vital_logs["vital/null_prediction_rate"] = null_prediction_rate
    if avg_null_sim is not None:
        vital_logs["vital/null_cosine_sim"] = avg_null_sim
    
    # Log to wandb (no explicit step - let wandb auto-increment to avoid step conflicts)
    wandb.log(vital_logs)
    
    # Check for kill signals and print warnings
    check_kill_signals(vital_logs, step, epoch)


def check_kill_signals(logs: dict, step: int, epoch: int):
    """
    Check for critical failure patterns and print warnings.
    """
    warnings = []
    
    # VAE Health Checks
    if logs["vital/vae_recon_per_valid"] > 6.0 and epoch > 0:
        warnings.append(f"🚨 HIGH RECON LOSS: {logs['vital/vae_recon_per_valid']:.2f} > 6.0 (VAE not learning)")
    
    if logs["vital/vae_kl_divergence"] < 0.001:
        warnings.append(f"🚨 POSTERIOR COLLAPSE: KL {logs['vital/vae_kl_divergence']:.4f} < 0.001")
    elif logs["vital/vae_kl_divergence"] > 100.0:
        warnings.append(f"🚨 KL EXPLOSION: KL {logs['vital/vae_kl_divergence']:.2f} > 100.0")
    
    if logs["vital/vae_active_latent_dims"] < 10:
        warnings.append(f"🚨 FEW ACTIVE DIMS: {logs['vital/vae_active_latent_dims']} < 10 (wasted capacity)")
    
    # Latent Distribution Checks
    if abs(logs["vital/scaled_mean"]) > 0.5:
        warnings.append(f"🚨 CENTERING ISSUE: Scaled mean {logs['vital/scaled_mean']:.3f} > 0.5")
    
    if logs["vital/scaled_std"] > 1.5 or logs["vital/scaled_std"] < 0.5:
        warnings.append(f"🚨 SCALING ISSUE: Scaled std {logs['vital/scaled_std']:.3f} not in [0.5, 1.5]")
    
    # Gradient Health Checks
    if logs["vital/grad_norm"] > 10.0:
        warnings.append(f"🚨 GRADIENT EXPLOSION: Norm {logs['vital/grad_norm']:.2f} > 10.0")
    elif logs["vital/grad_norm"] < 1e-4 and logs["vital/grad_norm"] > 0.0:
        warnings.append(f"🚨 VANISHING GRADIENTS: Norm {logs['vital/grad_norm']:.6f} < 1e-4")
    
    # Task-Specific Checks
    if "vital/null_prediction_rate" in logs:
        if logs["vital/null_prediction_rate"] > 0.95:
            warnings.append(f"🚨 NULL BIAS: {logs['vital/null_prediction_rate']:.1%} > 95% saying 'no answer'")
        elif logs["vital/null_prediction_rate"] < 0.05 and epoch > 0:
            warnings.append(f"🚨 IGNORING NULL: {logs['vital/null_prediction_rate']:.1%} < 5% unanswerable predictions")
    
    # Print warnings if any
    if warnings:
        print(f"\n⚠️  VITAL SIGNS WARNING (Step {step}, Epoch {epoch}):")
        for warning in warnings:
            print(f"   {warning}")
        print("   Consider stopping the run to investigate!")


def log_token_accuracy(model, batch, vae_output, step: int):
    """
    Log detailed token-level accuracy metrics.
    """
    device = next(model.parameters()).device
    
    # Get predictions and targets
    answer_ids = batch["answer_input_ids"].to(device)
    attention_mask = batch["answer_attention_mask"].to(device)
    
    # Get VAE reconstruction
    z = vae_output["z"]
    decoded = model.vae.decode(z)
    embed_weight = model.vae.embeddings.weight
    logits = torch.matmul(decoded, embed_weight.T)
    pred_ids = logits.argmax(dim=-1)
    
    # Calculate token-level accuracy (only on non-padding tokens)
    valid_mask = attention_mask.bool()
    correct_tokens = (pred_ids == answer_ids) & valid_mask
    token_accuracy = correct_tokens.sum().float() / valid_mask.sum().float()
    
    # Exact match accuracy (entire sequence)
    exact_match = ((pred_ids == answer_ids) | ~valid_mask).all(dim=1).float().mean()
    
    # Character-level accuracy
    pred_texts = [model.tokenizer.decode(p, skip_special_tokens=True) for p in pred_ids]
    target_texts = [model.tokenizer.decode(t, skip_special_tokens=True) for t in answer_ids]
    
    char_accuracies = []
    for pred, target in zip(pred_texts, target_texts):
        if target.strip():
            pred_chars = set(pred.lower())
            target_chars = set(target.lower())
            if target_chars:
                char_acc = len(pred_chars & target_chars) / len(target_chars)
                char_accuracies.append(char_acc)
    
    avg_char_accuracy = sum(char_accuracies) / len(char_accuracies) if char_accuracies else 0.0
    
    # Log token accuracy metrics (no explicit step - let wandb auto-increment)
    token_logs = {
        "token/accuracy": token_accuracy.item(),
        "token/exact_match": exact_match.item(),
        "token/char_accuracy": avg_char_accuracy,
    }
    wandb.log(token_logs)
    
    return token_accuracy.item(), exact_match.item(), avg_char_accuracy


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
        if hasattr(model.vae, "latent_seq_len"): # SequenceVAE
            z, mean, logvar, l_mask = model.vae.encode(answer_ids, answer_mask)
        else: # EmbeddingBridge
            z = model.vae.encode(answer_ids)
        
        print(f"\n[DEBUG Epoch {epoch_num}] Dimension Verification:")
        print(f"    Sample Answer: '{answer_text[:50]}...' " if len(answer_text) > 50 else f"    Sample Answer: '{answer_text}'")
        print(f"    BERT Embeddings: {embeddings.shape} (expected: [1, seq, 768])")
        print(f"    VAE Latent (z):  {z.shape} (expected: [1, 8, 768] for VAE or [1, seq, 768] for Bridge)")
        print(f"    Transformation: {embeddings.shape[-1]} → {z.shape[-1]} ✓" if z.shape[-1] == 768 else f"    ❌ ERROR: Expected 768, got {z.shape[-1]}")
    model.train()



def train_step(
    model, batch, optimizer, grad_scaler, device, use_amp, accumulation_steps=1, step_idx=0, train_vae_only=False, kl_weight=1e-5, global_step=0, epoch=0
):
    """Single training step with gradient accumulation support and vital signs monitoring."""
    context_ids = batch["context_input_ids"].to(device)
    context_mask = batch["context_attention_mask"].to(device)
    question_ids = batch["question_input_ids"].to(device)
    question_mask = batch["question_attention_mask"].to(device)
    answer_ids = batch["answer_input_ids"].to(device)
    answer_mask = batch["answer_attention_mask"].to(device)

    # Zero gradients at the start of accumulation
    if step_idx % accumulation_steps == 0:
        optimizer.zero_grad(set_to_none=True)

    current_grad_norm = None

    if use_amp and grad_scaler is not None:
        with autocast(device_type=device.type, dtype=torch.bfloat16):
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
            current_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            # Estimate unscaled norm for monitoring
            scale = grad_scaler.get_scale()
            current_grad_norm = get_grad_norm(model) / scale
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
            current_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
            optimizer.step()
        else:
            current_grad_norm = get_grad_norm(model)

    # Extract VAE and diffusion outputs for monitoring accurately
    vae_output = {
        "recon_loss": outputs.get("recon_loss", torch.tensor(0.0)),
        "kl_loss": outputs.get("kl_loss", torch.tensor(0.0)),
        "z": outputs.get("z", None),
        "mean": outputs.get("mean", None),
        "logvar": outputs.get("logvar", None),
    }
    
    diffusion_output = {
        "loss": outputs.get("diffusion_loss", torch.tensor(0.0)),
        "pred_noise": outputs.get("pred_noise", None),
    }

    # Log vital signs and token accuracy periodically (every log_every steps)
    log_every = 100 
    
    if global_step % log_every == 0:
        # During warmup, step_idx + 1 might not align with accumulation, 
        # but we want vital signs anyway to monitor health.
        log_vital_signs(model, batch, vae_output, diffusion_output, global_step, epoch, grad_norm=current_grad_norm)
        
        # Log token accuracy if VAE is active
        if vae_output["z"] is not None:
            log_token_accuracy(model, batch, vae_output, global_step)

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
        "grad_norm": current_grad_norm if current_grad_norm is not None else 0.0,
    }


@torch.no_grad()
def validate(model, val_loader, device, train_vae_only=False, max_metric_batches=20, kl_weight=1e-5, global_step=0, epoch=0):
    """
    Validation loop with F1 and EM metrics and vital signs monitoring.
    
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
    debug_sample = None
    import random
    debug_batch_idx = random.randint(1, max_metric_batches)

    for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validating")):
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
        
        # Extract VAE and diffusion outputs for monitoring
        vae_output = {
            "recon_loss": outputs.get("vae_recon_loss", torch.tensor(0.0)),
            "kl_loss": outputs.get("vae_kl_loss", torch.tensor(0.0)),
            "z": outputs.get("z", None),
            "mean": outputs.get("mean", None),
            "logvar": outputs.get("logvar", None),
        }
        
        diffusion_output = {
            "loss": outputs.get("diffusion_loss", torch.tensor(0.0)),
            "pred_noise": outputs.get("pred_noise", None),
        }

        # Log vital signs for validation (sample every 10 batches to avoid spam)
        if batch_idx % 10 == 0:
            log_vital_signs(model, batch, vae_output, diffusion_output, global_step + batch_idx, epoch)
            
            # Log token accuracy every 50 validation steps
            if batch_idx % 50 == 0 and vae_output["z"] is not None:
                log_token_accuracy(model, batch, vae_output, global_step + batch_idx)
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
                    num_inference_steps=20,
                    guidance_scale=get_config().inference.guidance_scale,
                )
            
            # Decode predictions
            pred_texts = model.decode_tokens_to_text(gen_outputs["tokens"], gen_outputs["is_null"])
            
            # Decode ground truth
            # Identify null answers in ground truth
            null_token_id = model.null_ans_token_id
            is_null_ref = (answer_ids[:, 1] == null_token_id)
            ref_texts = model.decode_tokens_to_text(answer_ids, is_null_ref)
            
            all_predictions.extend(pred_texts)
            all_references.extend(ref_texts)
            
            if num_batches == max_metric_batches:
                print(f"\n[INFO] Completed generation for {max_metric_batches} batches. Switching to fast loss-only validation for remaining batches...")
            
            if num_batches == debug_batch_idx:
                sample_idx = random.randint(0, len(pred_texts) - 1)
                debug_sample = {
                    "batch_idx": num_batches,
                    "sample_idx": sample_idx,
                    "answer_ids": answer_ids[sample_idx, :10].tolist(),
                    "gen_tokens": gen_outputs['tokens'][sample_idx, :10].tolist(),
                    "pred_text": pred_texts[sample_idx],
                    "ref_text": ref_texts[sample_idx],
                    "is_null_ref": is_null_ref[sample_idx].item(),
                    "is_null_pred": gen_outputs['is_null'][sample_idx].item(),
                }
                if not train_vae_only and 'latent' in gen_outputs:
                    latent = gen_outputs['latent']
                    debug_sample.update({
                        "latent_shape": list(latent.shape),
                        "latent_mean": latent.mean().item(),
                        "latent_std": latent.std().item(),
                    })
            

    avg_loss = total_loss / max(num_batches, 1)
    
    # Print debug info after loop
    if debug_sample:
        print(f"\n[DEBUG] Random Batch {debug_sample['batch_idx']}, Sample {debug_sample['sample_idx']}:")
        print(f"Answer IDs: {debug_sample['answer_ids']}")
        print(f"Gen Tokens: {debug_sample['gen_tokens']}")
        print(f"Pred Text: '{debug_sample['pred_text']}'")
        print(f"Ref Text: '{debug_sample['ref_text']}'")
        print(f"Is Null Ref: {debug_sample['is_null_ref']}")
        print(f"Is Null Pred: {debug_sample['is_null_pred']}")
        if "latent_shape" in debug_sample:
            print(f"Generated Latent Shape: {debug_sample['latent_shape']}")
            print(f"Latent Mean: {debug_sample['latent_mean']:.4f}, Std: {debug_sample['latent_std']:.4f}")
    
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
    # Disable TF32 for more stable AMP training
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # torch.backends.cuda.matmul.allow_tf32 = True
    # torch.backends.cudnn.allow_tf32 = True
    # Disable nested tensors for transformer layers to avoid prototype bugs
    torch._C._jit_set_texpr_fuser_enabled(False)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default="data/train-v2.0.json")
    parser.add_argument("--dev_file", type=str, default="data/dev-v2.0.json")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (overrides config if set)")
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
    if args.lr is not None:
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
        prediction_type=config.diffusion.prediction_type,
    )
    model = model.to(device)  # This now also moves scheduler efficiently
    
    # Watch model gradients and topology
    wandb.watch(model, log="all", log_freq=100)

    print(f"Trainable parameters: {count_parameters(model):,}")

    # Compute effective steps per epoch
    accumulation_steps = config.training.gradient_accumulation_steps
    steps_per_epoch = len(train_loader) // accumulation_steps
    total_steps = steps_per_epoch * args.epochs

    # Optimizer and scheduler for Phase 1 (VAE Warmup)
    optimizer = AdamW(model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay)
    
    # VAE Warmup total steps
    vae_total_steps = steps_per_epoch * config.training.vae_warmup_epochs
    
    # Use linear warmup + cosine decay
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=config.training.warmup_steps, 
        num_training_steps=vae_total_steps if vae_total_steps > 0 else 1
    )
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
                # Constant KL weight for "Wide & Loose" VAE
                current_kl = 1e-5
                
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
                    global_step=current_step,
                    epoch=epoch,
                )
                
                # Only step scheduler after full accumulation
                if (batch_idx + 1) % accumulation_steps == 0:
                    scheduler.step()
                
                epoch_vae_loss += metrics["vae_loss"]
                
                pbar.set_postfix(
                    {
                        "vae_loss": f"{metrics['vae_loss']:.4f}",
                        "kl_w": f"{current_kl:.4f}",
                        "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                        "gn": f"{metrics['grad_norm']:.2f}",
                    }
                )
                # Increment global step for VAE warmup too, or keep separate?
                # User wants tracking on both. Let's log a step for VAE.
                # We'll use a separate counter or the same global_step?
                # If we use the same global_step, it might be confusing if we resume?
                # Let's just log 'warmup/step' for now to be safe and explicit.
                
                # Log warmup metrics every log_every steps to avoid console spam
                log_every = 100
                if current_step % log_every == 0:
                    wandb.log(
                        {
                            "warmup/vae_loss": metrics["vae_loss"],
                            "warmup/kl_weight": current_kl,
                            "warmup/lr": scheduler.get_last_lr()[0],
                            "warmup/grad_norm": metrics["grad_norm"],
                            "warmup/epoch": epoch,
                            "warmup/step": current_step,
                        }
                    )
            
            avg_vae_loss = epoch_vae_loss / len(train_loader)
            
            # Validate VAE
            val_metrics = validate(model, val_loader, device, train_vae_only=True, max_metric_batches=20, global_step=current_step, epoch=epoch)
            val_vae_loss = val_metrics["loss"]
            val_f1 = val_metrics["f1"]
            val_em = val_metrics["em"]
            val_recon = val_metrics.get("recon_loss", 0.0)
            val_kl = val_metrics.get("kl_loss", 0.0)
            
            print(f"Warmup Epoch {epoch+1}: Train VAE Loss = {avg_vae_loss:.4f}, Val VAE Loss = {val_vae_loss:.4f} (Recon={val_recon:.4f}, KL={val_kl:.4f}), F1 = {val_f1:.2f}, EM = {val_em:.2f}")
            print(f"    HasAns: F1 = {val_metrics.get('has_ans_f1', 0):.2f}, EM = {val_metrics.get('has_ans_em', 0):.2f}")
            print(f"    NoAns:  F1 = {val_metrics.get('no_ans_f1', 0):.2f}, EM = {val_metrics.get('no_ans_em', 0):.2f}")
            
            wandb.log({
                "warmup/val_vae_loss": val_vae_loss,
                "warmup/val_recon_loss": val_recon,
                "warmup/val_kl_loss": val_kl,
                "warmup/val_f1": val_f1,
                "warmup/val_em": val_em,
                "warmup/val_has_ans_f1": val_metrics.get("has_ans_f1", 0),
                "warmup/val_no_ans_f1": val_metrics.get("no_ans_f1", 0),
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
        optimizer = AdamW(trainable_params, lr=config.training.learning_rate, weight_decay=config.training.weight_decay)
        
        # Re-init scheduler for diffusion phase with its own T_max
        diffusion_steps = steps_per_epoch * args.epochs
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.training.warmup_steps,
            num_training_steps=diffusion_steps if diffusion_steps > 0 else 1
        )
        
        # If we resumed, we need to reload the optimizer/scheduler state for the diffusion phase
        if args.resume:
            print("Reloading optimizer/scheduler state for diffusion phase...")
            checkpoint = torch.load(args.resume, map_location=device)
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"]:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
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
            current_step = epoch * len(train_loader) + batch_idx
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
                global_step=current_step,
                epoch=epoch,
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
                    "pen": f"{metrics['penalty']:.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                    "gn": f"{metrics['grad_norm']:.2f}",
                }
            )
            wandb.log(
                {
                    "train/loss": metrics["loss"],
                    "train/diffusion_loss": metrics["diff_loss"],
                    "train/penalty": metrics["penalty"],
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/grad_norm": metrics["grad_norm"],
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
        val_metrics = validate(model, val_loader, device, max_metric_batches=20, global_step=global_step, epoch=epoch)
        val_loss = val_metrics["loss"]
        val_f1 = val_metrics["f1"]
        val_em = val_metrics["em"]

        print(
            f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}, F1 = {val_f1:.2f}, EM = {val_em:.2f}"
        )
        print(f"    HasAns: F1 = {val_metrics['has_ans_f1']:.2f}, EM = {val_metrics['has_ans_em']:.2f} (Count: {val_metrics['total_has_ans']})")
        print(f"    NoAns:  F1 = {val_metrics['no_ans_f1']:.2f}, EM = {val_metrics['no_ans_em']:.2f} (Count: {val_metrics['total_no_ans']})")
        print(f"    Preds:  HasAns = {val_metrics['pred_has_ans']}, NoAns = {val_metrics['pred_no_ans']}")

        wandb.log({
            "val/loss": val_loss, 
            "val/f1": val_f1,
            "val/em": val_em,
            "val/has_ans_f1": val_metrics["has_ans_f1"],
            "val/has_ans_em": val_metrics["has_ans_em"],
            "val/no_ans_f1": val_metrics["no_ans_f1"],
            "val/no_ans_em": val_metrics["no_ans_em"],
            "val/pred_has_ans": val_metrics["pred_has_ans"],
            "val/pred_no_ans": val_metrics["pred_no_ans"],
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
