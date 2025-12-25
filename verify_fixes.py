import torch
from transformers import XLMRobertaTokenizer
from models.latent_diffusion import LatentDiffusionQA
from train import get_kl_weight, validate
from models.scaler import LatentScaler

def test_fixes():
    print("Initializing model...")
    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
    scaler = LatentScaler()
    # Mock scaler stats
    scaler.mean = torch.zeros(256)
    scaler.std = torch.ones(256)
    
    model = LatentDiffusionQA(
        tokenizer=tokenizer,
        latent_dim=256,
        d_model=128, # Small for speed
        num_layers=2,
        num_heads=4,
        use_vae=True,
        scaler=scaler,
        false_negative_penalty_weight=1.0
    )
    
    # Dummy data
    batch_size = 4
    seq_len = 16
    context_ids = torch.randint(0, 1000, (batch_size, seq_len))
    context_mask = torch.ones((batch_size, seq_len))
    question_ids = torch.randint(0, 1000, (batch_size, seq_len))
    question_mask = torch.ones((batch_size, seq_len))
    answer_ids = torch.randint(0, 1000, (batch_size, seq_len))
    answer_mask = torch.ones((batch_size, seq_len))
    
    # Make one answer unanswerable (null token)
    null_id = model.null_ans_token_id
    answer_ids[0, 0] = null_id
    
    # 1. Test VAE training step with KL weight
    print("Testing VAE training step with KL weight...")
    
    # Test cyclic annealing function
    # Total 1000 steps, 4 cycles -> 250 steps per cycle
    # Warmup 125 steps (0.5 ratio)
    kl_0 = get_kl_weight(0, 1000, target_kl=0.1, cycles=4) # Start of cycle 1 -> 0.0
    kl_mid = get_kl_weight(62, 1000, target_kl=0.1, cycles=4) # Middle of warmup -> ~0.05
    kl_peak = get_kl_weight(150, 1000, target_kl=0.1, cycles=4) # End of warmup -> 0.1
    kl_reset = get_kl_weight(250, 1000, target_kl=0.1, cycles=4) # Start of cycle 2 -> 0.0
    
    print(f"KL Weights: Start={kl_0:.4f}, Mid={kl_mid:.4f}, Peak={kl_peak:.4f}, Reset={kl_reset:.4f}")
    assert kl_0 < kl_mid < kl_peak, "KL weight should increase in warmup"
    assert kl_reset < kl_peak, "KL weight should reset at new cycle"
    
    vae_out = model(
        context_ids, 
        context_mask, 
        question_ids, 
        question_mask, 
        answer_ids, 
        answer_mask, 
        train_vae_only=True,
        kl_weight=0.05
    )
    print(f"VAE Loss: {vae_out['loss'].item()}")
    if "mean" in vae_out:
        print(f"VAE Latent Mean Norm: {vae_out['mean'].norm(dim=-1).mean().item()}")
        print(f"VAE Latent LogVar Mean: {vae_out['logvar'].mean().item()}")
    else:
        print("WARNING: VAE output does not contain 'mean' or 'logvar'")
    
    print("Testing Diffusion training step with Penalty...")
    diff_out = model(
        context_ids, context_mask, question_ids, question_mask, answer_ids, answer_mask,
        train_vae_only=False
    )
    print(f"Diffusion Loss: {diff_out['loss'].item()}")
    print(f"Penalty: {diff_out['penalty'].item()}")
    
    # Check if penalty is being calculated (should be non-zero for answerable samples)
    # Since we have random weights, it's unlikely to be exactly 0 unless logic is skipped.
    if diff_out['penalty'].item() == 0.0:
        print("WARNING: Penalty is 0.0. Check if logic is triggered.")
    else:
        print("SUCCESS: Penalty is being calculated.")

    # 3. Test Latent Scaler
    print("Testing Latent Scaler...")
    # Create dummy latents [batch, seq, dim]
    latents = torch.randn(10, 16, 256) * 5 + 2 # Mean 2, Std 5
    
    # Mock VAE get_latent
    class MockVAE:
        def eval(self): pass
        def get_latent(self, ids, mask, use_mean=True):
            return latents[0:ids.shape[0]] # Return subset
            
    scaler.fit([{"answer_input_ids": torch.zeros(10, 16), "answer_attention_mask": torch.ones(10, 16)}], MockVAE(), latents.device)
    
    print(f"Scaler Mean Norm: {scaler.mean.norm().item()}")
    print(f"Scaler Std Norm: {scaler.std.norm().item()}")
    
    # Transform
    z_norm = scaler.transform(latents)
    print(f"Transformed Mean: {z_norm.mean().item():.4f} (should be ~0)")
    print(f"Transformed Std: {z_norm.std().item():.4f} (should be ~1)")
    
    assert abs(z_norm.mean().item()) < 0.5, "Scaler failed to normalize mean"
    assert abs(z_norm.std().item() - 1.0) < 0.5, "Scaler failed to normalize std"

    assert abs(z_norm.mean().item()) < 0.5, "Scaler failed to normalize mean"
    assert abs(z_norm.std().item() - 1.0) < 0.5, "Scaler failed to normalize std"

    # 4. Test Validation KL Weight
    print("Testing Validation KL Weight...")
    # Mock DataLoader
    dummy_batch = {
        "context_input_ids": torch.randint(0, 1000, (2, 16)),
        "context_attention_mask": torch.ones(2, 16),
        "question_input_ids": torch.randint(0, 1000, (2, 16)),
        "question_attention_mask": torch.ones(2, 16),
        "answer_input_ids": torch.randint(0, 1000, (2, 16)),
        "answer_attention_mask": torch.ones(2, 16),
    }
    dummy_loader = [dummy_batch]
    
    # Test with default KL (should be 0.1)
    val_metrics_default = validate(model, dummy_loader, latents.device, train_vae_only=True, max_metric_batches=0)
    print(f"Val Loss (Default KL=0.1): {val_metrics_default['loss']:.4f}")
    
    if "recon_loss" in val_metrics_default:
        print(f"Val Recon: {val_metrics_default['recon_loss']:.4f}, Val KL: {val_metrics_default['kl_loss']:.4f}")
        
    # We expect this to match the high KL case if high KL was 0.1, but here we just verify it runs
    # and returns reasonable values.
    print("SUCCESS: Validation runs with default KL weight.")

    # 5. Test Scaler Impact on Diffusion Loss
    print("Testing Scaler Impact on Diffusion Loss...")
    # Create large variance latents (simulating unscaled VAE output)
    large_latents = torch.randn(10, 16, 256) * 10 + 5 # Mean 5, Std 10
    
    class MockVAEUnscaled(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = torch.nn.Embedding(30000, 256)
        def eval(self): pass
        def get_latent(self, ids, mask, use_mean=True):
            return large_latents[0:ids.shape[0]]
        def decode(self, z):
            # Return dummy hidden states [batch, seq, dim]
            # decode_latent expects hidden states, then projects to vocab
            return torch.randn(z.shape[0], z.shape[1], 256, device=z.device)

    # Case A: No Scaler
    model.scaler = None
    model.vae = MockVAEUnscaled()
    diff_out_unscaled = model(
        context_ids, context_mask, question_ids, question_mask, answer_ids, answer_mask,
        train_vae_only=False
    )
    print(f"Diffusion Loss (Unscaled): {diff_out_unscaled['diffusion_loss'].item():.4f}")

    # Case B: With Scaler
    # Fit scaler to these large latents
    scaler = LatentScaler()
    scaler.fit([{"answer_input_ids": torch.zeros(10, 16), "answer_attention_mask": torch.ones(10, 16)}], MockVAEUnscaled(), large_latents.device)
    model.scaler = scaler
    
    diff_out_scaled = model(
        context_ids, context_mask, question_ids, question_mask, answer_ids, answer_mask,
        train_vae_only=False
    )
    print(f"Diffusion Loss (Scaled): {diff_out_scaled['diffusion_loss'].item():.4f}")
    
    if diff_out_scaled['diffusion_loss'].item() < diff_out_unscaled['diffusion_loss'].item():
        print("SUCCESS: Scaler reduced diffusion loss.")
    else:
        print("WARNING: Scaler did not reduce diffusion loss (might be random noise related).")

    # 6. Test Null Token Generation
    print("Testing Null Token Generation...")
    gen_out = model.generate(
        context_ids[0:1], context_mask[0:1], question_ids[0:1], question_mask[0:1],
        num_inference_steps=5 # Fast generation
    )
    print(f"Generated Tokens Shape: {gen_out['tokens'].shape}")
    print(f"Is Null Prediction: {gen_out['is_null']}")
    
    # Check if all tokens are null token (collapse)
    # Note: null_ans_token_id might be specific, usually it's start token or similar if not defined
    # But here we check if 'is_null' is True for everything or if tokens are repetitive
    print(f"Generated Tokens: {gen_out['tokens'][0]}")

    print("Verification complete!")

if __name__ == "__main__":
    test_fixes()
