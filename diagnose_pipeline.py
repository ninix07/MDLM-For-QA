"""
Diagnostic script to test VAE and Diffusion pipeline.
Checks if: VAE encode -> Diffusion add/remove noise -> VAE decode works.
"""
import torch
from config import get_config
from models import LatentDiffusionQA
from models.scaler import LatentScaler
from transformers import AutoTokenizer

def main():
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(config.model.base_encoder)
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
        use_vae=True,
        base_encoder=config.model.base_encoder,
        false_negative_penalty_weight=config.training.false_negative_penalty_weight,
        scaler=latent_scaler,
        prediction_type=config.diffusion.prediction_type,
    ).to(device)
    
    # Try to load checkpoint
    import os
    vae_path = os.path.join(config.training.output_dir, "vae_warmup_best.pt")
    if os.path.exists(vae_path):
        print(f"Loading VAE checkpoint: {vae_path}")
        model.load_state_dict(torch.load(vae_path, map_location=device), strict=False)
    else:
        print(f"No checkpoint found at {vae_path}, using random weights")
    
    model.eval()
    
    # Test 1: VAE Reconstruction
    print("\n=== TEST 1: VAE Reconstruction ===")
    test_answer = "the anglo - ottoman treaty"
    encoded = tokenizer(
        test_answer, 
        max_length=config.model.max_answer_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).to(device)
    
    answer_ids = encoded["input_ids"]
    answer_mask = encoded["attention_mask"]
    
    with torch.no_grad():
        # Encode answer to latent
        z_raw = model.encode_answer(answer_ids, answer_mask)
        print(f"Raw VAE latent: mean={z_raw.mean().item():.4f}, std={z_raw.std().item():.4f}")
        
        # Decode back
        tokens_recon = model.decode_latent(z_raw, return_tokens=True)
        text_recon = tokenizer.decode(tokens_recon[0], skip_special_tokens=True)
        print(f"Original: '{test_answer}'")
        print(f"Reconstructed: '{text_recon}'")
        
    # Test 2: Scaler Transform/Inverse
    print("\n=== TEST 2: Scaler Transform/Inverse ===")
    with torch.no_grad():
        # First fit the scaler on a sample batch
        latent_scaler.fit(z_raw)
        model.scaler = latent_scaler
        
        z_norm = latent_scaler.transform(z_raw)
        print(f"Normalized latent: mean={z_norm.mean().item():.4f}, std={z_norm.std().item():.4f}")
        
        z_denorm = latent_scaler.inverse_transform(z_norm)
        print(f"Denormalized latent: mean={z_denorm.mean().item():.4f}, std={z_denorm.std().item():.4f}")
        
        # Check if denorm matches raw
        diff = (z_raw - z_denorm).abs().max().item()
        print(f"Max difference (raw vs denorm): {diff:.6f}")
        
        # Decode the denormalized latent
        tokens_denorm = model.decode_latent(z_denorm, return_tokens=True)
        text_denorm = tokenizer.decode(tokens_denorm[0], skip_special_tokens=True)
        print(f"After norm/denorm: '{text_denorm}'")
        
    # Test 3: Add noise and remove it (single step)
    print("\n=== TEST 3: Diffusion Round-Trip ===")
    with torch.no_grad():
        t = torch.tensor([100], device=device)  # Medium timestep
        
        # Add noise
        z_t, noise = model.diffusion.q_sample(z_norm, t)
        print(f"Noisy latent (t=100): mean={z_t.mean().item():.4f}, std={z_t.std().item():.4f}")
        
        # If we knew the EXACT noise, we could recover z_0
        z_recovered = model.diffusion.predict_x0_from_noise(z_t, t, noise)
        print(f"Recovered z_0: mean={z_recovered.mean().item():.4f}, std={z_recovered.std().item():.4f}")
        
        # Check recovery error
        recovery_error = (z_norm - z_recovered).abs().max().item()
        print(f"Recovery error (should be ~0): {recovery_error:.6f}")
        
        # Decode recovered
        z_recov_denorm = latent_scaler.inverse_transform(z_recovered)
        tokens_recov = model.decode_latent(z_recov_denorm, return_tokens=True)
        text_recov = tokenizer.decode(tokens_recov[0], skip_special_tokens=True)
        print(f"After diffusion round-trip: '{text_recov}'")

if __name__ == "__main__":
    main()
