import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from models.latent_diffusion import LatentDiffusionQA
from models.scaler import LatentScaler
import config
import numpy as np

def load_model():
    print("Initializing model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = config.get_config()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_encoder)
    
    model = LatentDiffusionQA(
        tokenizer=tokenizer,
        base_encoder=cfg.model.base_encoder,
        latent_dim=cfg.model.latent_dim,
        max_answer_len=cfg.model.max_answer_length,
        use_vae=True
    ).to(device)
    
    # Initialize scaler
    model.scaler = LatentScaler()
    model.scaler.to(device)
    
    return model, tokenizer, device

def check_vae_quality(model, tokenizer, device):
    print("\n=== 1. VAE Checklist (Manifold Quality) ===")
    
    # 1.1 Posterior Collapse Check
    print("Checking Posterior Collapse...")
    # Create dummy input
    text = "The quick brown fox jumps over the lazy dog."
    cfg = config.get_config()
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", max_length=cfg.model.max_answer_length, truncation=True).to(device)
    
    with torch.no_grad():
        outputs = model.vae.loss(inputs.input_ids, inputs.attention_mask)
        kl_loss = outputs["kl_loss"].item()
        
    print(f"KL Loss: {kl_loss:.4f}")
    if kl_loss < 0.001:
        print("FAIL: Posterior Collapse detected! KL Loss is near zero.")
    else:
        print("PASS: KL Loss is non-zero.")

    # 1.2 Latent Local Isometry
    print("\nChecking Latent Local Isometry...")
    s1 = "The quick brown fox jumps over the dog."
    s2 = "The quick brown fox jumps over the cat."
    
    t1 = tokenizer(s1, return_tensors="pt", padding="max_length", max_length=cfg.model.max_answer_length, truncation=True).to(device)
    t2 = tokenizer(s2, return_tensors="pt", padding="max_length", max_length=cfg.model.max_answer_length, truncation=True).to(device)
    
    with torch.no_grad():
        z1 = model.vae.get_latent(t1.input_ids, t1.attention_mask, use_mean=True)
        z2 = model.vae.get_latent(t2.input_ids, t2.attention_mask, use_mean=True)
        
    # Flatten for cosine similarity
    z1_flat = z1.view(1, -1)
    z2_flat = z2.view(1, -1)
    cos_sim = F.cosine_similarity(z1_flat, z2_flat).item()
    
    print(f"Cosine Similarity ('...dog' vs '...cat'): {cos_sim:.4f}")
    if cos_sim > 0.9:
        print("PASS: Latent space is locally isometric.")
    else:
        print("WARNING: Latent space might be discontinuous or too sparse.")

    # 1.3 Decoding Robustness (The Noise Gap)
    print("\nChecking Decoding Robustness...")
    with torch.no_grad():
        # Add noise to z1
        noise = torch.randn_like(z1) * 0.1
        z_noisy = z1 + noise
        
        # Decode
        decoded_logits = model.vae.decode(z_noisy)
        decoded_ids = torch.argmax(decoded_logits, dim=-1)
        decoded_text = tokenizer.decode(decoded_ids[0], skip_special_tokens=True)
        
    print(f"Original: {s1}")
    print(f"Noisy Decoded (sigma=0.1): {decoded_text}")
    
    # Simple check: is "fox" still in the sentence?
    if "fox" in decoded_text:
        print("PASS: Decoder handles noisy latents reasonably well.")
    else:
        print("WARNING: Decoder is brittle to noise.")

def check_scaler_quality(model, tokenizer, device):
    print("\n=== 2. Bridge Checklist (Normalization & Scaling) ===")
    
    # 2.1 Unit Variance Assumption
    print("Checking Unit Variance Assumption...")
    # Create a batch of random sentences
    sentences = [
        "The quick brown fox jumps over the dog.",
        "A wizard's job is to vex chumps quickly in fog.",
        "Pack my box with five dozen liquor jugs.",
        "How vexingly quick daft zebras jump!",
        "Sphinx of black quartz, judge my vow."
    ] * 10 # 50 sentences
    
    cfg = config.get_config()
    inputs = tokenizer(sentences, return_tensors="pt", padding="max_length", max_length=cfg.model.max_answer_length, truncation=True).to(device)
    
    # Fit scaler
    print("Fitting scaler on batch...")
    # Mock dataloader structure
    batch_list = [{"answer_input_ids": inputs.input_ids, "answer_attention_mask": inputs.attention_mask}]
    model.scaler.fit(batch_list, model.vae, device)
    
    # Transform
    with torch.no_grad():
        z = model.vae.get_latent(inputs.input_ids, inputs.attention_mask, use_mean=True)
        z_norm = model.scaler.transform(z, mask=inputs.attention_mask)
        
    # Check stats (ignoring padding for stats if possible, but scaler handles it)
    # Scaler transform applies mask, so padded values become 0.
    # We should check non-zero values or just global stats if mask is applied.
    # The scaler implementation multiplies by mask, so padding is 0.
    # We should filter out padding for accurate stats check.
    mask_bool = inputs.attention_mask.bool().unsqueeze(-1)
    z_valid = z_norm[mask_bool.expand_as(z_norm)].view(-1)
    
    mean = z_valid.mean().item()
    std = z_valid.std().item()
    
    print(f"Scaler Output Mean: {mean:.4f} (Target: 0.0)")
    print(f"Scaler Output Std:  {std:.4f} (Target: 1.0)")
    
    if abs(mean) < 0.1 and abs(std - 1.0) < 0.2:
        print("PASS: Scaler produces Unit Variance.")
    else:
        print("WARNING: Scaler output is not N(0, 1).")

    # 2.2 Outlier Distribution
    print("\nChecking Outlier Distribution...")
    max_val = z_valid.abs().max().item()
    print(f"Max Absolute Value in Latents: {max_val:.4f}")
    
    if max_val > 5.0:
        print("WARNING: Heavy tails detected! Max value > 5.0.")
    else:
        print("PASS: Latents are within reasonable bounds ([-5, 5]).")

def check_diffusion_quality(model, tokenizer, device):
    print("\n=== 3. Diffusion Checklist (Scheduling & Denoising) ===")
    
    # 3.1 SNR Schedule
    print("Checking SNR Schedule...")
    alphas_cumprod = model.sampler.scheduler.alphas_cumprod.cpu().numpy()
    snr = alphas_cumprod / (1 - alphas_cumprod)
    
    t_steps = [0, len(snr)//2, len(snr)-1]
    print(f"SNR at t=0:   {snr[t_steps[0]]:.4f}")
    print(f"SNR at t=T/2: {snr[t_steps[1]]:.4f}")
    print(f"SNR at t=T:   {snr[t_steps[2]]:.4f}")
    
    if snr[0] > snr[-1]:
        print("PASS: SNR is monotonically decreasing (roughly).")
    else:
        print("FAIL: SNR is not decreasing!")

    # 3.2 AdaLN Gate Initialization
    print("\nChecking AdaLN Gate Initialization...")
    # Access denoiser -> transformer blocks -> adaLN
    # Structure depends on implementation. Assuming DiT-like structure.
    # Let's check the first block's adaLN if accessible.
    try:
        # This path might vary based on your Denoiser implementation
        # Checking for 'adaLN_modulation' or similar in the first block
        first_block = model.denoiser.transformer.blocks[0]
        if hasattr(first_block, 'adaLN_modulation'):
            # Usually the last layer of adaLN modulation projects to shift/scale/gate
            # We want to check if it was zero-initialized.
            weight = first_block.adaLN_modulation[-1].weight
            bias = first_block.adaLN_modulation[-1].bias
            
            w_norm = weight.norm().item()
            b_norm = bias.norm().item()
            
            print(f"AdaLN Weight Norm: {w_norm:.6f}")
            print(f"AdaLN Bias Norm:   {b_norm:.6f}")
            
            if w_norm < 0.1 and b_norm < 0.1:
                 print("PASS: AdaLN appears to be zero-initialized (or close).")
            else:
                 print("WARNING: AdaLN weights are large. Context might be ignored initially.")
        else:
            print("SKIP: Could not locate AdaLN module in standard path.")
    except Exception as e:
        print(f"SKIP: Could not inspect AdaLN ({e})")

def check_squad_quality(model, tokenizer, device):
    print("\n=== 4. SQuAD 2.0 Checklist (The Task) ===")
    
    # 4.1 Sink Token Separation
    print("Checking Sink Token Separation...")
    
    # Get null token latent
    # Assuming null_ans_token_id is used. If not set, usually 0 or start token.
    null_id = model.null_ans_token_id if hasattr(model, 'null_ans_token_id') else tokenizer.pad_token_id
    print(f"Null Token ID: {null_id}")
    
    null_input = torch.tensor([[null_id]], device=device)
    # We need a sequence for VAE. Let's repeat it or pad it.
    # VAE expects [batch, seq_len].
    cfg = config.get_config()
    null_seq = torch.full((1, cfg.model.max_answer_length), tokenizer.pad_token_id, device=device)
    null_seq[0, 0] = null_id # Set first token to null identifier
    null_mask = torch.zeros((1, cfg.model.max_answer_length), device=device)
    null_mask[0, 0] = 1
    
    with torch.no_grad():
        z_null = model.vae.get_latent(null_seq, null_mask, use_mean=True)
        # z_null is [1, seq, dim]. We care about the first token's latent or the pooled one?
        # VAE outputs sequence latents.
        z_null_vec = z_null[0, 0] # First token latent
        
    # Get average answer latent
    text = "The quick brown fox."
    t = tokenizer(text, return_tensors="pt", padding="max_length", max_length=cfg.model.max_answer_length, truncation=True).to(device)
    with torch.no_grad():
        z_ans = model.vae.get_latent(t.input_ids, t.attention_mask, use_mean=True)
        z_ans_vec = z_ans[0, 0] # First token latent
        
    # Measure distance
    dist = torch.norm(z_null_vec - z_ans_vec).item()
    cos = F.cosine_similarity(z_null_vec.unsqueeze(0), z_ans_vec.unsqueeze(0)).item()
    
    print(f"Euclidean Distance (Null vs Answer): {dist:.4f}")
    print(f"Cosine Similarity (Null vs Answer):  {cos:.4f}")
    
    if dist > 1.0:
        print("PASS: Null token is distinct from answer token.")
    else:
        print("WARNING: Null token is very close to answer token (Risk of Mode Collapse).")

def main():
    model, tokenizer, device = load_model()
    check_vae_quality(model, tokenizer, device)
    check_scaler_quality(model, tokenizer, device)
    check_diffusion_quality(model, tokenizer, device)
    check_squad_quality(model, tokenizer, device)

if __name__ == "__main__":
    main()
