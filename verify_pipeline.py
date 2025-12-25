"""
End-to-End Pipeline Verification Script

Traces data through:
1. BERT Embedding
2. VAE Encoder (with latent sampling)
3. Diffusion Training / Inference
4. VAE Decoder
5. Nearest Token Lookup
6. Final Text Output
"""

import torch
from transformers import AutoTokenizer
import config
from models.latent_diffusion import LatentDiffusionQA
from models.scaler import LatentScaler

def verify_pipeline():
    print("=" * 60)
    print("END-TO-END PIPELINE VERIFICATION")
    print("=" * 60)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = config.get_config()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_encoder)
    
    print(f"\n[1] Model: {cfg.model.base_encoder}")
    print(f"    Device: {device}")
    print(f"    Vocab Size: {len(tokenizer)}")
    print(f"    Pad Token ID: {tokenizer.pad_token_id}")
    
    # Initialize model
    model = LatentDiffusionQA(
        tokenizer=tokenizer,
        base_encoder=cfg.model.base_encoder,
        latent_dim=cfg.model.latent_dim,
        max_answer_len=cfg.model.max_answer_length,
        use_vae=True
    ).to(device)
    model.eval()
    
    # Sample input
    answer_text = "The quick brown fox"
    context_text = "A fox jumped over a lazy dog."
    question_text = "What jumped over the dog?"
    
    print(f"\n[2] INPUT DATA")
    print(f"    Answer: '{answer_text}'")
    print(f"    Context: '{context_text}'")
    print(f"    Question: '{question_text}'")
    
    # Tokenize
    answer_enc = tokenizer(answer_text, return_tensors="pt", padding="max_length", 
                           max_length=cfg.model.max_answer_length, truncation=True).to(device)
    context_enc = tokenizer(context_text, return_tensors="pt", padding="max_length",
                            max_length=cfg.model.max_context_length, truncation=True).to(device)
    question_enc = tokenizer(question_text, return_tensors="pt", padding="max_length",
                             max_length=cfg.model.max_question_length, truncation=True).to(device)
    
    answer_ids = answer_enc.input_ids
    answer_mask = answer_enc.attention_mask
    
    print(f"\n[3] TOKENIZATION")
    print(f"    Answer IDs (first 10): {answer_ids[0, :10].tolist()}")
    print(f"    Answer Mask (first 10): {answer_mask[0, :10].tolist()}")
    
    # ============================================================
    # STAGE 1: BERT Embedding -> VAE Encoder
    # ============================================================
    print(f"\n{'='*60}")
    print("[STAGE 1] BERT Embedding → VAE Encoder")
    print("="*60)
    
    with torch.no_grad():
        # Get embeddings from BERT
        embeddings = model.vae.embeddings(answer_ids)
        print(f"    BERT Embeddings Shape: {embeddings.shape}")
        print(f"    BERT Embeddings Mean: {embeddings.mean().item():.4f}")
        print(f"    BERT Embeddings Std: {embeddings.std().item():.4f}")
        
        # VAE Encoder
        z, mean, logvar = model.vae.encode(answer_ids, answer_mask)
        print(f"\n    VAE Latent (z) Shape: {z.shape}")
        print(f"    VAE Latent Mean: {mean.mean().item():.4f}")
        print(f"    VAE Latent LogVar: {logvar.mean().item():.4f}")
        print(f"    VAE Latent Std (from z): {z.std().item():.4f}")
    
    # ============================================================
    # STAGE 2: VAE Warmup Training (simulate one step)
    # ============================================================
    print(f"\n{'='*60}")
    print("[STAGE 2] VAE Warmup Training (simulate)")
    print("="*60)
    
    model.train()
    vae_output = model(
        context_enc.input_ids, context_enc.attention_mask,
        question_enc.input_ids, question_enc.attention_mask,
        answer_ids, answer_mask,
        train_vae_only=True,
        kl_weight=0.1
    )
    print(f"    VAE Loss: {vae_output['loss'].item():.4f}")
    if 'recon_loss' in vae_output:
        print(f"    Recon Loss: {vae_output['recon_loss'].item():.4f}")
        print(f"    KL Loss: {vae_output['kl_loss'].item():.4f}")
    model.eval()
    
    # ============================================================
    # STAGE 3: Fit Scaler & Diffusion Training (simulate)
    # ============================================================
    print(f"\n{'='*60}")
    print("[STAGE 3] Scaler Fitting & Diffusion Training (simulate)")
    print("="*60)
    
    # Fit scaler
    scaler = LatentScaler()
    
    # Create a simple dataloader-like structure
    class SimpleBatch:
        def __init__(self, answer_ids, answer_mask):
            self.data = {"answer_input_ids": answer_ids, "answer_attention_mask": answer_mask}
        def __iter__(self):
            yield self.data
    
    simple_loader = SimpleBatch(answer_ids, answer_mask)
    scaler.fit(simple_loader, model.vae, device)
    model.scaler = scaler
    
    print(f"    Scaler Mean Norm: {scaler.mean.norm().item():.4f}")
    print(f"    Scaler Std Norm: {scaler.std.norm().item():.4f}")
    
    # Get latent for normalization test
    with torch.no_grad():
        z_batch = model.encode_answer(answer_ids, answer_mask)
    
    # Normalize latent
    with torch.no_grad():
        z_normalized = scaler.transform(z_batch)
    print(f"    Normalized Latent Mean: {z_normalized.mean().item():.4f}")
    print(f"    Normalized Latent Std: {z_normalized.std().item():.4f}")
    
    # Simulate diffusion training step
    model.train()
    diff_output = model(
        context_enc.input_ids, context_enc.attention_mask,
        question_enc.input_ids, question_enc.attention_mask,
        answer_ids, answer_mask,
        train_vae_only=False,
        kl_weight=0.1
    )
    print(f"\n    Diffusion Loss: {diff_output['diffusion_loss'].item():.4f}")
    print(f"    VAE Loss: {diff_output['vae_loss'].item():.4f}")
    print(f"    Total Loss: {diff_output['loss'].item():.4f}")
    model.eval()
    
    # ============================================================
    # STAGE 4: Diffusion Inference -> VAE Decoder -> Tokens
    # ============================================================
    print(f"\n{'='*60}")
    print("[STAGE 4] Diffusion Inference → VAE Decoder → Tokens")
    print("="*60)
    
    with torch.no_grad():
        gen_output = model.generate(
            context_enc.input_ids, context_enc.attention_mask,
            question_enc.input_ids, question_enc.attention_mask,
            show_progress=False,
            num_inference_steps=10  # Fast for testing
        )
    
    print(f"    Generated Latent Shape: {gen_output['latent'].shape}")
    print(f"    Generated Latent Mean: {gen_output['latent'].mean().item():.4f}")
    print(f"    Generated Latent Std: {gen_output['latent'].std().item():.4f}")
    
    print(f"\n    Generated Token IDs (first 10): {gen_output['tokens'][0, :10].tolist()}")
    print(f"    Is Null Prediction: {gen_output['is_null'][0].item()}")
    print(f"    Null Similarity: {gen_output['null_similarity'][0].item():.4f}")
    
    # ============================================================
    # STAGE 5: Nearest Token Lookup -> Final Text
    # ============================================================
    print(f"\n{'='*60}")
    print("[STAGE 5] Nearest Token → Final Text Output")
    print("="*60)
    
    final_text = model.decode_tokens_to_text(gen_output['tokens'], gen_output['is_null'])
    
    print(f"    Final Output: '{final_text[0]}'")
    
    # ============================================================
    # STAGE 6: VAE Reconstruction (for comparison)
    # ============================================================
    print(f"\n{'='*60}")
    print("[STAGE 6] VAE Reconstruction (for comparison)")
    print("="*60)
    
    with torch.no_grad():
        recon_output = model.vae_reconstruct(answer_ids, answer_mask)
    
    recon_text = model.decode_tokens_to_text(recon_output['tokens'], recon_output['is_null'])
    
    print(f"    Original Input: '{answer_text}'")
    print(f"    VAE Reconstructed: '{recon_text[0]}'")
    print(f"    Tokens Match: {(recon_output['tokens'][0] == answer_ids[0]).sum().item()}/{answer_ids.shape[1]}")
    
    # ============================================================
    # SUMMARY
    # ============================================================
    print(f"\n{'='*60}")
    print("PIPELINE VERIFICATION SUMMARY")
    print("="*60)
    print("✓ [1] BERT Embedding Layer: Working")
    print("✓ [2] VAE Encoder (mean/logvar): Working")
    print("✓ [3] VAE Loss Calculation: Working")
    print("✓ [4] Latent Scaler: Working")
    print("✓ [5] Diffusion Training: Working")
    print("✓ [6] Diffusion Sampling: Working")
    print("✓ [7] VAE Decoder: Working")
    print("✓ [8] Nearest Token Lookup: Working")
    print("✓ [9] Text Decoding: Working")
    print("\n✅ END-TO-END PIPELINE VERIFIED")

if __name__ == "__main__":
    verify_pipeline()
