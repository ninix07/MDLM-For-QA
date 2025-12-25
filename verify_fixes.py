import torch
from transformers import XLMRobertaTokenizer
from models.latent_diffusion import LatentDiffusionQA
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
    
    print("Testing VAE training step with KL weight...")
    vae_out = model(
        context_ids, context_mask, question_ids, question_mask, answer_ids, answer_mask,
        train_vae_only=True, kl_weight=0.5
    )
    print(f"VAE Loss: {vae_out['loss'].item()}")
    assert "loss" in vae_out
    
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

    print("Verification complete!")

if __name__ == "__main__":
    test_fixes()
