import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import config
from models.latent_diffusion import LatentDiffusionQA

def reproduce_issue():
    print("Initializing...")
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
    
    # Create a dummy batch
    text = ["France", "Germany"]
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", max_length=cfg.model.max_answer_length, truncation=True).to(device)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    
    print(f"Input IDs: {input_ids[0, :10].tolist()}")
    
    # 1. Test Loss with Real Forward Pass (Random Weights)
    print("\n--- Real Forward Pass (Random Weights) ---")
    with torch.no_grad():
        outputs = model.vae.loss(input_ids, attention_mask)
    print(f"Loss: {outputs['loss'].item():.4f}")
    print(f"Recon: {outputs['recon_loss'].item():.4f}")
    
    # 2. Test Loss with "Perfect" Logits (Mocking)
    print("\n--- Mock Perfect Logits ---")
    # Create logits that perfectly predict input_ids
    # Shape: [batch, seq, vocab]
    vocab_size = model.vae.vocab_size
    batch_size, seq_len = input_ids.shape
    
    # Initialize with large negative value
    perfect_logits = torch.full((batch_size, seq_len, vocab_size), -100.0, device=device)
    
    # Set correct token index to high positive value
    for b in range(batch_size):
        for s in range(seq_len):
            token_id = input_ids[b, s]
            perfect_logits[b, s, token_id] = 100.0
            
    # Compute loss manually
    recon_loss = F.cross_entropy(
        perfect_logits.view(-1, vocab_size),
        input_ids.view(-1),
        reduction="mean",
        ignore_index=tokenizer.pad_token_id
    )
    print(f"Perfect Recon Loss: {recon_loss.item():.4f}")
    
    # 3. Test Loss with "All Padding" Prediction
    print("\n--- Mock All-Padding Logits ---")
    pad_logits = torch.full((batch_size, seq_len, vocab_size), -100.0, device=device)
    pad_id = tokenizer.pad_token_id
    pad_logits[:, :, pad_id] = 100.0
    
    recon_loss_pad = F.cross_entropy(
        pad_logits.view(-1, vocab_size),
        input_ids.view(-1),
        reduction="mean",
        ignore_index=tokenizer.pad_token_id
    )
    print(f"All-Pad Recon Loss: {recon_loss_pad.item():.4f}")

if __name__ == "__main__":
    reproduce_issue()
