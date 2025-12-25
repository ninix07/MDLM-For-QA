import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import config
from data import create_dataloader
from models.latent_diffusion import LatentDiffusionQA
from metrics import compute_metrics

def debug_decoding():
    print("Initializing...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = config.get_config()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_encoder)
    
    # Load model
    model = LatentDiffusionQA(
        tokenizer=tokenizer,
        base_encoder=cfg.model.base_encoder,
        latent_dim=cfg.model.latent_dim,
        max_answer_len=cfg.model.max_answer_length,
        use_vae=True
    ).to(device)
    
    # Load validation loader directly
    print("Loading validation data...")
    val_loader, _ = create_dataloader(
        data_path=cfg.dev_file,
        tokenizer=tokenizer,
        batch_size=4,
        max_context_length=cfg.model.max_context_length,
        max_question_length=cfg.model.max_question_length,
        max_answer_length=cfg.model.max_answer_length,
        use_balanced_sampler=False,
        shuffle=False
    )
    
    batch = next(iter(val_loader))
    answer_ids = batch["answer_input_ids"].to(device)
    answer_mask = batch["answer_attention_mask"].to(device)
    
    print(f"\nNull Token ID in Model: {model.null_ans_token_id}")
    print(f"Null Token in Tokenizer: {tokenizer.convert_tokens_to_ids(cfg.model.null_ans_token)}")
    
    # Check first few examples
    for i in range(4):
        ids = answer_ids[i]
        print(f"\nExample {i}:")
        print(f"IDs: {ids[:10].tolist()}...")
        print(f"Is Null (by ID): {ids[0] == model.null_ans_token_id}")
        
        # Simulate "Perfect" Prediction (Identity)
        tokens = ids.unsqueeze(0)
        is_null = (tokens[:, 0] == model.null_ans_token_id)
        
        # Decode
        text = model.decode_tokens_to_text(tokens, is_null)[0]
        print(f"Decoded Text: '{text}'")
        
        # Check Reference Logic
        is_null_ref = (ids[0] == model.null_ans_token_id).unsqueeze(0)
        ref_text = model.decode_tokens_to_text(tokens, is_null_ref)[0]
        print(f"Reference Text: '{ref_text}'")
        
        # Compute Metric
        metrics = compute_metrics([text], [ref_text])
        print(f"Metrics: {metrics}")

if __name__ == "__main__":
    debug_decoding()
