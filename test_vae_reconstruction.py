#!/usr/bin/env python3
"""
Test VAE reconstruction quality after fixes using actual dataset samples.
This script tests the critical fixes:
1. Clipping range [-5,5] matching scaler
2. Removed causal mask from decoder
3. Added output normalization
4. Lowered KL weight to prevent posterior collapse

Features:
- CSV output for detailed analysis
- Batched processing for efficiency
- Progress tracking
"""

import torch
import torch.nn.functional as F
import csv
import os
from datetime import datetime
from transformers import AutoTokenizer
from models.vae import SequenceVAE
from metrics import compute_metrics
from data.dataset import SQuAD2Dataset
from config import ModelConfig as modelconfig
from tqdm import tqdm

def test_vae_reconstruction():
    """Test VAE reconstruction on actual dataset samples with batching and CSV output."""
    print("=== Testing VAE Reconstruction Quality on Dataset ===")
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(modelconfig.base_encoder)
    
    # Add null token
    null_token = "<NULL_ANS>"
    if null_token not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [null_token]})
    
    # Create VAE
    vae = SequenceVAE(
        vocab_size=len(tokenizer),
        embedding_dim=768,
        latent_dim=128,
        num_layers=4,
        num_heads=8,
        dropout=0.1,
        pad_token_id=tokenizer.pad_token_id,
    ).to(device)
    
    # Load trained weights
    checkpoint_path = "checkpoints/vae_warmup_best.pt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Extract VAE weights from checkpoint
        if isinstance(checkpoint, dict):
            if 'vae_state_dict' in checkpoint:
                vae_state_dict = checkpoint['vae_state_dict']
            elif 'model_state_dict' in checkpoint:
                vae_state_dict = {k.replace('vae.', ''): v for k, v in checkpoint['model_state_dict'].items() if k.startswith('vae.')}
            else:
                # Handle checkpoint with 'vae.' prefix
                vae_state_dict = {k.replace('vae.', ''): v for k, v in checkpoint.items() if k.startswith('vae.')}
        else:
            vae_state_dict = {k.replace('vae.', ''): v for k, v in checkpoint.state_dict().items() if k.startswith('vae.')}
        
        # Load only VAE weights
        missing_keys, unexpected_keys = vae.load_state_dict(vae_state_dict, strict=False)
        
        if missing_keys:
            print(f"⚠️  Missing VAE keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"⚠️  Unexpected keys: {len(unexpected_keys)}")
            
        print(f"✅ Loaded trained VAE from {checkpoint_path}")
        print(f"   VAE parameters loaded: {len(vae_state_dict)}")
    else:
        print("⚠️  Warning: No trained VAE found, using random weights")
        print(f"    Expected checkpoint at: {checkpoint_path}")
    
    # Load dataset
    print("Loading dataset...")
    dataset = SQuAD2Dataset(
        data_path="data/dev-v2.0.json",
        tokenizer=tokenizer,
        max_context_length=384,
        max_question_length=64,
        max_answer_length=64,
        null_ans_token=null_token,
    )
    
    # Sample diverse examples from dataset
    num_samples = 100
    batch_size = 16  # Process in batches for efficiency
    print(f"Testing on {num_samples} diverse samples from dataset (batch_size={batch_size})...")
    
    # Get a mix of answerable and unanswerable questions (5% unanswerable, 95% answerable)
    answerable_samples = []
    unanswerable_samples = []
    
    # Calculate target counts: 5% unanswerable, 95% answerable
    num_unanswerable = max(1, int(num_samples * 0.05))  # At least 1 unanswerable
    num_answerable = num_samples - num_unanswerable
    
    # Search more broadly for samples
    max_search = min(5000, len(dataset.examples))  # Search more examples
    
    print(f"Searching for {num_answerable} answerable and {num_unanswerable} unanswerable samples in first {max_search} examples...")
    
    # Debug: Check first few examples to see their types
    print("Debug: Checking first 10 examples:")
    for i in range(min(10, len(dataset.examples))):
        example = dataset.examples[i]
        print(f"  Example {i}: is_impossible={example.is_impossible}, answer='{example.answer[:50]}...'")
    
    for i in range(max_search):
        example = dataset.examples[i]
        
        # Check if we need more samples of each type
        need_answerable = len(answerable_samples) < num_answerable
        need_unanswerable = len(unanswerable_samples) < num_unanswerable
        
        if not need_answerable and not need_unanswerable:
            break
            
        # Add sample if we need it and it matches the type
        if example.is_impossible and need_unanswerable:
            unanswerable_samples.append(example)
        elif not example.is_impossible and need_answerable:
            answerable_samples.append(example)
    
    # If we couldn't find enough unanswerable, adjust the counts
    if len(unanswerable_samples) == 0:
        print("Warning: No unanswerable samples found, using all answerable samples")
        test_samples = answerable_samples[:num_samples]
    else:
        # Combine samples with the desired ratio
        test_samples = answerable_samples[:num_answerable] + unanswerable_samples[:num_unanswerable]
    
    print(f"Found {len(answerable_samples)} answerable, {len(unanswerable_samples)} unanswerable")
    print(f"Using {len(test_samples)} total samples")
    
    print(f"Device: {device}")
    print(f"VAE parameters: {sum(p.numel() for p in vae.parameters()):,}")
    print(f"Dataset size: {len(dataset.examples):,}")
    print(f"Answerable samples: {len(answerable_samples)}")
    print(f"Unanswerable samples: {len(unanswerable_samples)}")
    
    # Prepare CSV output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"vae_reconstruction_results_{timestamp}.csv"
    
    # CSV headers
    csv_headers = [
        'sample_id', 'type', 'question', 'original_answer', 'reconstructed_answer',
        'recon_loss', 'kl_loss', 'total_loss', 'char_overlap', 'f1_score', 'exact_match'
    ]
    
    # Storage for results
    results = []
    all_predictions = []
    all_references = []
    answerable_f1s = []
    unanswerable_f1s = []
    
    # Process in batches
    vae.eval()
    with torch.no_grad():
        for batch_start in tqdm(range(0, len(test_samples), batch_size), desc="Processing batches"):
            batch_end = min(batch_start + batch_size, len(test_samples))
            batch_samples = test_samples[batch_start:batch_end]
            
            # Prepare batch data
            batch_answers = [example.answer for example in batch_samples]
            batch_inputs = tokenizer(
                batch_answers,
                max_length=64,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            
            input_ids = batch_inputs["input_ids"].to(device)
            attention_mask = batch_inputs["attention_mask"].to(device)
            
            # Forward pass for entire batch
            outputs = vae.loss(input_ids, attention_mask, kl_weight=0.01)
            
            # Decode reconstruction for entire batch
            decoded = vae.decode(outputs["z"])
            embed_weight = vae.embeddings.weight
            logits = torch.matmul(decoded, embed_weight.T)
            pred_ids = logits.argmax(dim=-1)
            
            # NEW: Manual Truncation Logic for entire batch
            decoded_texts = []
            for i in range(len(pred_ids)):
                # Convert to list
                token_list = pred_ids[i].tolist()
                
                # TRUNCATE at the first special token (EOS or PAD)
                # Assuming tokenizer.sep_token_id is EOS and tokenizer.pad_token_id is PAD
                if tokenizer.sep_token_id in token_list:
                    sep_idx = token_list.index(tokenizer.sep_token_id)
                    token_list = token_list[:sep_idx]
                elif tokenizer.pad_token_id in token_list:
                    pad_idx = token_list.index(tokenizer.pad_token_id)
                    token_list = token_list[:pad_idx]
                    
                # Now decode
                text = tokenizer.decode(token_list, skip_special_tokens=True).strip()
                decoded_texts.append(text)
            
            # Process each sample in the batch
            for i, example in enumerate(batch_samples):
                sample_idx = batch_start + i
                
                # Get individual results with proper truncation
                pred_text = decoded_texts[i]
                ref_text = tokenizer.decode(input_ids[i], skip_special_tokens=True)
                
                recon_loss = outputs["recon_loss"].item()
                kl_loss = outputs["kl_loss"].item()
                total_loss = outputs["loss"].item()
                
                # Calculate character-level accuracy
                if ref_text.strip():
                    ref_chars = set(ref_text.lower())
                    pred_chars = set(pred_text.lower())
                    char_overlap = len(ref_chars & pred_chars) / max(len(ref_chars), 1)
                else:
                    char_overlap = 0.0
                
                # Calculate F1 and EM
                sample_metrics = compute_metrics([pred_text], [ref_text])
                f1_score = sample_metrics['f1']
                exact_match = sample_metrics['em']
                
                # Store result
                result = {
                    'sample_id': sample_idx + 1,
                    'type': 'UNANSWERABLE' if example.is_impossible else 'ANSWERABLE',
                    'question': example.question,
                    'original_answer': ref_text,
                    'reconstructed_answer': pred_text,
                    'recon_loss': recon_loss,
                    'kl_loss': kl_loss,
                    'total_loss': total_loss,
                    'char_overlap': char_overlap,
                    'f1_score': f1_score,
                    'exact_match': exact_match
                }
                results.append(result)
                
                # Track for overall metrics
                all_predictions.append(pred_text)
                all_references.append(ref_text)
                
                if ref_text.strip():  # Answerable
                    answerable_f1s.append(f1_score)
                else:  # Unanswerable
                    unanswerable_f1s.append(f1_score)
    
    # Write results to CSV
    print(f"\nWriting detailed results to {csv_filename}...")
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
        writer.writeheader()
        writer.writerows(results)
    
    # Calculate overall metrics
    avg_recon_loss = sum(r['recon_loss'] for r in results) / len(results)
    avg_kl_loss = sum(r['kl_loss'] for r in results) / len(results)
    avg_char_overlap = sum(r['char_overlap'] for r in results) / len(results)
    
    print(f"\n=== Overall Results ===")
    print(f"Total samples processed: {len(results)}")
    print(f"Average Reconstruction Loss: {avg_recon_loss:.4f}")
    print(f"Average KL Loss: {avg_kl_loss:.4f}")
    print(f"Average Character Overlap: {avg_char_overlap:.2%}")
    
    # Compute F1 scores
    metrics = compute_metrics(all_predictions, all_references)
    print(f"Overall F1 Score: {metrics['f1']:.2f}")
    print(f"Overall Exact Match: {metrics['em']:.2f}")
    
    # Breakdown by type
    if answerable_f1s:
        avg_answerable_f1 = sum(answerable_f1s) / len(answerable_f1s)
        print(f"Answerable F1: {avg_answerable_f1:.2f} (n={len(answerable_f1s)})")
    
    if unanswerable_f1s:
        avg_unanswerable_f1 = sum(unanswerable_f1s) / len(unanswerable_f1s)
        print(f"Unanswerable F1: {avg_unanswerable_f1:.2f} (n={len(unanswerable_f1s)})")
    
    # Performance analysis
    print(f"\n=== Performance Analysis ===")
    perfect_reconstructions = sum(1 for r in results if r['f1_score'] == 1.0)
    good_reconstructions = sum(1 for r in results if r['f1_score'] >= 0.8)
    fair_reconstructions = sum(1 for r in results if r['f1_score'] >= 0.5)
    
    print(f"Perfect reconstructions (F1=1.0): {perfect_reconstructions}/{len(results)} ({perfect_reconstructions/len(results):.1%})")
    print(f"Good reconstructions (F1≥0.8): {good_reconstructions}/{len(results)} ({good_reconstructions/len(results):.1%})")
    print(f"Fair reconstructions (F1≥0.5): {fair_reconstructions}/{len(results)} ({fair_reconstructions/len(results):.1%})")
    
    # Verdict
    print(f"\n=== Verdict ===")
    if metrics['f1'] >= 0.90:
        print("✅ VAE reconstruction quality is GOOD (F1 >= 90%)")
        print("   Ready for diffusion training!")
    elif metrics['f1'] >= 0.70:
        print("⚠️  VAE reconstruction quality is FAIR (F1 >= 70%)")
        print("   Consider more VAE training or lower KL weight further.")
    else:
        print("❌ VAE reconstruction quality is POOR (F1 < 70%)")
        print("   Fix VAE before attempting diffusion training!")
        print("   Suggestions:")
        print("   - Lower KL weight to 1e-4 or 1e-6")
        print("   - Increase latent dimension")
        print("   - Train VAE longer")
        print("   - Check if VAE is properly trained")
    
    print(f"\n📊 Detailed results saved to: {csv_filename}")
    
    return metrics, results

if __name__ == "__main__":
    metrics, results = test_vae_reconstruction()
