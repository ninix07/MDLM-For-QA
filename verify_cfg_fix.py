import torch
import torch.nn as nn
from models.transformer_blocks import ConditionalTransformerBlock

def test_cfg_fix():
    print("Testing CFG Fix (Mask=1 for dropped samples)...")
    
    # Setup minimal components
    d_model = 32
    num_heads = 4
    
    # Create a block
    block = ConditionalTransformerBlock(d_model, num_heads, ff_dim=64)
    
    # Inputs
    batch_size = 2
    seq_len = 10
    cond_len = 5
    
    x = torch.randn(batch_size, seq_len, d_model)
    condition = torch.randn(batch_size, cond_len, d_model)
    time_emb = torch.randn(batch_size, d_model)
    
    # Case: Dropped Sample (CFG)
    # The fix sets mask to 1 (Valid) even for dropped samples (which are PAD tokens)
    
    # mask: 1 = valid, 0 = padding
    # With fix: mask is ALL ONES for the dropped sample
    mask_dropped_fixed = torch.ones(cond_len)
    
    condition_mask = torch.stack([mask_dropped_fixed, mask_dropped_fixed]) # [2, 5]
    
    # Convert to key_padding_mask format used in Denoiser
    # ~1 -> 0 (False) -> Keep
    key_padding_mask = ~condition_mask.bool() 
    
    print(f"Key Padding Mask (Fixed): {key_padding_mask[0]}")
    # Should be all False (Keep)
    
    try:
        attn = block.cross_attn
        query = x
        key = condition
        value = condition
        
        print("Running MultiheadAttention with Fixed Mask...")
        attn_out, _ = attn(query, key, value, key_padding_mask=key_padding_mask)
        
        print("Output computed.")
        if torch.isnan(attn_out).any():
            print("FAILURE: NaNs still present!")
        else:
            print("SUCCESS: No NaNs with fixed mask.")
            
    except Exception as e:
        print(f"Crashed: {e}")

if __name__ == "__main__":
    test_cfg_fix()
