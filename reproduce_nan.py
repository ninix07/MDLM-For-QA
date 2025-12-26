import torch
import torch.nn as nn
from models.denoiser import ConditionalDenoiser
from models.transformer_blocks import ConditionalTransformerBlock

def test_cfg_dropout_nan():
    print("Testing CFG Dropout NaN Hypothesis...")
    
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
    time_emb = torch.randn(batch_size, d_model) # Block expects d_model input to its internal MLP
    
    # Case 1: Normal Mask (some 1s, some 0s)
    # PyTorch MultiheadAttention: key_padding_mask=True means IGNORE
    # In our code: condition_mask is 1 for valid, 0 for padding.
    # passed as ~condition_mask.bool() -> 0 (False) for valid, 1 (True) for padding.
    
    # Let's simulate the bug: All zeros in condition_mask (meaning all padding)
    # This means key_padding_mask will be ALL TRUE.
    
    # Batch element 0: Valid
    # Batch element 1: All Padding (CFG Dropout)
    
    # mask: 1 = valid, 0 = padding
    mask_valid = torch.ones(cond_len)
    mask_dropped = torch.zeros(cond_len)
    
    condition_mask = torch.stack([mask_valid, mask_dropped]) # [2, 5]
    
    # Convert to key_padding_mask format used in Denoiser
    key_padding_mask = ~condition_mask.bool() # [2, 5]
    # Row 0: False, False... (Keep)
    # Row 1: True, True... (Ignore All)
    
    print(f"Key Padding Mask (Row 1 - Dropped): {key_padding_mask[1]}")
    
    try:
        # Run Cross Attention
        # We need to manually run the block logic roughly
        # The block does: h, _ = self.cross_attn(h, condition, condition, key_padding_mask=condition_mask)
        
        # We need to initialize the time_mlp in the block to avoid error
        # But we can just call cross_attn directly to isolate
        
        attn = block.cross_attn
        
        # query, key, value
        query = x
        key = condition
        value = condition
        
        print("Running MultiheadAttention...")
        attn_out, _ = attn(query, key, value, key_padding_mask=key_padding_mask)
        
        print("Output computed.")
        print(f"Output Row 0 NaN: {torch.isnan(attn_out[0]).any().item()}")
        print(f"Output Row 1 NaN: {torch.isnan(attn_out[1]).any().item()}")
        
        if torch.isnan(attn_out[1]).any():
            print("SUCCESS: Reproduced NaN with all-zero mask!")
        else:
            print("FAILURE: Did not produce NaN. PyTorch might handle this?")
            
    except Exception as e:
        print(f"Crashed: {e}")

if __name__ == "__main__":
    test_cfg_dropout_nan()
