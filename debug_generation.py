import torch
import torch.nn as nn
from models.vae import SequenceVAE
from models.latent_diffusion import LatentDiffusionQA
from transformers import AutoTokenizer

def test_generation_logic():
    print("Testing Generation Logic...")
    
    # 1. Test VAE Decoding with Random Latents
    print("\n1. Testing VAE Decoding (Random Latents)...")
    vocab_size = 100
    latent_dim = 32
    embedding_dim = 32
    
    vae = SequenceVAE(vocab_size=vocab_size, embedding_dim=embedding_dim, latent_dim=latent_dim)
    
    # Create random latents
    batch_size = 2
    seq_len = 10
    z = torch.randn(batch_size, seq_len, latent_dim)
    
    # Decode
    hidden = vae.decode(z)
    print(f"Hidden shape: {hidden.shape}")
    
    # Project to logits (manual simulation of LatentDiffusionQA logic)
    embed_weight = vae.embeddings.weight
    logits = torch.matmul(hidden, embed_weight.T)
    print(f"Logits shape: {logits.shape}")
    
    tokens = logits.argmax(dim=-1)
    print(f"Tokens: {tokens}")
    
    if (tokens == 0).all():
        print("FAILURE: VAE decoded to all zeros!")
    else:
        print("SUCCESS: VAE produced non-zero tokens.")

    # 2. Test LatentDiffusionQA.generate wrapper
    print("\n2. Testing LatentDiffusionQA.generate wrapper...")
    
    # Mock tokenizer
    class MockTokenizer:
        pad_token_id = 0
        sep_token_id = 99
        def __len__(self): return 100
        def get_vocab(self): return {"<NULL_ANS>": 1}
        def add_special_tokens(self, *args): pass
        def convert_tokens_to_ids(self, *args): return 1
        def decode(self, ids, **kwargs): return "decoded_string"
        
    tokenizer = MockTokenizer()
    
    model = LatentDiffusionQA(
        tokenizer=tokenizer,
        latent_dim=latent_dim,
        d_model=32,
        num_layers=1,
        num_heads=1,
        ff_dim=32,
        use_vae=True,
        base_encoder="bert-base-uncased" # This might try to load...
    )
    
    # Inject our dummy VAE
    model.vae = vae
    
    # Mock components to avoid heavy loading
    model.denoiser = nn.Linear(1, 1) # Dummy
    model.denoiser.encode_condition = lambda *args: (torch.randn(2, 5, 32), torch.ones(2, 5))
    model.denoiser.forward = lambda *args, **kwargs: torch.randn(2, 10, 32) # Predict random noise
    
    # Mock sampler
    model.sampler.sample = lambda *args, **kwargs: torch.randn(2, 10, 32) # Return random z_0
    
    # Run generate
    context_ids = torch.randint(0, 100, (2, 5))
    context_mask = torch.ones((2, 5))
    question_ids = torch.randint(0, 100, (2, 5))
    question_mask = torch.ones((2, 5))
    
    try:
        output = model.generate(context_ids, context_mask, question_ids, question_mask)
        print(f"Generated Tokens: {output['tokens']}")
        
        if (output['tokens'] == 0).all():
             print("FAILURE: Generate produced all zeros!")
        else:
             print("SUCCESS: Generate produced non-zero tokens.")
             
    except Exception as e:
        print(f"Error in generate: {e}")

if __name__ == "__main__":
    test_generation_logic()
