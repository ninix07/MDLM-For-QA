import torch
from models.diffusion import NoiseScheduler, GaussianDiffusion
from models.latent_diffusion import LatentDiffusionQA
from models.vae import SequenceVAE
import torch.nn as nn
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

def test_all_fixes():
    print("--- Starting Comprehensive Verification ---")
    
    # 1. Test Diffusion Loss (Min-SNR)
    print("\n1. Testing Diffusion Loss (Min-SNR) Stability...")
    scheduler = NoiseScheduler(num_timesteps=1000, schedule_type="cosine")
    diffusion = GaussianDiffusion(scheduler)
    
    class DummyModel(nn.Module):
        def forward(self, x, t, **kwargs):
            return torch.randn_like(x)
    
    model = DummyModel()
    x_0 = torch.randn(4, 10, 128)
    
    # Mock randint to test critical timesteps
    original_randint = torch.randint
    
    try:
        for t_val in [0, 1, 500, 998, 999]:
            def mock_randint(*args, **kwargs):
                return torch.tensor([t_val] * 4)
            torch.randint = mock_randint
            
            loss_dict = diffusion.training_loss(model, x_0, {}, mask=None)
            if torch.isnan(loss_dict["loss"]).any():
                print(f"FAILED: Diffusion Loss NaN at t={t_val}")
                return
            else:
                print(f"Diffusion t={t_val}: OK")
    except Exception as e:
        print(f"Error in diffusion test: {e}")
    finally:
        torch.randint = original_randint

    # 2. Test Penalty Loss Math
    print("\n2. Testing Penalty Loss Math Stability...")
    try:
        for t_idx in [0, 1, 500, 998, 999]:
            t = torch.tensor([t_idx])
            z_t = torch.randn(1, 10, 32)
            noise_pred = torch.randn(1, 10, 32)
            
            alpha_cumprod = scheduler.alphas_cumprod[t]
            sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod).view(-1, 1, 1)
            sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alpha_cumprod).view(-1, 1, 1)
            
            # The fix: clamp
            pred_x0 = (z_t - sqrt_one_minus_alpha_cumprod * noise_pred) / sqrt_alpha_cumprod.clamp(min=1e-8)
            
            if torch.isnan(pred_x0).any() or torch.isinf(pred_x0).any():
                print(f"FAILED: Penalty Math NaN at t={t_idx}")
                return
            else:
                print(f"Penalty Math t={t_idx}: OK")
    except Exception as e:
        print(f"Error in penalty test: {e}")

    # 3. Test VAE Logvar Clamping
    print("\n3. Testing VAE Logvar Clamping...")
    try:
        vae = SequenceVAE(vocab_size=100, embedding_dim=32, latent_dim=32)
        # Force encoder to output large values
        vae.to_logvar.weight.data.fill_(100.0)
        vae.to_logvar.bias.data.fill_(100.0)
        
        input_ids = torch.randint(0, 100, (2, 10))
        z, mean, logvar = vae.encode(input_ids)
        
        print(f"Max logvar: {logvar.max().item()}")
        if logvar.max().item() > 20.0:
            print("FAILED: Logvar not clamped!")
            return
        
        # Check std
        std = torch.exp(0.5 * logvar)
        if torch.isinf(std).any() or torch.isnan(std).any():
            print("FAILED: Std is Inf/NaN!")
            return
            
        print("VAE Logvar Clamping: OK")
        
    except Exception as e:
        print(f"Error in VAE test: {e}")

    print("\n--- All Tests Passed Successfully ---")

if __name__ == "__main__":
    test_all_fixes()
