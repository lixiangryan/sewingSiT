import torch
from diffusers import AutoencoderKL
from PIL import Image
import numpy as np

def test_vae():
    model_id = "segmind/SSD-1B"
    # Local cache path from your previous run.py
    cache_dir = "/mnt/f/nccu/project/diffusion/models" 
    
    print(f"Loading VAE from {model_id} (cache: {cache_dir})...")
    try:
        vae = AutoencoderKL.from_pretrained(
            model_id, 
            subfolder="vae", 
            cache_dir=cache_dir,
            torch_dtype=torch.float16 # Use fp16 to save memory if on GPU, else float32
        )
    except OSError:
        print("Could not load from local cache or download. Please ensure internet access or correct path.")
        # Fallback to standard SDXL VAE if SSD-1B specific one fails or is same
        vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae", cache_dir=cache_dir)

    print("VAE Loaded successfully.")
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae.to(device)
    if device == "cpu":
        vae.to(dtype=torch.float32) # CPU doesn't support fp16 well
    
    # Create a dummy image (1024x1024)
    img_size = 1024
    dummy_img = torch.randn(1, 3, img_size, img_size).to(device, dtype=vae.dtype)
    
    print(f"Encoding image of shape {dummy_img.shape}...")
    with torch.no_grad():
        # Encode
        # VAE output is a distribution, we sample from it
        latents = vae.encode(dummy_img).latent_dist.sample()
        # Scale factor is important! SDXL uses 0.13025
        latents = latents * vae.config.scaling_factor
        
    print(f"Latents shape: {latents.shape}")
    
    expected_shape = (1, 4, img_size // 8, img_size // 8) # VAE f=8
    assert latents.shape == expected_shape, f"Expected {expected_shape}, got {latents.shape}"
    
    print("Success! VAE integration verified.")
    print(f"Compression ratio: {img_size} -> {latents.shape[2]} (Factor: {img_size/latents.shape[2]})")

if __name__ == "__main__":
    test_vae()
