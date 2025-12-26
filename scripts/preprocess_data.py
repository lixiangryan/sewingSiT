import torch
import os
import argparse
from glob import glob
from tqdm import tqdm
from PIL import Image
from diffusers import AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
import numpy as np

# Define paths
CACHE_DIR = "/mnt/f/nccu/project/diffusion/models"
MODEL_ID_VAE = "segmind/SSD-1B" # Or sd-vae-ft-mse
MODEL_ID_TEXT = "openai/clip-vit-large-patch14" # Standard CLIP used in SD 1.5/2.1 (SSD-1B actually uses 2 text encoders, let's start with one for simplicity or match SDXL)

# Note on Text Encoder:
# SSD-1B / SDXL uses TWO text encoders (CLIP ViT-L and OpenCLIP ViT-bigG).
# To fully match SSD-1B, we should use both and concat their outputs.
# For this 'sewing' experiment, sticking to one (ViT-L) is easier to start with, 
# but if you want high quality, we should eventually support both.
# Let's start with CLIP ViT-L/14 (hidden_size=768) which matches our default SiT config.

def setup_models(device):
    print("Loading VAE...")
    try:
        vae = AutoencoderKL.from_pretrained(MODEL_ID_VAE, subfolder="vae", cache_dir=CACHE_DIR, torch_dtype=torch.float16)
    except:
        vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae", cache_dir=CACHE_DIR, torch_dtype=torch.float16)
    vae.to(device)
    
    print("Loading Text Encoder...")
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID_TEXT, cache_dir=CACHE_DIR)
    text_encoder = CLIPTextModel.from_pretrained(MODEL_ID_TEXT, cache_dir=CACHE_DIR, torch_dtype=torch.float16)
    text_encoder.to(device)
    
    return vae, tokenizer, text_encoder

def process_data(data_dir, output_dir, img_size=1024):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    vae, tokenizer, text_encoder = setup_models(device)
    
    # Simple recursive search for images
    extensions = ['*.jpg', '*.png', '*.jpeg']
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob(os.path.join(data_dir, '**', ext), recursive=True))
    
    print(f"Found {len(image_paths)} images.")
    
    for img_path in tqdm(image_paths):
        # 1. Load and process Image (VAE)
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((img_size, img_size), Image.LANCZOS)
            # Transform to tensor [-1, 1]
            img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 127.5 - 1.0
            img_tensor = img_tensor.unsqueeze(0).to(device, dtype=torch.float16)
            
            with torch.no_grad():
                latents = vae.encode(img_tensor).latent_dist.sample() * vae.config.scaling_factor
            
            # 2. Process Text (CLIP)
            # Look for a .txt file with same name
            txt_path = os.path.splitext(img_path)[0] + ".txt"
            if os.path.exists(txt_path):
                with open(txt_path, 'r', encoding='utf-8') as f:
                    caption = f.read().strip()
            else:
                caption = "" # Empty caption
            
            tokens = tokenizer(
                caption, 
                padding="max_length", 
                max_length=77, 
                truncation=True, 
                return_tensors="pt"
            ).input_ids.to(device)
            
            with torch.no_grad():
                text_embeddings = text_encoder(tokens)[0] # (1, 77, 768)
            
            # 3. Save
            file_name = os.path.basename(img_path).split('.')[0]
            save_path = os.path.join(output_dir, file_name)
            np.savez(save_path, 
                     latent=latents.cpu().numpy(), 
                     text_embed=text_embeddings.cpu().numpy(),
                     caption=caption)
                     
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to raw images")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save processed .npz files")
    parser.add_argument("--size", type=int, default=1024, help="Image resolution")
    args = parser.parse_args()
    
    process_data(args.data_dir, args.output_dir, args.size)
