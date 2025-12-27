import os
import json
import argparse
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from diffusers import AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
import random

# Reuse the setup logic from preprocess_data.py
# Ideally we should refactor this into src.utils later
MODEL_ID_VAE = "segmind/SSD-1B"
MODEL_ID_TEXT = "openai/clip-vit-large-patch14"
CACHE_DIR = "/mnt/f/nccu/project/diffusion/models"

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

def process_coco(img_dir, ann_file, output_dir, img_size=1024):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Annotations
    print(f"Loading annotations from {ann_file}...")
    with open(ann_file, 'r') as f:
        coco = json.load(f)
        
    # Create image_id -> filename mapping
    # COCO JSON structure: {'images': [{'id': 123, 'file_name': '000.jpg'}, ...], 'annotations': [{'image_id': 123, 'caption': '...'}, ...]}
    img_map = {img['id']: img['file_name'] for img in coco['images']}
    
    # Group captions by image_id
    captions = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        if img_id not in captions:
            captions[img_id] = []
        captions[img_id].append(ann['caption'])
        
    print(f"Found {len(img_map)} images and {len(coco['annotations'])} captions.")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 2. Setup Models
    vae, tokenizer, text_encoder = setup_models(device)
    
    # 3. Processing Loop
    success_count = 0
    
    for img_id, file_name in tqdm(img_map.items()):
        img_path = os.path.join(img_dir, file_name)
        if not os.path.exists(img_path):
            continue
            
        save_path = os.path.join(output_dir, os.path.splitext(file_name)[0]) + ".npz"
        if os.path.exists(save_path):
            success_count += 1
            continue # Skip existing
            
        try:
            # --- Image Processing ---
            img = Image.open(img_path).convert("RGB")
            img = img.resize((img_size, img_size), Image.LANCZOS)
            # [-1, 1]
            img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 127.5 - 1.0
            img_tensor = img_tensor.unsqueeze(0).to(device, dtype=torch.float16)
            
            with torch.no_grad():
                latents = vae.encode(img_tensor).latent_dist.sample() * vae.config.scaling_factor
            
            # --- Text Processing ---
            # Pick one random caption for now to save space. 
            # Or save all? Let's save one random caption to keep things simple for now.
            # Ideally for training, you might want to dynamically sample captions, 
            # but pre-computing embeddings means we fix the text now.
            caption = random.choice(captions[img_id])
            
            tokens = tokenizer(
                caption, 
                padding="max_length", 
                max_length=77, 
                truncation=True, 
                return_tensors="pt"
            ).input_ids.to(device)
            
            with torch.no_grad():
                text_embeddings = text_encoder(tokens)[0]
                
            # --- Save ---
            np.savez(save_path, 
                     latent=latents.cpu().numpy(), 
                     text_embed=text_embeddings.cpu().numpy(),
                     caption=caption) # Save raw caption too just in case
            
            success_count += 1
            
        except Exception as e:
            # print(f"Error processing {file_name}: {e}")
            pass
            
    print(f"Successfully processed {success_count} images.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, required=True, help="Path to COCO images (e.g., train2017)")
    parser.add_argument("--ann_file", type=str, required=True, help="Path to COCO annotation json")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save .npz files")
    parser.add_argument("--size", type=int, default=1024)
    args = parser.parse_args()
    
    process_coco(args.img_dir, args.ann_file, args.output_dir, args.size)
