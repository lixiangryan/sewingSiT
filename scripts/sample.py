import torch
import argparse
import sys
import os
import numpy as np
from PIL import Image
from diffusers import AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from models import SiT_models
from transport import Transport, ModelType, PathType, WeightType, Sampler

MODEL_ID_VAE = "segmind/SSD-1B"
MODEL_ID_TEXT = "openai/clip-vit-large-patch14"
CACHE_DIR = "/mnt/f/nccu/project/diffusion/models"

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Load VAE & Text Encoder
    print("Loading VAE and Text Encoder...")
    try:
        vae = AutoencoderKL.from_pretrained(MODEL_ID_VAE, subfolder="vae", cache_dir=CACHE_DIR, torch_dtype=torch.float16).to(device)
    except:
        vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae", cache_dir=CACHE_DIR, torch_dtype=torch.float16).to(device)

    tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID_TEXT, cache_dir=CACHE_DIR)
    text_encoder = CLIPTextModel.from_pretrained(MODEL_ID_TEXT, cache_dir=CACHE_DIR, torch_dtype=torch.float16).to(device)
    
    # 2. Load Trained SiT Model
    print(f"Loading SiT model from {args.checkpoint}...")
    # NOTE: img_size must match training latent size (32 for 256px image)
    model = SiT_models[args.model](
        input_size=args.latent_size, 
        in_channels=4, 
        context_dim=768, 
        learn_sigma=False
    ).to(device)
    
    # Load weights
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    # 3. Setup Transport Sampler
    transport = Transport(
        model_type=ModelType.VELOCITY, 
        path_type=PathType.LINEAR,
        loss_type=WeightType.NONE, # Just placeholder for inference
        train_eps=1e-3, 
        sample_eps=1e-3
    )
    sampler = Sampler(transport)
    
    # 4. Prepare Input Prompt
    prompts = [args.prompt] * args.num_samples
    print(f"Generating {args.num_samples} images for prompt: '{args.prompt}'")
    
    with torch.no_grad():
        # Encode Text
        tokens = tokenizer(
            prompts, 
            padding="max_length", 
            max_length=77, 
            truncation=True, 
            return_tensors="pt"
        ).input_ids.to(device)
        context = text_encoder(tokens)[0].float() # SiT expects float32 usually, unless fully fp16
        
        # Prepare Model Kwargs
        model_kwargs = dict(context=context)
        
        # Initial Noise (x0) - Transport integrates from Noise (0) to Data (1)
        z = torch.randn(args.num_samples, 4, args.latent_size, args.latent_size, device=device)
        
        # Sampling Loop (ODE Solver)
        # 1. Get the sampling function
        sample_fn = sampler.sample_ode(
            sampling_method="dopri5", 
            num_steps=50,
        )
        
        # 2. Execute sampling
        samples = sample_fn(
            z, 
            model, 
            **model_kwargs
        )
        
        # samples is the final latent at t=1
        latents = samples[-1]
        
        # 5. Decode Latents to Images
        latents = latents.to(torch.float16) / vae.config.scaling_factor
        images = vae.decode(latents).sample
        
        # Post-process
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.permute(0, 2, 3, 1).cpu().numpy()
        images = (images * 255).round().astype("uint8")
        
        # Save
        os.makedirs("output", exist_ok=True)
        for i, img_arr in enumerate(images):
            pil_img = Image.fromarray(img_arr)
            save_name = f"output/sample_{i}_{args.prompt.replace(' ', '_')[:20]}.png"
            pil_img.save(save_name)
            print(f"Saved: {save_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="A beautiful landscape")
    parser.add_argument("--model", type=str, default="SiT-B/2")
    parser.add_argument("--latent_size", type=int, default=32, help="Size of latent map (32 for 256px)")
    parser.add_argument("--num_samples", type=int, default=1)
    args = parser.parse_args()
    
    main(args)
