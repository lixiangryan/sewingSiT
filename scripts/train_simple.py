import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import sys
import glob

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from models import SiT_models

# --- 1. Dataset ---
class PrecomputedDataset(Dataset):
    def __init__(self, data_dir):
        self.files = glob.glob(os.path.join(data_dir, "*.npz"))
        if len(self.files) == 0:
            print(f"Warning: No .npz files found in {data_dir}. Please run preprocess_data.py first.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        data = np.load(path)
        # Latents: (1, 4, 128, 128) -> squeeze to (4, 128, 128)
        latent = torch.from_numpy(data['latent']).squeeze(0).float()
        # Text Embed: (1, 77, 768) -> squeeze to (77, 768)
        text_embed = torch.from_numpy(data['text_embed']).squeeze(0).float()
        return latent, text_embed

# --- 2. Training Script ---
def train():
    # Config
    data_dir = "data/processed"
    model_name = "SiT-B/2" # Use small model for testing
    batch_size = 2
    epochs = 5
    lr = 1e-4
    img_size = 128 # Size of Latent (1024 / 8)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Device: {device}")
    
    # 1. Setup Data
    dataset = PrecomputedDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"Dataset size: {len(dataset)}")
    
    # 2. Setup Model
    model = SiT_models[model_name](
        input_size=img_size,
        in_channels=4,
        context_dim=768 # CLIP ViT-L
    ).to(device)
    model.train()
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    
    print(f"Start training {model_name} for {epochs} epochs...")
    
    for epoch in range(epochs):
        epoch_loss = 0
        for step, (x, context) in enumerate(dataloader):
            x = x.to(device)       # (B, 4, 128, 128)
            context = context.to(device) # (B, 77, 768)
            
            # --- Diffusion Process (Simplified) ---
            # Sample random timesteps
            t = torch.randint(0, 1000, (x.shape[0],), device=device)
            
            # Standard Diffusion Training:
            # Noise prediction objective
            noise = torch.randn_like(x)
            
            # Add noise (Simple linear scheduler for demo, reusing SiT logic would need Transport)
            # IMPORTANT: Real SiT uses 'transport' module for exact noise scheduling.
            # Here we just demo the Forward Pass hookup.
            # Let's import transport if possible, or just mock it.
            # To avoid complexity, we just check if model.forward runs and loss backprops.
            # For a real run, we need the rectifed flow / transport ode coefficients.
            
            # Let's just feed noise as input for a "dry run" of validity
            model_output = model(x, t, context)
            
            # Split output (Mean + Variance)
            model_pred, _ = model_output.chunk(2, dim=1) # (B, 4, 128, 128)
            
            # Simple loss (Dummy target for structural test)
            loss = mse_loss(model_pred, noise if False else x) # Just checking gradients
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(dataloader):.4f}")
        
    print("Training loop finished successfully!")
    
    # Save checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/sit_text_test.pt")
    print("Saved model to checkpoints/sit_text_test.pt")

if __name__ == "__main__":
    train()
