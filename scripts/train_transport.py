import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import sys
import glob
import argparse
import wandb

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from models import SiT_models
from transport import Transport, ModelType, PathType, WeightType

# --- 1. Dataset (Same as before) ---
class PrecomputedDataset(Dataset):
    def __init__(self, data_dir):
        self.files = glob.glob(os.path.join(data_dir, "*.npz"))
        if len(self.files) == 0:
            print(f"Warning: No .npz files found in {data_dir}.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        data = np.load(path)
        latent = torch.from_numpy(data['latent']).squeeze(0).float()
        text_embed = torch.from_numpy(data['text_embed']).squeeze(0).float()
        return latent, text_embed

# --- 2. Advanced Training Script with Transport & WandB ---
def train(args):
    # Setup WandB
    if args.use_wandb:
        wandb.init(project="sewing-sit", name=args.run_name, config=args)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Setup Data
    dataset = PrecomputedDataset(args.data_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Setup Model
    model = SiT_models[args.model](
        input_size=args.img_size,
        in_channels=4,
        context_dim=768,
        learn_sigma=False  # Important for Velocity Matching (output dim == input dim)
    ).to(device)
    model.train()
    
    # Setup Transport (The Core Magic)
    # Using Velocity prediction with Linear path (Rectified Flow / InstaFlow style) is SOTA usually.
    transport = Transport(
        model_type=ModelType.VELOCITY, 
        path_type=PathType.LINEAR,
        loss_type=WeightType.NONE, # Velocity matching usually uses no weighting or standard weighting
        train_eps=1e-3, 
        sample_eps=1e-3
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    print(f"Start training {args.model} with Transport...")
    
    global_step = 0
    for epoch in range(args.epochs):
        epoch_loss = 0
        for step, (x, context) in enumerate(dataloader):
            x = x.to(device)
            context = context.to(device)
            
            # Prepare kwargs for the model (to be passed via Transport)
            model_kwargs = dict(context=context)
            
            # Calculate Transport Loss
            # x is treated as x1 (data), Transport handles x0 (noise) sampling internally
            loss_dict = transport.training_losses(model, x, model_kwargs)
            loss = loss_dict['loss'].mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            if args.use_wandb and global_step % 10 == 0:
                wandb.log({"train/loss": loss.item(), "epoch": epoch})
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f}")
        
    print("Training finished.")
    
    # Save
    os.makedirs("checkpoints", exist_ok=True)
    save_path = os.path.join("checkpoints", f"{args.run_name}.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Saved model to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--model", type=str, default="SiT-B/2")
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--run_name", type=str, default="sit_velocity_test")
    parser.add_argument("--use_wandb", action="store_true")
    args = parser.parse_args()
    
    train(args)
