import torch
import sys
import os

# Add the src folder to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import SiT_models

def test_model():
    model_name = 'SiT-B/2' # Smaller model for quick test
    img_size = 32
    in_channels = 4
    context_dim = 768
    
    # Instantiate
    print(f"Instantiating {model_name}...")
    model = SiT_models[model_name](
        input_size=img_size, 
        in_channels=in_channels, 
        context_dim=context_dim
    )
    model.eval()
    
    # Dummy inputs
    N = 2
    x = torch.randn(N, in_channels, img_size, img_size)
    t = torch.randint(0, 1000, (N,))
    context = torch.randn(N, 77, context_dim) # 77 tokens (CLIP standard)
    
    print("Running forward pass with context shape:", context.shape)
    with torch.no_grad():
        out = model(x, t, context)
    
    print(f"Output shape: {out.shape}")
    # learn_sigma=True by default so out channels = in_channels * 2
    expected_shape = (N, in_channels * 2, img_size, img_size) 
    assert out.shape == expected_shape, f"Expected {expected_shape}, got {out.shape}"
    print("Success!")

if __name__ == "__main__":
    test_model()
