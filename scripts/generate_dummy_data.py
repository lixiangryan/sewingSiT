import os
import random
import numpy as np
from PIL import Image

def generate_dummy_data(data_dir, num_samples=5):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    print(f"Generating {num_samples} dummy samples in {data_dir}...")
    
    prompts = [
        "A photo of a cute cat sitting on a bench",
        "A futuristic cityscape with flying cars at night",
        "An abstract painting of geometric shapes",
        "A delicious plate of spaghetti carbonara",
        "A majestic dragon flying over a mountain"
    ]
    
    for i in range(num_samples):
        # Allow reuse of prompts if num_samples > len(prompts)
        prompt = prompts[i % len(prompts)]
        
        # Create random image
        # Using random noise to simulate an image
        img_array = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        file_base = f"sample_{i:03d}"
        
        # Save Image
        img_path = os.path.join(data_dir, f"{file_base}.jpg")
        img.save(img_path)
        
        # Save Text
        txt_path = os.path.join(data_dir, f"{file_base}.txt")
        with open(txt_path, 'w') as f:
            f.write(prompt)
            
    print("Done generating dummy data.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="dummy_dataset", help="Path to save dummy data")
    args = parser.parse_args()
    
    generate_dummy_data(args.data_dir)
