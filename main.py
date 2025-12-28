import os
import sys
import subprocess
import glob

# Default Paths (Relative)
DEFAULT_DATA_DIR = "../data/2dImg/coco_2017/processed_val"
DEFAULT_RAW_DIR = "../data/2dImg/coco_2017/val2017"
DEFAULT_ANN_FILE = "../data/2dImg/coco_2017/annotations/captions_val2017.json"
DEFAULT_TARGET_DIR = "../data/2dImg/coco_2017"

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def get_input(prompt_text, default_value=None):
    if default_value:
        user_input = input(f"{prompt_text} [Default: {default_value}]: ").strip()
        return user_input if user_input else default_value
    else:
        return input(f"{prompt_text}: ").strip()

def list_checkpoints():
    files = glob.glob("checkpoints/*.pt")
    if not files:
        return []
    # Sort by modification time (newest first)
    files.sort(key=os.path.getmtime, reverse=True)
    return files

def run_script(script_name, args_dict):
    cmd = f"python scripts/{script_name} "
    for key, value in args_dict.items():
        if value is True:
            cmd += f"--{key} "
        elif value is False or value is None:
            continue
        else:
            cmd += f"--{key} \"{value}\" "
    
    print(f"\nExecuting: {cmd}\n")
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError:
        print("\nExecution failed.")
    except KeyboardInterrupt:
        print("\nExecution interrupted.")
    
    input("\nPress Enter to return to menu...")

def menu_download():
    print("\n--- Download COCO Data ---")
    target_dir = get_input("Target Directory", DEFAULT_TARGET_DIR)
    split = get_input("Split (test_small/train/val)", "test_small")
    
    run_script("download_coco.py", {
        "target_dir": target_dir,
        "split": split
    })

def menu_process():
    print("\n--- Process Data (Images -> Latents) ---")
    img_dir = get_input("Image Directory", DEFAULT_RAW_DIR)
    ann_file = get_input("Annotation File", DEFAULT_ANN_FILE)
    output_dir = get_input("Output Directory", DEFAULT_DATA_DIR)
    size = get_input("Image Resolution", "256")
    
    run_script("process_coco.py", {
        "img_dir": img_dir,
        "ann_file": ann_file,
        "output_dir": output_dir,
        "size": size
    })

def menu_train():
    print("\n--- Train Model ---")
    data_dir = get_input("Processed Data Directory", DEFAULT_DATA_DIR)
    epochs = get_input("Epochs", "50")
    batch_size = get_input("Batch Size", "32")
    img_size = get_input("Latent Size", "32")
    run_name = get_input("Run Name (for WandB/Checkpoint)", "manual_run")
    use_wandb = get_input("Use WandB? (y/n)", "y").lower() == 'y'
    
    run_script("train_transport.py", {
        "data_dir": data_dir,
        "epochs": epochs,
        "batch_size": batch_size,
        "img_size": img_size,
        "run_name": run_name,
        "use_wandb": use_wandb
    })

def menu_sample():
    print("\n--- Generate Image ---")
    
    # Checkpoint Selection
    checkpoints = list_checkpoints()
    ckpt_path = ""
    
    if checkpoints:
        print("\nSelect Checkpoint:")
        for idx, f in enumerate(checkpoints):
            print(f"[{idx+1}] {f}")
        print(f"[{len(checkpoints)+1}] Enter path manually")
        
        choice = get_input("Choice", "1")
        try:
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(checkpoints):
                ckpt_path = checkpoints[choice_idx]
            else:
                ckpt_path = get_input("Enter Checkpoint Path")
        except:
            ckpt_path = get_input("Enter Checkpoint Path")
    else:
        print("\nNo checkpoints found in checkpoints/.")
        ckpt_path = get_input("Enter Checkpoint Path")

    prompt = get_input("Enter Prompt")
    if not prompt:
        print("Prompt cannot be empty!")
        return

    latent_size = get_input("Latent Size", "32")
    
    run_script("sample.py", {
        "checkpoint": ckpt_path,
        "prompt": prompt,
        "latent_size": latent_size
    })

def main():
    while True:
        clear_screen()
        print("========================================")
        print("   SewingSiT Interactive Menu")
        print("========================================")
        print("1. Download Data")
        print("2. Process Data")
        print("3. Train Model")
        print("4. Generate Image (Sample)")
        print("5. Exit")
        print("========================================")
        
        choice = input("Select an option (1-5): ").strip()
        
        if choice == '1':
            menu_download()
        elif choice == '2':
            menu_process()
        elif choice == '3':
            menu_train()
        elif choice == '4':
            menu_sample()
        elif choice == '5':
            print("Goodbye!")
            sys.exit(0)
        else:
            input("Invalid option. Press Enter to try again...")

if __name__ == "__main__":
    main()
