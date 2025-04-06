import os
import torch
import numpy as np
import random
import pickle

def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seeds set to {seed}")

def save_init_weights(model, path, name="init_weights"):
    """Save initial weights for reproducibility"""
    os.makedirs(path, exist_ok=True)
    save_path = os.path.join(path, f"{name}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Initial weights saved to {save_path}")
    return save_path

def load_init_weights(model, path, strict=False):
    """Load initial weights for reproducibility"""
    print("loading the weights at:", path)
    
    # Load state dict with strict=False to allow missing keys
    state_dict = torch.load(path)
    model.load_state_dict(state_dict, strict=strict)
    
    if not strict:
        # Get info about what was loaded and what wasn't
        current_state = model.state_dict()
        missing_keys = [k for k in current_state.keys() if k not in state_dict]
        print(f"Initial weights loaded from {path}")
        print(f"- Loaded {len(current_state) - len(missing_keys)} parameters")
        print(f"- Randomly initialized {len(missing_keys)} missing parameters (transformer layers)")
    else:
        print(f"Initial weights loaded from {path} (strict mode)")
        
    return model