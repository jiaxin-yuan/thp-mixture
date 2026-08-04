
import os
import random
import numpy as np
import torch


def set_all_seeds(seed: int) -> torch.Generator:
    """Seed random, NumPy and torch, make cuDNN deterministic, return a seeded DataLoader generator."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return torch.Generator().manual_seed(seed)