
import os
import random
import numpy as np
import torch


def set_all_seeds(seed: int) -> torch.Generator:
    """Seed ``random``, NumPy and PyTorch (CPU and all CUDA devices), and put
    cuDNN in deterministic mode.

    Returns a seeded :class:`torch.Generator` for DataLoaders and samplers.
    Runs repeat exactly on the same hardware and library versions; results can
    still differ across GPU models or PyTorch releases.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return torch.Generator().manual_seed(seed)