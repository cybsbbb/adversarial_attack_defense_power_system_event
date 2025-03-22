import torch
import random
import numpy as np


def setup_random_seeds(seed=428):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    return
