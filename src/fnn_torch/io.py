import numpy as np
import torch

def load_tensor(filename, encoding):
    data = np.fromfile(filename, dtype=encoding)
    return torch.from_numpy(data)
