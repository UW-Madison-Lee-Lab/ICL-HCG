import torch

def to_python_float(x):
    if isinstance(x, torch.Tensor):
        # Assumes `x` is a 1-element tensor (scalar) on CPU or CUDA
        return x.item()
    else:
        return float(x)
