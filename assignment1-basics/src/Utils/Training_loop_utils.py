import numpy as np
from typing import Tuple, Optional
import torch
import os
def data_loader(dataset: np.ndarray, batch_size: int, context_length: int , device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
    max_start_idx = len(dataset) - context_length
    start_indices = np.random.randint(0, max_start_idx, size=batch_size)

    indices_x = start_indices[:, None] + np.arange(context_length)[None, :]
    indices_y = start_indices[:, None] + np.arange(1, context_length + 1)[None, :]
    x = np.array(dataset[indices_x], dtype=np.uint16)
    y = np.array(dataset[indices_y], dtype=np.uint16)
    
    x_tensor = torch.from_numpy(x).to(device)
    y_tensor = torch.from_numpy(y).to(device)
    return x_tensor, y_tensor

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str):
    # 如果输出路径的目录不存在，则创建目录
    output_dir = os.path.dirname(out)
    if output_dir:  # 如果目录路径不为空
        os.makedirs(output_dir, exist_ok=True)
    
    checkpoint = {
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, out)

def load_checkpoint(src: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, map_location: str = "cpu") -> int:
    checkpoint = torch.load(src, map_location=map_location)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['iteration']