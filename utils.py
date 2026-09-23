import os
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def get_device_info():
    if torch.cuda.is_available():
        return {
            'device': 'cuda',
            'device_count': torch.cuda.device_count(),
            'device_name': torch.cuda.get_device_name(0),
            'memory_allocated': torch.cuda.memory_allocated(0) / 1024**3,
            'memory_reserved': torch.cuda.memory_reserved(0) / 1024**3,
            'memory_total': torch.cuda.get_device_properties(0).total_memory / 1024**3
        }
    return {'device': 'cpu', 'device_count': 0}


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict:
    mse = torch.nn.functional.mse_loss(pred, target).item()
    mae = torch.nn.functional.l1_loss(pred, target).item()
    return {'mse': mse, 'mae': mae}


def move_to_device(data: Dict, device: torch.device) -> Dict:
    if device.type == 'cpu':
        return data

    result = {}
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.to(device)
        elif isinstance(value, dict):
            result[key] = move_to_device(value, device)
        elif isinstance(value, list):
            result[key] = [v.to(device) if isinstance(v, torch.Tensor) else v for v in value]
        else:
            result[key] = value
    return result


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def print_gpu_memory():
    if torch.cuda.is_available():
        print(f"GPU:  {torch.cuda.memory_allocated() / 1024**3:.2f}GB, "
              f"{torch.cuda.memory_reserved() / 1024**3:.2f}GB")