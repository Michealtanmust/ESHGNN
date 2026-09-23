import time
import torch
from typing import Dict, Optional

from config import ModelConfig
from models import ESHGNN
from utils import move_to_device, get_device


class ESHGNNInference:
    def __init__(self, model: ESHGNN, config: ModelConfig, device: Optional[torch.device] = None):
        self.config = config
        self.device = device or get_device()

        self.model = model.to(self.device)
        self.model.eval()

        if config.use_multi_gpu and torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)

    @torch.no_grad()
    def predict(self, batch: Dict) -> Dict:
        batch = move_to_device(batch, self.device)

        outputs = self.model(
            batch['r'], batch['factor_returns'],
            batch['ic'], batch['rank'],
            batch['sector'], batch['features'],
            batch['state'], batch['events'],
            batch['volatility'],
            batch['dcc_rho'], batch['tail_upper'],
            batch['tail_lower'], batch['granger_mask'],
            batch['attention']
        )

        if self.device.type == 'cuda':
            outputs = {k: v.cpu() if isinstance(v, torch.Tensor) else v
                      for k, v in outputs.items()}

        return outputs

    def compute_efficiency(self, batch: Dict) -> Dict:
        if self.device.type == 'cuda':
            for _ in range(3):
                _ = self.predict(batch)
            torch.cuda.synchronize(self.device)

        start_time = time.time()
        outputs = self.predict(batch)

        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

        time_ms = (time.time() - start_time) * 1000

        spikes = outputs['spikes']
        effective_steps = (spikes > 0).sum().item()
        total_steps = spikes.numel()
        sparsity = effective_steps / total_steps if total_steps > 0 else 0

        gpu_memory = {}
        if self.device.type == 'cuda':
            gpu_memory = {
                'allocated': torch.cuda.memory_allocated(self.device) / 1024**3,
                'reserved': torch.cuda.memory_reserved(self.device) / 1024**3,
            }

        return {
            'time_ms': time_ms,
            'sparsity': sparsity,
            'effective_steps': effective_steps,
            'total_steps': total_steps,
            'outputs': outputs,
            'gpu_memory': gpu_memory
        }

    def batch_predict(self, batches: list) -> list:
        results = []
        for batch in batches:
            results.append(self.predict(batch))
        return results