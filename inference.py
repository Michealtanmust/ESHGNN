"""
推理引擎模块 - 支持GPU推理
"""
import time
import torch
from typing import Dict, Optional

from config import ModelConfig
from models import ESHGNN
from utils import move_to_device, get_device


class ESHGNNInference:
    """ESH-GNN推理引擎 - 支持GPU推理"""

    def __init__(self, model: ESHGNN, config: ModelConfig, device: Optional[torch.device] = None):
        self.config = config
        self.device = device or get_device()

        # 模型移到设备
        self.model = model.to(self.device)
        self.model.eval()

        # 多GPU支持
        if config.use_multi_gpu and torch.cuda.device_count() > 1:
            print(f"使用 {torch.cuda.device_count()} 个GPU进行推理")
            self.model = torch.nn.DataParallel(self.model)

        print(f"推理设备: {self.device}")

    @torch.no_grad()
    def predict(self, batch: Dict) -> Dict:
        """预测 - 自动处理设备"""
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

        # 将结果移回CPU
        if self.device.type == 'cuda':
            outputs = {k: v.cpu() if isinstance(v, torch.Tensor) else v
                      for k, v in outputs.items()}

        return outputs

    def compute_efficiency(self, batch: Dict) -> Dict:
        """计算推理效率指标"""
        # 预热（仅GPU）
        if self.device.type == 'cuda':
            for _ in range(3):
                _ = self.predict(batch)
            torch.cuda.synchronize(self.device)

        # 计时
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
        """批量预测"""
        results = []
        for batch in batches:
            results.append(self.predict(batch))
        return results