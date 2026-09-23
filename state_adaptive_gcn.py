import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from config import ModelConfig


class StateAdaptiveGCN(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, state_dim: int, config: ModelConfig):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.config = config

        self.W = nn.Linear(in_dim, out_dim)

    def forward(self, h: torch.Tensor, adj_mats: List[torch.Tensor],
                lam: torch.Tensor, z: torch.Tensor, activation: torch.Tensor) -> torch.Tensor:
        batch, N, _ = h.shape

        h_new = torch.zeros(batch, N, self.out_dim, device=h.device)

        for k, adj in enumerate(adj_mats):
            deg = adj.sum(dim=-1, keepdim=True) + 1e-6
            adj_norm = adj / deg

            agg = torch.matmul(adj_norm, h)
            lambda_k = lam[:, k].unsqueeze(-1).unsqueeze(-1)
            h_new = h_new + lambda_k * self.W(agg)

        h_new = F.relu(h_new)
        return h_new