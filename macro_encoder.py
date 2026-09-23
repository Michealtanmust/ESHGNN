import torch
import torch.nn as nn
import torch.nn.functional as F


class MACROEncoder(nn.Module):
    def __init__(self, state_dim: int, num_channels: int = 3):
        super().__init__()
        self.u = nn.Parameter(torch.randn(num_channels, state_dim) * 0.01)
        self.num_channels = num_channels

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [batch, state_dim]
        logits = torch.matmul(z, self.u.T)  # [batch, num_channels]
        return F.softmax(logits, dim=-1)