import torch
import torch.nn as nn
import torch.nn.functional as F


class FactorEncoder(nn.Module):
    def __init__(self, num_factors: int, num_features: int, window: int = 20, xi: float = 30.0):
        super().__init__()
        self.num_factors = num_factors
        self.num_features = num_features
        self.window = window
        self.xi = xi

        self.gate_mlp = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

        self.register_buffer('omega', self._compute_weights())

    def _compute_weights(self) -> torch.Tensor:
        tau = torch.arange(self.window, dtype=torch.float32)
        weights = torch.exp(-tau / self.xi)
        return weights / (weights.sum() + 1e-6)

    def forward(self, r: torch.Tensor, factor_returns: torch.Tensor,
                ic: torch.Tensor, rank: torch.Tensor) -> torch.Tensor:
        batch, T, N = r.shape
        K = self.num_factors

        beta = self._estimate_beta(r, factor_returns)

        d = torch.zeros(batch, K, device=r.device)
        for k in range(K):
            gate_input = torch.stack([ic[:, k], rank[:, k]], dim=-1)
            d[:, k] = self.gate_mlp(gate_input).squeeze(-1)

        x = beta * d.unsqueeze(1)
        return x

    def _estimate_beta(self, r: torch.Tensor, factor_returns: torch.Tensor) -> torch.Tensor:
        batch, T, N = r.shape
        K = factor_returns.shape[-1]

        if T > self.window:
            r_window = r[:, -self.window:, :]
            f_window = factor_returns[:, -self.window:, :]
        else:
            r_window = r
            f_window = factor_returns

        W = r_window.shape[1]
        omega = self.omega[:W].view(1, W, 1)

        r_weighted = r_window * omega
        numerator = torch.zeros(batch, N, K, device=r.device)
        for k in range(K):
            f_k = f_window[:, :, k]
            numerator[:, :, k] = torch.sum(r_weighted * f_k.unsqueeze(-1), dim=1)

        denominator = torch.zeros(batch, K, device=r.device)
        for k in range(K):
            f_k = f_window[:, :, k]
            denominator[:, k] = torch.sum(omega.squeeze(-1) * (f_k ** 2), dim=1) + 1e-6

        denominator = denominator.unsqueeze(1)
        beta = numerator / denominator

        return beta