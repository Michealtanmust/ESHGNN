import torch
import torch.nn as nn
import torch.nn.functional as F


class HeterogeneousGraphConstructor(nn.Module):
    def __init__(self, num_features: int, state_dim: int, sigma: float = 0.5, eta_tail: float = 0.35):
        super().__init__()
        self.num_features = num_features
        self.sigma = sigma
        self.eta_tail = eta_tail

        self.fund_mlp = nn.Sequential(
            nn.Linear(state_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
            nn.Softplus()
        )

    def build_fundamental_graph(self, sector: torch.Tensor, features: torch.Tensor,
                                state: torch.Tensor) -> torch.Tensor:
        batch, N, _ = features.shape

        sector_onehot = F.one_hot(sector, num_classes=10).float()
        sector_sim = torch.matmul(sector_onehot, sector_onehot.transpose(-2, -1))

        feat_norm = F.normalize(features, p=2, dim=-1)
        cos_sim = torch.matmul(feat_norm, feat_norm.transpose(-2, -1))
        cos_sim = torch.clamp(cos_sim, -1.0, 1.0)

        beta = self.fund_mlp(state)
        beta1, beta2 = beta[:, 0:1], beta[:, 1:2]

        weights = beta1.unsqueeze(-1) * sector_sim + beta2.unsqueeze(-1) * cos_sim
        weights = torch.sigmoid(weights)

        mask = weights > 0.65
        weights = weights * mask.float()
        weights = weights + torch.eye(N, device=weights.device).unsqueeze(0)

        return weights

    def build_risk_graph(self, dcc_rho: torch.Tensor, tail_upper: torch.Tensor,
                         tail_lower: torch.Tensor, granger_mask: torch.Tensor) -> torch.Tensor:
        weights = self.eta_tail * tail_upper + (1 - self.eta_tail) * tail_lower
        weights = weights * granger_mask.float()

        N = weights.shape[-1]
        weights = weights + torch.eye(N, device=weights.device).unsqueeze(0)

        return weights

    def build_sentiment_graph(self, attention: torch.Tensor) -> torch.Tensor:
        batch, T_att, N = attention.shape

        att_norm = F.normalize(attention, p=2, dim=1)
        dtw_dist = torch.cdist(att_norm.transpose(-2, -1), att_norm.transpose(-2, -1), p=2)

        weights = torch.exp(-dtw_dist ** 2 / (2 * self.sigma ** 2))
        mask = dtw_dist < 0.30
        weights = weights * mask.float()
        weights = weights + torch.eye(N, device=weights.device).unsqueeze(0)

        return weights