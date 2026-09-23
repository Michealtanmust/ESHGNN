import torch
import torch.nn as nn
from typing import List, Tuple

from config import ModelConfig
from .lif_neuron import LIFNeuron
from .state_adaptive_gcn import StateAdaptiveGCN


class SpikingGNNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, state_dim: int, config: ModelConfig):
        super().__init__()
        self.config = config

        self.lif = LIFNeuron(config)
        self.sa_gcn = StateAdaptiveGCN(in_dim, out_dim, state_dim, config)

    def forward(self, h: torch.Tensor, adj_mats: List[torch.Tensor],
                lam: torch.Tensor, z: torch.Tensor,
                events: torch.Tensor, volatility: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, N, _ = h.shape

        tau_m = self.config.tau_base * torch.exp(-self.config.gamma * volatility.unsqueeze(-1))
        tau_m = tau_m.expand(batch, N)

        theta_high = self.config.theta_0 * (1 + self.config.rho * volatility.unsqueeze(-1) / 21.5)
        theta_high = theta_high.expand(batch, N)
        theta_low = 0.5 * theta_high

        sigma_i = volatility.unsqueeze(-1).expand(batch, N) * 10
        ext_current = self.lif.compute_ext_current(events, sigma_i)
        V = torch.zeros(batch, N, device=h.device)
        h_gcn = self.sa_gcn(h, adj_mats, lam, z, torch.zeros(batch, N, device=h.device))

        T_s = self.config.T_s
        if T_s <= 0:
            T_s = 1
        spike_input = h_gcn.mean(dim=-1).unsqueeze(-1).expand(-1, -1, T_s)
        V_updated, spikes = self.lif(V, spike_input, ext_current, tau_m, theta_high, theta_low)
        activation = self.lif.compute_activation(spikes)
        h_new = self.sa_gcn(h, adj_mats, lam, z, activation)

        return h_new, activation, spikes