import torch
import torch.nn as nn
from typing import Tuple

from config import ModelConfig


class LIFNeuron(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.tau_base = config.tau_base
        self.gamma = config.gamma
        self.theta_0 = config.theta_0
        self.rho = config.rho
        self.V_rest = config.V_rest
        self.T_s = config.T_s
        self.T_e = config.T_e

    def compute_ext_current(self, events: torch.Tensor, sigma_i: torch.Tensor) -> torch.Tensor:
        batch, N = events.shape
        T_s = self.T_s

        if T_s <= 0:
            return torch.zeros(batch, N, 1, device=events.device)

        t = torch.arange(T_s, device=events.device).float().view(1, 1, -1)
        event_pos = max(1, T_s // 2)

        I_ext = torch.zeros(batch, N, T_s, device=events.device)

        for b in range(batch):
            for n in range(N):
                if events[b, n] > 0.3:
                    pulse = torch.exp(-(t - event_pos) ** 2 / (2 * (sigma_i[b, n] + 1e-6) ** 2))
                    I_ext[b, n] = events[b, n] * pulse

        I_ext = I_ext * (t < self.T_e).float()
        return I_ext

    def forward(self, V: torch.Tensor, spike_input: torch.Tensor,
                ext_current: torch.Tensor, tau_m: torch.Tensor,
                theta_high: torch.Tensor, theta_low: torch.Tensor,
                dt: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, N = V.shape
        T_s = spike_input.shape[-1]

        spikes = torch.zeros(batch, N, T_s, device=V.device)
        V_curr = V.clone()

        for t in range(T_s):
            dV = -1 / (tau_m + 1e-6) * (V_curr - self.V_rest) + spike_input[:, :, t] + ext_current[:, :, t]
            V_curr = V_curr + dt * dV

            spike_level = torch.zeros_like(V_curr)
            spike_level[V_curr >= theta_high] = 2.0
            spike_level[(V_curr >= theta_low) & (V_curr < theta_high)] = 1.0

            V_curr[V_curr >= theta_high] = self.V_rest
            V_curr[(V_curr >= theta_low) & (V_curr < theta_high)] = self.V_rest * 0.5

            spikes[:, :, t] = spike_level

        return V_curr, spikes

    def compute_activation(self, spikes: torch.Tensor) -> torch.Tensor:
        batch, N, T_s = spikes.shape

        total_spikes = spikes.sum(dim=-1)

        cv_isi = torch.zeros(batch, N, device=spikes.device)
        for b in range(batch):
            for n in range(N):
                spike_times = torch.where(spikes[b, n] > 0)[0]
                if len(spike_times) > 1:
                    isi = (spike_times[1:] - spike_times[:-1]).float()
                    cv_isi[b, n] = isi.std() / (isi.mean() + 1e-6)

        activation = self.config.alpha * total_spikes - self.config.beta * cv_isi
        return activation