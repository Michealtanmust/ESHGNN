import torch
import torch.nn as nn
from typing import Dict

from config import ModelConfig
from .macro_encoder import MACROEncoder
from .factor_encoder import FactorEncoder
from .graph_constructor import HeterogeneousGraphConstructor
from .spiking_gnn_layer import SpikingGNNLayer


class ESHGNN(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        hidden_dim = max(config.hidden_dim, 32)
        gru_dim = max(config.gru_dim, 16)

        self.macro_encoder = MACROEncoder(config.state_dim, 3)
        self.factor_encoder = FactorEncoder(config.num_factors, config.num_features)
        self.graph_constructor = HeterogeneousGraphConstructor(
            config.num_features, config.state_dim
        )

        self.spiking_layers = nn.ModuleList()
        in_dim = config.num_factors
        for l in range(max(config.num_layers, 2)):
            out_dim = hidden_dim if l < config.num_layers - 1 else hidden_dim
            self.spiking_layers.append(
                SpikingGNNLayer(in_dim, out_dim, config.state_dim, config)
            )
            in_dim = out_dim

        self.gru = nn.GRU(hidden_dim, gru_dim, batch_first=True, num_layers=2, dropout=0.1)

        self.predictor = nn.Sequential(
            nn.Linear(gru_dim, gru_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(gru_dim // 2, 1)
        )

        self.alpha_predictor = nn.Sequential(
            nn.Linear(gru_dim, gru_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(gru_dim // 2, config.num_factors)
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GRU):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

    def compute_auxiliary_loss(self, r_pred: torch.Tensor, factor_returns: torch.Tensor,
                                h_time: torch.Tensor) -> torch.Tensor:
        batch, N, _ = r_pred.shape

        beta_hat = self.alpha_predictor(h_time)
        factor_expected = factor_returns.mean(dim=1, keepdim=True)
        alpha = r_pred - torch.sum(beta_hat * factor_expected, dim=-1, keepdim=True)

        return alpha.abs().mean()

    def forward(self, r: torch.Tensor, factor_returns: torch.Tensor,
                ic: torch.Tensor, rank: torch.Tensor,
                sector: torch.Tensor, features: torch.Tensor,
                state: torch.Tensor, events: torch.Tensor,
                volatility: torch.Tensor,
                dcc_rho: torch.Tensor, tail_upper: torch.Tensor,
                tail_lower: torch.Tensor, granger_mask: torch.Tensor,
                attention: torch.Tensor) -> Dict[str, torch.Tensor]:

        batch = r.shape[0]
        lam = self.macro_encoder(state)
        x = self.factor_encoder(r, factor_returns, ic, rank)

        adj_fund = self.graph_constructor.build_fundamental_graph(sector, features, state)
        adj_risk = self.graph_constructor.build_risk_graph(dcc_rho, tail_upper, tail_lower, granger_mask)
        adj_sent = self.graph_constructor.build_sentiment_graph(attention)
        adj_mats = [adj_fund, adj_risk, adj_sent]

        h = x
        activations = []
        spikes_list = []

        for layer in self.spiking_layers:
            h, activation, spikes = layer(h, adj_mats, lam, state, events, volatility)
            activations.append(activation)
            spikes_list.append(spikes)

        h_time, _ = self.gru(h)

        r_pred = self.predictor(h_time)

        aux_loss = self.compute_auxiliary_loss(r_pred, factor_returns, h_time)

        return {
            'r_pred': r_pred,
            'lambda': lam,
            'activation': activations[-1] if activations else torch.zeros_like(x[:, :, 0]),
            'spikes': spikes_list[-1] if spikes_list else torch.zeros(batch, x.shape[1], 1, device=x.device),
            'aux_loss': aux_loss,
            'h_time': h_time
        }