from dataclasses import dataclass

@dataclass
class ModelConfig:
    num_nodes: int = 50
    num_features: int = 80
    num_factors: int = 10
    hidden_dim: int = 64
    gru_dim: int = 32
    state_dim: int = 5

    tau_base: float = 20.0
    gamma: float = 0.5
    theta_0: float = 0.8
    rho: float = 1.5
    V_rest: float = 0.0
    T_s: int = 8
    T_e: int = 30
    num_layers: int = 2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 4
    max_epochs: int = 30
    patience: int = 20
    H: int = 5
    alpha: float = 0.7
    beta: float = 0.3
    data_dir: str = "./data"
    dataset: str = "CSI-300"

    gpu_id: int = 0
    use_multi_gpu: bool = False
    use_amp: bool = True
    gradient_accumulation_steps: int = 1
    num_workers: int = 4
    pin_memory: bool = True