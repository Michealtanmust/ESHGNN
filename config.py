"""
配置类定义
"""
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """模型配置参数"""
    # 数据维度 - 增加特征数量
    num_nodes: int = 50
    num_features: int = 80  # 增加特征维度
    num_factors: int = 10   # 增加因子数量
    hidden_dim: int = 64    # 增加隐藏层维度
    gru_dim: int = 32       # 增加GRU维度
    state_dim: int = 5

    # 脉冲神经网络参数
    tau_base: float = 20.0
    gamma: float = 0.5
    theta_0: float = 0.8
    rho: float = 1.5
    V_rest: float = 0.0
    T_s: int = 8    # 增加模拟步数
    T_e: int = 30

    # 图结构参数 - 增加层数
    num_layers: int = 2

    # 训练参数 - 提高学习率
    learning_rate: float = 1e-3  # 提高学习率
    weight_decay: float = 1e-4   # 降低权重衰减
    batch_size: int = 4
    max_epochs: int = 30
    patience: int = 20

    # 预测参数
    H: int = 5

    # 脉冲激活参数
    alpha: float = 0.7
    beta: float = 0.3

    # 数据参数
    data_dir: str = "./data"
    dataset: str = "CSI-300"

    # GPU参数
    gpu_id: int = 0
    use_multi_gpu: bool = False
    use_amp: bool = True
    gradient_accumulation_steps: int = 1
    num_workers: int = 4
    pin_memory: bool = True