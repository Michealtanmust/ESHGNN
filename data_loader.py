"""
数据下载和加载模块 - 从本地CSV文件读取股票数据
"""
import os
import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')

from config import ModelConfig
from utils import ensure_dir


class StockDataLoader:
    """
    股票数据加载器 - 从CSV文件读取数据
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.data_dir = config.data_dir
        self.num_nodes = config.num_nodes
        self.num_features = config.num_features
        self.num_factors = config.num_factors
        self.T = 20
        self.T_att = 15

        # 特征列名 - 使用技术指标作为特征
        self.feature_columns = [
            'ma_tt_5', 'ma_tt_10', 'ma_tt_20', 'ma_tt_60',
            'macd_tt_dif', 'macd_tt_dea', 'macd_tt_macd',
            'rsi_tt_3', 'rsi_tt_14', 'rsi_tt_6', 'rsi_tt_12',
            'boll_tt_upper', 'boll_tt_mid', 'boll_tt_lower',
            'high_low_gap', 'swing_rate', 'swing_volatility_5_0',
            'c_h_rate_500', 'l_c_rate_500', 'c_h_rate_250', 'l_c_rate_250',
            'ref_close_rate_20', 'ref_close_rate_10', 'ref_close_rate_5', 'ref_close_rate_3',
            'up_amount_rate_5', 'vhf_tt_5', 'atr_tt_1_gui1_ma14', 'mfi_tt_14',
            'bias_tt_6', 'bias_tt_12', 'bias_tt_24', 'cci_tt_14',
            'STD_3', 'STD_5', 'STD_10', 'STD_20', 'STD_60', 'STD_120',
            'RVI_3', 'RVI_6', 'RVI_12', 'RVI_14',
            'RSV_3', 'RSV_5', 'RSV_10', 'RSV_20', 'RSV_30', 'RSV_60', 'RSV_120', 'RSV_240',
            'BETA_3', 'BETA_5', 'BETA_10', 'BETA_20', 'BETA_30', 'BETA_60', 'BETA_120', 'BETA_240',
            'MAX_3', 'MAX_5', 'MAX_10', 'MAX_20', 'MAX_30', 'MAX_60', 'MAX_120', 'MAX_240',
            'MIN_3', 'MIN_5', 'MIN_10', 'MIN_20', 'MIN_30', 'MIN_60', 'MIN_120', 'MIN_240',
            'stochastic_ratio_3', 'stochastic_ratio_5', 'stochastic_ratio_10',
            'stochastic_ratio_20', 'stochastic_ratio_60', 'stochastic_ratio_120',
            'Volume_Ratio_3', 'Volume_Ratio_5', 'Volume_Ratio_10',
            'Volume_Ratio_20', 'Volume_Ratio_60', 'Volume_Ratio_120',
            'Volume_Oscillator_3', 'Volume_Oscillator_5', 'Volume_Oscillator_6',
            'Volume_Oscillator_12', 'Volume_Oscillator_20', 'Volume_Oscillator_60',
            'Volume_std_3', 'Volume_std_5', 'Volume_std_10',
            'Volume_std_20', 'Volume_std_60', 'Volume_std_120',
            'ATR_3', 'ATR_5', 'ATR_12', 'ATR_10', 'ATR_20', 'ATR_60', 'ATR_120',
            'ama_5', 'efficiency_ratio_5', 'smoothing_constant_5',
            'ama_10', 'efficiency_ratio_10', 'smoothing_constant_10',
            'ama_20', 'efficiency_ratio_20', 'smoothing_constant_20',
            'ama_40', 'efficiency_ratio_40', 'smoothing_constant_40',
            'lyyl_rineibodong', 'circulating_market_cap', 'circulating_cap',
            'turnoverRatio'
        ]

        self.factor_columns = [
            'ma_tt_5', 'ma_tt_10', 'ma_tt_20', 'ma_tt_60',
            'rsi_tt_14', 'rsi_tt_6', 'macd_tt_dif', 'boll_tt_mid',
            'turnoverRatio', 'atr_tt_1_gui1_ma14'
        ]
        self.state_columns = ['swing_volatility_5_0', 'atr_tt_1_gui1_ma14', 'turnoverRatio']
        # 使用原始收益率作为目标，而不是标准化后的
        self.target_column = 'ref_close_rate_5'  # 5日收益率（原始值）

    def load_stock_data(self, stock_code: str) -> pd.DataFrame:
        """加载单个股票的CSV数据"""
        file_path = os.path.join(self.data_dir, f"{stock_code}.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据文件不存在: {file_path}")

        df = pd.read_csv(file_path)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)

        return df

    def preprocess_data(self, df: pd.DataFrame) -> Dict:
        """预处理单个股票数据 - 保留原始收益范围"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # 先提取原始收益率（在标准化之前）
        target_values = None
        if self.target_column in df.columns:
            target_values = df[self.target_column].values

        # 填充缺失值
        df_numeric = df[numeric_cols].fillna(method='ffill').fillna(0)

        # 使用RobustScaler对特征进行标准化（对异常值更鲁棒）
        # 但保留原始收益率范围
        scaler = RobustScaler()
        df_scaled = pd.DataFrame(
            scaler.fit_transform(df_numeric),
            columns=numeric_cols
        )

        # 如果目标列存在，用原始值替换标准化后的值
        if target_values is not None and self.target_column in df_scaled.columns:
            df_scaled[self.target_column] = target_values

        # 提取特征
        features = []
        for col in self.feature_columns:
            if col in df_scaled.columns:
                features.append(df_scaled[col].values)
            else:
                features.append(np.zeros(len(df_scaled)))
        features = np.array(features).T

        # 提取因子特征
        factor_features = []
        for col in self.factor_columns:
            if col in df_scaled.columns:
                factor_features.append(df_scaled[col].values)
            else:
                factor_features.append(np.zeros(len(df_scaled)))
        factor_features = np.array(factor_features).T

        # 提取状态特征
        state_features = []
        for col in self.state_columns:
            if col in df_scaled.columns:
                state_features.append(df_scaled[col].values[-1])
            else:
                state_features.append(0.0)
        state_features = np.array(state_features)

        # 目标值（原始收益率）
        target = df_scaled[self.target_column].values[-1] if self.target_column in df_scaled.columns else 0.0

        # 收益率序列（用于因子编码器）
        returns = df_scaled['close'].pct_change().fillna(0).values if 'close' in df_scaled.columns else np.zeros(len(df_scaled))

        # 计算波动率
        if 'close' in df_scaled.columns:
            volatility = df_scaled['close'].pct_change().std()
            # 事件信号：价格波动超过2倍标准差
            events = (np.abs(df_scaled['close'].pct_change()) > 2 * volatility).astype(float)
        else:
            volatility = 1.0
            events = np.zeros(len(df_scaled))

        return {
            'features': features,
            'factor_features': factor_features,
            'returns': returns,
            'state': state_features,
            'events': events,
            'volatility': volatility,
            'target': target,  # 原始收益率
            'close': df_scaled['close'].values if 'close' in df_scaled.columns else np.zeros(len(df_scaled)),
            'attention': factor_features,
        }

    def get_stock_list(self) -> List[str]:
        """获取所有股票代码列表"""
        stock_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        return [f.replace('.csv', '') for f in stock_files]

    def prepare_data(self) -> Dict[str, List]:
        """准备训练、验证、测试数据"""
        stock_list = self.get_stock_list()
        if len(stock_list) > self.num_nodes:
            stock_list = stock_list[:self.num_nodes]

        print(f"加载 {len(stock_list)} 个股票数据...")

        all_data = []
        for stock_code in stock_list:
            try:
                df = self.load_stock_data(stock_code)
                if len(df) < self.T + 10:
                    continue
                processed = self.preprocess_data(df)
                all_data.append(processed)
            except Exception as e:
                print(f"加载股票 {stock_code} 失败: {e}")
                continue

        # 按股票数量划分
        num_stocks = len(all_data)
        train_end = int(num_stocks * 0.7)
        val_end = int(num_stocks * 0.85)

        train_data = self._create_batches(all_data[:train_end])
        val_data = self._create_batches(all_data[train_end:val_end])
        test_data = self._create_batches(all_data[val_end:])

        return {'train': train_data, 'val': val_data, 'test': test_data}

    def _create_batches(self, data_list: List[Dict]) -> List[Dict]:
        """创建批次数据"""
        if not data_list:
            return []

        batch_size = self.config.batch_size
        num_batches = max(1, len(data_list) // batch_size)

        batches = []
        for i in range(num_batches):
            start = i * batch_size
            end = min(start + batch_size, len(data_list))
            batch_data = data_list[start:end]
            batch = self._merge_batch(batch_data)
            batches.append(batch)

        return batches

    def _merge_batch(self, batch_data: List[Dict]) -> Dict:
        """合并批次数据"""
        batch_size = len(batch_data)
        T = self.T
        N = batch_size
        K = len(self.factor_columns)  # 使用实际因子数量
        F = len(self.feature_columns)  # 使用实际特征数量
        S = self.config.state_dim

        device = torch.device('cpu')

        # 初始化张量
        r = torch.zeros(batch_size, T, N, device=device)
        factor_returns = torch.zeros(batch_size, T, K, device=device)
        ic = torch.zeros(batch_size, K, device=device)
        rank = torch.zeros(batch_size, K, device=device)
        sector = torch.randint(0, 5, (batch_size, N), device=device)
        features = torch.zeros(batch_size, N, F, device=device)
        state = torch.zeros(batch_size, S, device=device)
        events = torch.zeros(batch_size, N, device=device)
        volatility = torch.zeros(batch_size, device=device)
        dcc_rho = torch.eye(N, device=device).unsqueeze(0).repeat(batch_size, 1, 1) * 0.3
        tail_upper = torch.rand(batch_size, N, N, device=device) * 0.3
        tail_lower = torch.rand(batch_size, N, N, device=device) * 0.3
        granger_mask = torch.bernoulli(torch.full((batch_size, N, N), 0.05, device=device))
        attention = torch.zeros(batch_size, self.T_att, N, device=device)
        y = torch.zeros(batch_size, N, 1, device=device)

        for b, data in enumerate(batch_data):
            # 收益序列
            returns_data = data['returns'][-T:] if len(data['returns']) >= T else data['returns']
            feature_len = min(T, len(returns_data))
            r[b, :feature_len, b] = torch.tensor(returns_data[:feature_len], dtype=torch.float32)

            # 因子特征
            factor_data = data['factor_features'][-T:, :K] if len(data['factor_features']) >= T else data['factor_features']
            factor_len = min(T, len(factor_data))
            factor_returns[b, :factor_len, :] = torch.tensor(factor_data[:factor_len, :K], dtype=torch.float32)

            # 节点特征
            features_data = data['features'][-1, :F] if len(data['features']) > 0 else np.zeros(F)
            features[b, b, :min(F, len(features_data))] = torch.tensor(
                features_data[:min(F, len(features_data))], dtype=torch.float32
            )

            # 状态
            state[b, :min(S, len(data['state']))] = torch.tensor(data['state'][:min(S, len(data['state']))], dtype=torch.float32)

            # 事件
            events[b, b] = torch.tensor(data['events'][-1] if len(data['events']) > 0 else 0, dtype=torch.float32)

            # 波动率
            volatility[b] = torch.tensor(data['volatility'], dtype=torch.float32)

            # 目标值（原始收益率）
            y[b, b, 0] = torch.tensor(data['target'], dtype=torch.float32)

            # 随机生成ic和rank
            ic[b, :] = torch.randn(K, device=device) * 0.1
            rank[b, :] = torch.randn(K, device=device) * 0.1

        return {
            'r': r, 'factor_returns': factor_returns, 'ic': ic, 'rank': rank,
            'sector': sector, 'features': features, 'state': state,
            'events': events, 'volatility': volatility,
            'dcc_rho': dcc_rho, 'tail_upper': tail_upper, 'tail_lower': tail_lower,
            'granger_mask': granger_mask, 'attention': attention, 'y': y
        }


class DataManager:
    """数据管理器"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.data_dir = config.data_dir
        ensure_dir(self.data_dir)
        self.stock_loader = StockDataLoader(config)

    def get_data(self) -> Dict[str, List]:
        """获取数据"""
        print(f"从 {self.data_dir} 加载股票数据...")

        if not os.path.exists(self.data_dir):
            print(f"数据目录 {self.data_dir} 不存在，生成模拟数据...")
            return self._generate_empty_data()

        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        if not csv_files:
            print(f"数据目录 {self.data_dir} 中没有CSV文件，生成模拟数据...")
            return self._generate_empty_data()

        try:
            data = self.stock_loader.prepare_data()
            print(f"数据加载完成: 训练集 {len(data['train'])} 批次, "
                  f"验证集 {len(data['val'])} 批次, 测试集 {len(data['test'])} 批次")
            return data
        except Exception as e:
            print(f"加载数据失败: {e}，生成模拟数据...")
            return self._generate_empty_data()

    def _generate_empty_data(self) -> Dict[str, List]:
        """生成模拟数据 - 使用更大的收益范围"""
        print("生成模拟数据用于测试...")
        generator = SyntheticDataGenerator(self.config)
        return {
            'train': [generator.generate_batch(self.config.batch_size) for _ in range(5)],
            'val': [generator.generate_batch(self.config.batch_size) for _ in range(3)],
            'test': [generator.generate_batch(self.config.batch_size) for _ in range(3)]
        }


class SyntheticDataGenerator:
    """合成数据生成器 - 使用更真实的收益范围"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.N = config.num_nodes
        self.K = config.num_factors
        self.T = 20
        self.T_att = 15

    def generate_batch(self, batch_size: int) -> Dict:
        """生成一个batch的合成数据 - 收益范围更真实"""
        device = torch.device('cpu')

        # 使用更真实的收益范围 (-0.1, 0.1)
        returns = torch.randn(batch_size, self.T, self.N, device=device) * 0.02

        return {
            'r': returns,
            'factor_returns': torch.randn(batch_size, self.T, self.K, device=device) * 0.02,
            'ic': torch.randn(batch_size, self.K, device=device) * 0.1,
            'rank': torch.randn(batch_size, self.K, device=device) * 0.1,
            'sector': torch.randint(0, 5, (batch_size, self.N), device=device),
            'features': torch.randn(batch_size, self.N, 8, device=device),
            'state': torch.randn(batch_size, self.config.state_dim, device=device),
            'events': torch.rand(batch_size, self.N, device=device) * 0.5,
            'volatility': torch.randn(batch_size, device=device).abs() * 0.1 + 0.15,
            'dcc_rho': torch.randn(batch_size, self.N, self.N, device=device) * 0.3,
            'tail_upper': torch.randn(batch_size, self.N, self.N, device=device).abs() * 0.3,
            'tail_lower': torch.randn(batch_size, self.N, self.N, device=device).abs() * 0.3,
            'granger_mask': torch.bernoulli(torch.full((batch_size, self.N, self.N), 0.05, device=device)),
            'attention': torch.randn(batch_size, self.T_att, self.N, device=device),
            # 目标值使用更大的范围 (-0.15, 0.15)
            'y': torch.randn(batch_size, self.N, 1, device=device) * 0.05
        }


class SimpleLoader:
    """简单的数据加载器"""
    def __init__(self, data):
        self.data = data
    def __iter__(self):
        for batch in self.data:
            yield batch
    def __len__(self):
        return len(self.data)