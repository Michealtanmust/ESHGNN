"""
训练器模块 - 支持GPU训练，优化损失函数
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
from torch.cuda.amp import GradScaler, autocast

from config import ModelConfig
from models import ESHGNN
from utils import AverageMeter, move_to_device, get_device, get_device_info, print_gpu_memory


class ESHGNNTrainer:
    """ESH-GNN模型训练器 - 支持GPU训练"""

    def __init__(self, model: ESHGNN, config: ModelConfig, device: Optional[torch.device] = None):
        self.config = config
        self.device = device or get_device()

        # 模型移到设备
        self.model = model.to(self.device)

        # 多GPU支持
        if config.use_multi_gpu and torch.cuda.device_count() > 1:
            print(f"使用 {torch.cuda.device_count()} 个GPU进行训练")
            self.model = nn.DataParallel(self.model)

        # 优化器 - 使用更高的学习率
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999)
        )

        # 学习率调度器 - 使用更激进的调度
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=config.max_epochs // 3, T_mult=2, eta_min=1e-6
        )

        # 混合精度训练
        self.use_amp = config.use_amp and torch.cuda.is_available()
        self.scaler = GradScaler(enabled=self.use_amp)

        # 梯度累积
        self.gradient_accumulation_steps = config.gradient_accumulation_steps

        # 训练状态
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []

        # 打印GPU信息
        if torch.cuda.is_available():
            info = get_device_info()
            print(f"GPU信息: {info['device_name']}, 显存: {info['memory_total']:.2f}GB")
            if self.use_amp:
                print("混合精度训练: 启用")
            if self.gradient_accumulation_steps > 1:
                print(f"梯度累积步数: {self.gradient_accumulation_steps}")

    def train_epoch(self, dataloader) -> float:
        """训练一个epoch"""
        self.model.train()
        loss_meter = AverageMeter()
        mae_meter = AverageMeter()
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(dataloader):
            # 将数据移到GPU
            batch = move_to_device(batch, self.device)

            # 混合精度前向传播
            with autocast(enabled=self.use_amp):
                outputs = self.model(
                    batch['r'], batch['factor_returns'],
                    batch['ic'], batch['rank'],
                    batch['sector'], batch['features'],
                    batch['state'], batch['events'],
                    batch['volatility'],
                    batch['dcc_rho'], batch['tail_upper'],
                    batch['tail_lower'], batch['granger_mask'],
                    batch['attention']
                )

                pred = outputs['r_pred']
                target = batch['y']

                # 组合损失：MSE + MAE + 排名损失
                # 1. MSE损失（主要）
                mse_loss = F.mse_loss(pred, target)

                # 2. MAE损失（对异常值更鲁棒）
                mae_loss = F.l1_loss(pred, target)

                # 3. 排名损失：鼓励预测保持正确的相对顺序
                # 使用Spearman秩相关的近似
                rank_loss = self._ranking_loss(pred, target)

                # 4. 辅助损失
                aux_loss = outputs['aux_loss']

                # 5. 正则化损失
                reg_loss = 0.0
                for param in self.model.parameters():
                    reg_loss += param.norm() * 1e-5

                # 组合损失 - 给MAE和排名损失更大权重
                loss = mse_loss + 0.5 * mae_loss + 0.3 * rank_loss + 0.1 * aux_loss + reg_loss
                loss = loss / self.gradient_accumulation_steps

            # 反向传播
            self.scaler.scale(loss).backward()

            # 梯度累积
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            # 记录损失
            loss_meter.update(loss.item() * self.gradient_accumulation_steps)
            mae_meter.update(mae_loss.item())

            # 清理GPU缓存
            if torch.cuda.is_available() and batch_idx % 10 == 0:
                torch.cuda.empty_cache()

        return loss_meter.avg

    def _ranking_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        排名损失 - 鼓励预测的排序与目标一致
        使用成对比较的近似
        """
        batch_size, N, _ = pred.shape

        # 展平
        pred_flat = pred.view(batch_size, -1)  # [batch, N]
        target_flat = target.view(batch_size, -1)  # [batch, N]

        # 计算每个batch内的排名相关性（近似）
        # 使用成对差异的平方损失
        loss = 0.0
        for b in range(batch_size):
            # 计算预测和目标的差异矩阵
            pred_diff = pred_flat[b:b+1] - pred_flat[b:b+1].T  # [N, N]
            target_diff = target_flat[b:b+1] - target_flat[b:b+1].T  # [N, N]

            # 符号一致性损失
            sign_loss = torch.sigmoid(-pred_diff * target_diff).mean()
            loss = loss + sign_loss

        return loss / batch_size

    def validate(self, dataloader) -> Tuple[float, float, float]:
        """验证 - 返回MSE, MAE, 预测范围"""
        self.model.eval()
        loss_meter = AverageMeter()
        mae_meter = AverageMeter()
        pred_min = float('inf')
        pred_max = float('-inf')
        target_min = float('inf')
        target_max = float('-inf')

        with torch.no_grad():
            for batch in dataloader:
                batch = move_to_device(batch, self.device)

                outputs = self.model(
                    batch['r'], batch['factor_returns'],
                    batch['ic'], batch['rank'],
                    batch['sector'], batch['features'],
                    batch['state'], batch['events'],
                    batch['volatility'],
                    batch['dcc_rho'], batch['tail_upper'],
                    batch['tail_lower'], batch['granger_mask'],
                    batch['attention']
                )

                pred = outputs['r_pred']
                target = batch['y']

                mse_loss = F.mse_loss(pred, target)
                mae_loss = F.l1_loss(pred, target)

                loss_meter.update(mse_loss.item())
                mae_meter.update(mae_loss.item())

                # 记录范围
                pred_min = min(pred_min, pred.min().item())
                pred_max = max(pred_max, pred.max().item())
                target_min = min(target_min, target.min().item())
                target_max = max(target_max, target.max().item())

        return loss_meter.avg, mae_meter.avg, pred_min, pred_max, target_min, target_max

    def train(self, train_loader, val_loader, epochs: int = None):
        """完整训练流程"""
        epochs = epochs or self.config.max_epochs

        print(f"初始预测范围: 目标范围待观察")

        for epoch in range(epochs):
            # 训练
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)

            # 验证
            val_loss, val_mae, pred_min, pred_max, target_min, target_max = self.validate(val_loader)
            self.val_losses.append(val_loss)

            # 学习率调度
            self.scheduler.step()

            # 打印
            lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.6f}, "
                  f"Val MSE={val_loss:.6f}, Val MAE={val_mae:.6f}, LR={lr:.2e}")
            print(f"  预测范围: [{pred_min:.4f}, {pred_max:.4f}], "
                  f"目标范围: [{target_min:.4f}, {target_max:.4f}]")

            # 计算预测与目标的范围比
            if target_max - target_min > 1e-6:
                pred_range = pred_max - pred_min
                target_range = target_max - target_min
                range_ratio = pred_range / target_range
                print(f"  范围比率: {range_ratio:.3f}")

            # 打印GPU内存
            if torch.cuda.is_available():
                print_gpu_memory()

            # 早停检查
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_checkpoint('best_eshgnn.pt', epoch, val_loss)
                print(f"  -> 保存最佳模型, Val MSE={val_loss:.6f}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

    def _save_checkpoint(self, path: str, epoch: int, val_loss: float):
        """保存检查点"""
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.use_amp else None,
            'best_val_loss': val_loss,
            'config': self.config
        }
        torch.save(state, path)

    def load_checkpoint(self, path: str, load_optimizer: bool = False):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)

        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])

        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if self.use_amp and checkpoint.get('scaler_state_dict'):
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            self.best_val_loss = checkpoint['best_val_loss']

        return checkpoint['epoch']

    def get_gpu_memory_usage(self) -> Dict:
        """获取GPU显存使用情况"""
        if not torch.cuda.is_available():
            return {}

        return {
            'allocated': torch.cuda.memory_allocated(self.device) / 1024**3,
            'reserved': torch.cuda.memory_reserved(self.device) / 1024**3,
            'max_allocated': torch.cuda.max_memory_allocated(self.device) / 1024**3,
        }