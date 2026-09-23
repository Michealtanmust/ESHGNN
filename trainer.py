import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
from torch.cuda.amp import GradScaler, autocast

from config import ModelConfig
from models import ESHGNN
from utils import AverageMeter, move_to_device, get_device, get_device_info, print_gpu_memory


class ESHGNNTrainer:
    def __init__(self, model: ESHGNN, config: ModelConfig, device: Optional[torch.device] = None):
        self.config = config
        self.device = device or get_device()

        self.model = model.to(self.device)

        if config.use_multi_gpu and torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999)
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=config.max_epochs // 3, T_mult=2, eta_min=1e-6
        )

        self.use_amp = config.use_amp and torch.cuda.is_available()
        self.scaler = GradScaler(enabled=self.use_amp)

        self.gradient_accumulation_steps = config.gradient_accumulation_steps

        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []

        if torch.cuda.is_available():
            info = get_device_info()
            print(f"GPU info: {info['device_name']}, memory: {info['memory_total']:.2f}GB")
            if self.use_amp:
                print("Mixed precision training: enabled")
            if self.gradient_accumulation_steps > 1:
                print(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")

    def train_epoch(self, dataloader) -> float:
        self.model.train()
        loss_meter = AverageMeter()
        mae_meter = AverageMeter()
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(dataloader):
            batch = move_to_device(batch, self.device)

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
                mse_loss = F.mse_loss(pred, target)
                mae_loss = F.l1_loss(pred, target)
                rank_loss = self._ranking_loss(pred, target)
                aux_loss = outputs['aux_loss']

                reg_loss = 0.0
                for param in self.model.parameters():
                    reg_loss += param.norm() * 1e-5

                loss = mse_loss + 0.5 * mae_loss + 0.3 * rank_loss + 0.1 * aux_loss + reg_loss
                loss = loss / self.gradient_accumulation_steps

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            loss_meter.update(loss.item() * self.gradient_accumulation_steps)
            mae_meter.update(mae_loss.item())

            if torch.cuda.is_available() and batch_idx % 10 == 0:
                torch.cuda.empty_cache()

        return loss_meter.avg

    def _ranking_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        batch_size, N, _ = pred.shape

        pred_flat = pred.view(batch_size, -1)  # [batch, N]
        target_flat = target.view(batch_size, -1)  # [batch, N]

        loss = 0.0
        for b in range(batch_size):
            pred_diff = pred_flat[b:b+1] - pred_flat[b:b+1].T  # [N, N]
            target_diff = target_flat[b:b+1] - target_flat[b:b+1].T  # [N, N]

            sign_loss = torch.sigmoid(-pred_diff * target_diff).mean()
            loss = loss + sign_loss

        return loss / batch_size

    def validate(self, dataloader) -> Tuple[float, float, float]:
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

                # Record ranges
                pred_min = min(pred_min, pred.min().item())
                pred_max = max(pred_max, pred.max().item())
                target_min = min(target_min, target.min().item())
                target_max = max(target_max, target.max().item())

        return loss_meter.avg, mae_meter.avg, pred_min, pred_max, target_min, target_max

    def train(self, train_loader, val_loader, epochs: int = None):
        epochs = epochs or self.config.max_epochs

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)

            val_loss, val_mae, pred_min, pred_max, target_min, target_max = self.validate(val_loader)
            self.val_losses.append(val_loss)

            self.scheduler.step()

    def _save_checkpoint(self, path: str, epoch: int, val_loss: float):
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
        if not torch.cuda.is_available():
            return {}

        return {
            'allocated': torch.cuda.memory_allocated(self.device) / 1024**3,
            'reserved': torch.cuda.memory_reserved(self.device) / 1024**3,
            'max_allocated': torch.cuda.max_memory_allocated(self.device) / 1024**3,
        }