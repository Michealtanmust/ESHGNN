"""
主程序 - 超参数配置和运行入口
"""
import argparse
import torch
import os
import time

from config import ModelConfig
from models import ESHGNN
from data_loader import DataManager, SimpleLoader
from trainer import ESHGNNTrainer
from inference import ESHGNNInference
from utils import get_device, get_device_info, set_seed, ensure_dir, print_gpu_memory


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='ESH-GNN: 事件驱动脉冲异质图神经网络')

    # 数据参数
    parser.add_argument('--data_dir', type=str, default='./data', help='数据目录')
    parser.add_argument('--dataset', type=str, default='CSI-300',
                        choices=['CSI-300', 'S&P-500'], help='数据集选择')
    parser.add_argument('--num_nodes', type=int, default=20, help='节点数量')

    # 模型参数
    parser.add_argument('--num_features', type=int, default=8, help='特征维度')
    parser.add_argument('--num_factors', type=int, default=5, help='因子数量')
    parser.add_argument('--hidden_dim', type=int, default=16, help='隐藏层维度')
    parser.add_argument('--gru_dim', type=int, default=8, help='GRU维度')
    parser.add_argument('--state_dim', type=int, default=5, help='状态维度')
    parser.add_argument('--num_layers', type=int, default=1, help='GNN层数')

    # 脉冲神经网络参数
    parser.add_argument('--tau_base', type=float, default=20.0, help='基础时间常数')
    parser.add_argument('--gamma', type=float, default=0.5, help='波动率敏感系数')
    parser.add_argument('--theta_0', type=float, default=0.8, help='基础阈值')
    parser.add_argument('--rho', type=float, default=1.5, help='阈值波动系数')
    parser.add_argument('--T_s', type=int, default=6, help='模拟步数')
    parser.add_argument('--T_e', type=int, default=30, help='事件影响窗口')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=2, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减')
    parser.add_argument('--max_epochs', type=int, default=5, help='最大训练轮数')
    parser.add_argument('--patience', type=int, default=20, help='早停耐心值')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='梯度累积步数')

    # 预测参数
    parser.add_argument('--H', type=int, default=5, help='预测未来H日收益')

    # 脉冲激活参数
    parser.add_argument('--alpha', type=float, default=0.7, help='脉冲强度权重')
    parser.add_argument('--beta', type=float, default=0.3, help='CV_ISI惩罚权重')

    # GPU参数
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('--use_multi_gpu', action='store_true', help='使用多GPU训练')
    parser.add_argument('--no_amp', action='store_true', help='禁用混合精度训练')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载worker数量')
    parser.add_argument('--no_pin_memory', action='store_true', help='禁用pin_memory')

    # 其他参数
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--mode', type=str, default='both',
                        choices=['train', 'inference', 'both'], help='运行模式')
    parser.add_argument('--model_path', type=str, default='best_eshgnn.pt', help='模型权重路径')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='检查点目录')

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    start_time = time.time()

    print("=" * 60)
    print("ESH-GNN: 事件驱动脉冲异质图神经网络")
    print("基于 ICLR 2026 论文实现 (GPU支持)")
    print("=" * 60)

    # 设置随机种子
    set_seed(args.seed)

    # 检查GPU
    device = get_device()
    gpu_info = get_device_info()
    print(f"设备: {device}")
    if gpu_info['device_count'] > 0:
        print(f"GPU数量: {gpu_info['device_count']}, GPU型号: {gpu_info['device_name']}")
        print(f"GPU显存: {gpu_info['memory_total']:.2f} GB")

    # 创建检查点目录
    ensure_dir(args.checkpoint_dir)

    # 创建配置
    config = ModelConfig(
        num_nodes=args.num_nodes,
        num_features=args.num_features,
        num_factors=args.num_factors,
        hidden_dim=args.hidden_dim,
        gru_dim=args.gru_dim,
        state_dim=args.state_dim,
        tau_base=args.tau_base,
        gamma=args.gamma,
        theta_0=args.theta_0,
        rho=args.rho,
        T_s=args.T_s,
        T_e=args.T_e,
        num_layers=args.num_layers,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        H=args.H,
        alpha=args.alpha,
        beta=args.beta,
        data_dir=args.data_dir,
        dataset=args.dataset,
        gpu_id=args.gpu_id,
        use_multi_gpu=args.use_multi_gpu,
        use_amp=not args.no_amp,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_memory
    )

    # 设置GPU设备
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        print(f"使用GPU: {args.gpu_id}")

    # 加载数据
    data_manager = DataManager(config)
    data = data_manager.get_data()

    # 创建数据加载器
    train_loader = SimpleLoader(data['train'])
    val_loader = SimpleLoader(data['val'])
    test_loader = SimpleLoader(data['test'])

    # 创建模型
    model = ESHGNN(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数量: {total_params:,}")

    if args.mode in ['train', 'both']:
        print("\n开始训练...")
        trainer = ESHGNNTrainer(model, config, device)
        trainer.train(train_loader, val_loader)

        # 保存最终模型
        final_path = os.path.join(args.checkpoint_dir, 'final_eshgnn.pt')
        trainer._save_checkpoint(final_path, args.max_epochs, trainer.best_val_loss)
        print(f"最终模型已保存到 {final_path}")

        # 打印GPU显存使用
        if torch.cuda.is_available():
            memory = trainer.get_gpu_memory_usage()
            print(f"训练后GPU显存: 已分配 {memory.get('allocated', 0):.2f}GB, "
                  f"已预留 {memory.get('reserved', 0):.2f}GB")

    if args.mode in ['inference', 'both']:
        print("\n" + "=" * 60)
        print("推理演示...")

        # 加载模型
        if args.mode == 'inference':
            checkpoint_path = args.model_path if os.path.exists(args.model_path) else \
                             os.path.join(args.checkpoint_dir, 'final_eshgnn.pt')
            print(f"加载模型: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)

        inference = ESHGNNInference(model, config, device)

        # 使用测试数据进行推理
        for i, batch in enumerate(test_loader):
            result = inference.compute_efficiency(batch)

            print(f"\n测试样本 {i+1}:")
            print(f"推理耗时: {result['time_ms']:.2f}ms")
            print(f"脉冲稀疏度: {result['sparsity']:.2%}")
            print(f"有效更新步: {result['effective_steps']:,} / {result['total_steps']:,}")
            print(f"预测收益范围: [{result['outputs']['r_pred'].min().item():.4f}, "
                  f"{result['outputs']['r_pred'].max().item():.4f}]")

            if result['gpu_memory']:
                print(f"GPU显存: 已分配 {result['gpu_memory']['allocated']:.2f}GB, "
                      f"已预留 {result['gpu_memory']['reserved']:.2f}GB")

            lam = result['outputs']['lambda']
            print(f"协调权重 λ: 基本面={lam[0,0].item():.3f}, "
                  f"风险={lam[0,1].item():.3f}, 注意力={lam[0,2].item():.3f}")

            if i >= 2:  # 只显示前3个
                break

    elapsed = time.time() - start_time
    print(f"\n总运行时间: {elapsed:.2f}秒")
    print("=" * 60)
    print("运行完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()