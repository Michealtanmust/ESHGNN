import argparse
import torch
import os
import time

from config import ModelConfig
from eshgnn import ESHGNN
from data_loader import DataManager, SimpleLoader
from trainer import ESHGNNTrainer
from inference import ESHGNNInference
from utils import get_device, get_device_info, set_seed, ensure_dir, print_gpu_memory


def parse_args():
    parser = argparse.ArgumentParser(description='ESH-GNN:')
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--dataset', type=str, default='CSI-300',
                        choices=['CSI-300', 'S&P-500'], help='Dataset selection')
    parser.add_argument('--num_nodes', type=int, default=20, help='Number of nodes')

    # Model parameters
    parser.add_argument('--num_features', type=int, default=8, help='Feature dimension')
    parser.add_argument('--num_factors', type=int, default=5, help='Number of factors')
    parser.add_argument('--hidden_dim', type=int, default=16, help='Hidden layer dimension')
    parser.add_argument('--gru_dim', type=int, default=8, help='GRU dimension')
    parser.add_argument('--state_dim', type=int, default=5, help='State dimension')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of GNN layers')

    parser.add_argument('--tau_base', type=float, default=20.0, help='Base time constant')
    parser.add_argument('--gamma', type=float, default=0.5, help='Volatility sensitivity coefficient')
    parser.add_argument('--theta_0', type=float, default=0.8, help='Base threshold')
    parser.add_argument('--rho', type=float, default=1.5, help='Threshold fluctuation coefficient')
    parser.add_argument('--T_s', type=int, default=6, help='Number of simulation steps')
    parser.add_argument('--T_e', type=int, default=30, help='Event influence window')

    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--max_epochs', type=int, default=5, help='Maximum number of training epochs')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation steps')

    parser.add_argument('--H', type=int, default=5, help='Predict future H-day returns')

    parser.add_argument('--alpha', type=float, default=0.7, help='Pulse intensity weight')
    parser.add_argument('--beta', type=float, default=0.3, help='CV_ISI penalty weight')

    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('--use_multi_gpu', action='store_true', help='Use multi-GPU training')
    parser.add_argument('--no_amp', action='store_true', help='Disable mixed precision training')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--no_pin_memory', action='store_true', help='Disable pin_memory')

    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--mode', type=str, default='both',
                        choices=['train', 'inference', 'both'], help='Run mode')
    parser.add_argument('--model_path', type=str, default='best_eshgnn.pt', help='Model weight path')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Checkpoint directory')

    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()

    set_seed(args.seed)

    device = get_device()
    gpu_info = get_device_info()
    if gpu_info['device_count'] > 0:
        print(f"GPU count: {gpu_info['device_count']}, GPU model: {gpu_info['device_name']}")
        print(f"GPU memory: {gpu_info['memory_total']:.2f} GB")

    ensure_dir(args.checkpoint_dir)

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

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)

    data_manager = DataManager(config)
    data = data_manager.get_data()

    train_loader = SimpleLoader(data['train'])
    val_loader = SimpleLoader(data['val'])
    test_loader = SimpleLoader(data['test'])

    model = ESHGNN(config)
    total_params = sum(p.numel() for p in model.parameters())

    if args.mode in ['train', 'both']:
        trainer = ESHGNNTrainer(model, config, device)
        trainer.train(train_loader, val_loader)

        final_path = os.path.join(args.checkpoint_dir, 'final_eshgnn.pt')
        trainer._save_checkpoint(final_path, args.max_epochs, trainer.best_val_loss)

        if torch.cuda.is_available():
            memory = trainer.get_gpu_memory_usage()

    if args.mode in ['inference', 'both']:
        if args.mode == 'inference':
            checkpoint_path = args.model_path if os.path.exists(args.model_path) else \
                             os.path.join(args.checkpoint_dir, 'final_eshgnn.pt')
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)

        inference = ESHGNNInference(model, config, device)

        for i, batch in enumerate(test_loader):
            result = inference.compute_efficiency(batch)
            if result['gpu_memory']:
                print(f"GPU memory: allocated {result['gpu_memory']['allocated']:.2f}GB, "
                      f"reserved {result['gpu_memory']['reserved']:.2f}GB")


            if i >= 2:  # Only display the first 3
                break

    elapsed = time.time() - start_time
    print(f"\nTotal runtime: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()