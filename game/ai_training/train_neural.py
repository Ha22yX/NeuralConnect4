from neural_mcts import NeuralMCTSTrainer
import torch
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='训练四子棋AI')
    parser.add_argument('--batch_size', type=int, default=512, help='批次大小')
    parser.add_argument('--num_simulations', type=int, default=800, help='MCTS模拟次数')
    parser.add_argument('--num_iterations', type=int, default=50, help='训练迭代次数')
    parser.add_argument('--games_per_iteration', type=int, default=100, help='每次迭代的自我对弈局数')
    parser.add_argument('--epochs_per_iteration', type=int, default=5, help='每次迭代的训练轮数')
    parser.add_argument('--load_model', type=str, help='加载已有模型文件')
    parser.add_argument('--use_cpu', action='store_true', help='强制使用CPU训练')
    args = parser.parse_args()

    # 检查是否强制使用CPU
    use_cpu = args.use_cpu or not torch.cuda.is_available()
    
    if use_cpu:
        print("使用CPU进行训练")
        device = torch.device('cpu')
    else:
        print(f"使用GPU进行训练: {torch.cuda.get_device_name(0)}")
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True

    # 创建训练器
    trainer = NeuralMCTSTrainer(
        batch_size=args.batch_size,
        num_simulations=args.num_simulations,
        device=device
    )

    # 加载已有模型（如果指定）
    if args.load_model and os.path.exists(os.path.join('models', args.load_model)):
        print(f"加载模型: {args.load_model}")
        trainer.load_model(args.load_model)

    # 开始训练
    trainer.train(
        num_iterations=args.num_iterations,
        games_per_iteration=args.games_per_iteration,
        epochs_per_iteration=args.epochs_per_iteration
    )

if __name__ == "__main__":
    main() 