# NeuralConnect4 (神经四子棋)

[English Version](#english-version) | [中文版本](#中文版本)

## English Version

### Introduction
NeuralConnect4 is an advanced AI system for the game of Connect Four, implementing state-of-the-art deep learning and reinforcement learning techniques. The project combines Neural Networks with Monte Carlo Tree Search (MCTS) to create a powerful AI that learns and improves through self-play, similar to the approaches used in AlphaGo and AlphaZero.

### Key Features
- Deep Neural Network with ResNet architecture
- Monte Carlo Tree Search with neural network guidance
- Self-play training with prioritized experience replay
- Dynamic temperature-based exploration
- Adaptive learning rate scheduling
- Multi-threaded game simulation
- CPU optimization for high-performance training
- Comprehensive logging and visualization of training progress

### Technical Architecture
1. **Neural Network**
   - ResNet-based architecture with multiple residual blocks
   - Dual-headed output for policy and value prediction
   - Batch normalization and ReLU activation

2. **MCTS Implementation**
   - UCB-based tree search with neural network evaluation
   - Defensive move recognition and prioritization
   - State caching for improved performance
   - Parallel simulation support

3. **Training System**
   - Self-play data generation with quality scoring
   - Prioritized experience replay buffer
   - Cosine annealing learning rate scheduling
   - Dynamic loss weighting for policy and value heads

### Requirements
```
python >= 3.8
torch >= 1.8.0
numpy >= 1.19.0
psutil >= 5.8.0
matplotlib >= 3.3.0
```

### Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/NeuralConnect4.git
cd NeuralConnect4
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage
1. **Training the AI**:
```bash
python game/ai_training/train_neural.py --batch_size 512 --num_simulations 400 --num_iterations 50
```

2. **Play the current local GUI**:
```bash
python game/main.py
```

Note: `game/main.py` currently starts the local Connect Four GUI. A human-vs-AI
inference mode for loading trained checkpoints is planned but not implemented in
the current repository state.

### Training Parameters
- `batch_size`: Size of training batches (default: 512)
- `num_simulations`: Number of MCTS simulations per move (default: 400)
- `num_iterations`: Number of training iterations (default: 50)
- `games_per_iteration`: Number of self-play games per iteration (default: 100)
- `epochs_per_iteration`: Number of training epochs per iteration (default: 5)

### Project Structure
```
NeuralConnect4/
├── game/
│   ├── ai_training/
│   │   ├── neural_net.py      # Neural network architecture
│   │   ├── neural_mcts.py     # MCTS implementation
│   │   ├── state.py           # Game state management
│   │   └── train_neural.py    # Training script
│   ├── board.py               # Game board implementation
│   ├── gui.py                 # Graphical user interface
│   └── main.py                # Main game entry point
├── models/                     # Saved model checkpoints
├── requirements.txt           # Project dependencies
└── README.md                 # This file
```

---

## 中文版本

### 项目介绍
神经四子棋（NeuralConnect4）是一个基于深度学习和强化学习的高级四子棋AI系统。该项目结合了神经网络和蒙特卡洛树搜索（MCTS）技术，通过自我对弈不断学习和提升，采用了类似AlphaGo和AlphaZero的方法。

### 主要特性
- 基于ResNet架构的深度神经网络
- 神经网络引导的蒙特卡洛树搜索
- 优先经验回放的自我对弈训练
- 动态温度参数探索策略
- 自适应学习率调度
- 多线程游戏模拟
- CPU优化的高性能训练
- 全面的训练过程日志和可视化

### 技术架构
1. **神经网络**
   - 基于ResNet的残差块架构
   - 策略和价值预测的双头输出
   - 批量归一化和ReLU激活

2. **MCTS实现**
   - 基于UCB的神经网络评估树搜索
   - 防守动作识别和优先级
   - 状态缓存优化性能
   - 并行模拟支持

3. **训练系统**
   - 带质量评分的自我对弈数据生成
   - 优先经验回放缓冲区
   - 余弦退火学习率调度
   - 策略和价值头的动态损失权重

### 环境要求
```
python >= 3.8
torch >= 1.8.0
numpy >= 1.19.0
psutil >= 5.8.0
matplotlib >= 3.3.0
```

### 安装步骤
1. 克隆仓库：
```bash
git clone https://github.com/yourusername/NeuralConnect4.git
cd NeuralConnect4
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

### 使用方法
1. **训练AI**：
```bash
python game/ai_training/train_neural.py --batch_size 512 --num_simulations 400 --num_iterations 50
```

2. **与AI对战**：
```bash
python game/main.py
```

### 训练参数
- `batch_size`：训练批次大小（默认：512）
- `num_simulations`：每步MCTS模拟次数（默认：400）
- `num_iterations`：训练迭代次数（默认：50）
- `games_per_iteration`：每次迭代的自我对弈局数（默认：100）
- `epochs_per_iteration`：每次迭代的训练轮数（默认：5）

### 项目结构
```
NeuralConnect4/
├── game/
│   ├── ai_training/
│   │   ├── neural_net.py      # 神经网络架构
│   │   ├── neural_mcts.py     # MCTS实现
│   │   ├── state.py           # 游戏状态管理
│   │   └── train_neural.py    # 训练脚本
│   ├── board.py               # 游戏棋盘实现
│   ├── gui.py                 # 图形用户界面
│   └── main.py                # 主游戏入口
├── models/                     # 保存的模型检查点
├── requirements.txt           # 项目依赖
└── README.md                 # 本文件
``` 
