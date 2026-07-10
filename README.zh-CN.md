<div align="center">
  <h1>NeuralConnect4</h1>
  <p>一个结合神经网络、MCTS 和自我对弈训练的四子棋 AI 实验项目。</p>

  <p>
    <a href="README.md">English</a>
    &middot;
    <a href="#快速开始">快速开始</a>
    &middot;
    <a href="#技术栈">技术栈</a>
  </p>

  <p>
    <img alt="Python: PyTorch" src="https://img.shields.io/badge/Python-PyTorch-3776AB?style=for-the-badge&logo=python&logoColor=white" />
    <img alt="Game AI: MCTS" src="https://img.shields.io/badge/Game%20AI-MCTS-287866?style=for-the-badge" />
    <img alt="Training: self-play" src="https://img.shields.io/badge/Training-self-play-7d73b7?style=for-the-badge" />
  </p>
</div>

<p align="center">
  <img src=".github/assets/readme-hero.svg" alt="NeuralConnect4 项目概览图" width="100%" />
</p>

## 项目价值

四子棋足够小，便于快速实验；同时又足够有策略性，适合研究搜索、策略/价值网络和 AlphaZero 风格自我对弈。

## 快速开始

```bash
git clone https://github.com/Ha22yX/NeuralConnect4.git
cd NeuralConnect4
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python game/main.py

cd game/ai_training
python train_neural.py --num_iterations 5 --num_simulations 100
```

GUI 会启动本地四子棋棋盘。建议先用小参数测试，再增加训练规模。

## 核心功能

- 基于 Pygame 的本地四子棋界面。
- ResNet 风格策略/价值网络实验。
- 神经网络引导的 MCTS 和自我对弈训练循环。
- 包含训练日志和可视化辅助，用于复盘迭代过程。

## 技术栈

| Layer | Technology | Role |
| --- | --- | --- |
| 游戏 | Pygame | 棋盘界面和本地对局循环。 |
| 学习 | PyTorch | 策略/价值神经网络。 |
| 搜索 | MCTS | 由网络评估引导的走子选择。 |
| 分析 | matplotlib, TensorBoard | 训练过程查看。 |


## 项目说明

这是学习与实验仓库，不是完整游戏产品。模型加载稳定后，人机对战推理模式是自然的下一步。
