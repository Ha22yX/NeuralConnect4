<div align="center">
  <h1>NeuralConnect4</h1>
  <p>一个结合神经网络、蒙特卡洛树搜索和自博弈训练的 Connect Four AI 实验。</p>

  <p>
    <a href="README.md">English</a>
    &middot;
    <a href="#快速开始">快速开始</a>
    &middot;
    <a href="#核心能力">核心能力</a>
    &middot;
    <a href="#技术栈">技术栈</a>
  </p>

  <p>
    <img alt="Python: PyTorch" src="https://img.shields.io/badge/Python-PyTorch-3776AB?style=for-the-badge&logo=python&logoColor=white" />
    <img alt="Game AI: MCTS" src="https://img.shields.io/badge/Game%20AI-MCTS-287866?style=for-the-badge" />
    <img alt="Training: self-play" src="https://img.shields.io/badge/Training-self--play-7d73b7?style=for-the-badge" />
  </p>
</div>

<p align="center">
  <img src=".github/assets/readme-hero.svg" alt="NeuralConnect4 项目概览图" width="100%" />
</p>

## 项目概览

Connect Four 足够小，便于快速迭代；同时又足够有策略性，可以研究策略/价值网络、搜索和自博弈训练。

这个仓库用可读的 Python 项目实现一条 AlphaZero 风格学习路线：棋盘状态编码、MCTS、神经网络评估、训练日志和可玩的 Pygame 界面。

## 核心能力

- 基于 Pygame 的本地 Connect Four 棋盘。
- 用于搜索和模型训练的棋盘/状态表示。
- MCTS 和神经网络引导 MCTS 实验。
- 通过自博弈数据训练策略/价值网络。
- 训练日志和绘图入口，便于观察进度。

## 工作方式

1. 先运行本地游戏验证棋盘逻辑和 UI。
2. 把棋盘状态作为搜索和神经网络评估输入。
3. 用 MCTS 引导决策生成自博弈样本。
4. 训练模型检查点，并观察长期表现。

## 快速开始

可以用下面的命令在本地运行项目。

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

建议先用较小的迭代次数和模拟次数做冒烟测试，再逐步扩大训练规模。

## 配置项

| 项目 | 作用 |
| --- | --- |
| 训练轮数 | 确认完整流程可跑通后再增加。 |
| MCTS 模拟次数 | 越高搜索质量越好，但自博弈会更慢。 |
| 检查点 | 长时间训练时建议把模型输出与源码分开保存。 |
| 设备 | PyTorch 默认可用 CPU；有 GPU 时再额外配置。 |

## 技术栈

| 层级 | 技术 | 作用 |
| --- | --- | --- |
| 游戏 | Pygame | 棋盘 UI 和本地游戏循环。 |
| 学习 | PyTorch | 策略/价值神经网络。 |
| 搜索 | MCTS | 基于评估的落子选择。 |
| 分析 | matplotlib, TensorBoard | 训练进度观察。 |

## 项目结构

```text
game/main.py                 本地游戏入口
game/board.py                棋盘逻辑
game/gui.py                  Pygame UI
game/ai_training/            神经网络、MCTS、状态和训练脚本
requirements.txt             Python 依赖
training.log                 示例训练日志
```

## 项目状态

这是 AI 学习项目，不是完整游戏发行版。人类对战检查点和更完整评估工具是自然的后续方向。

## 许可证

当前仓库尚未声明项目级开源许可证；公开复用或分发前建议先补充 License。
