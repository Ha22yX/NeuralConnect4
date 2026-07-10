<div align="center">
  <h1>NeuralConnect4</h1>
  <p>A Connect Four AI experiment combining neural networks, MCTS, and self-play training.</p>

  <p>
    <a href="README.zh-CN.md">Chinese</a>
    &middot;
    <a href="#quickstart">Quickstart</a>
    &middot;
    <a href="#tech-stack">Tech Stack</a>
  </p>

  <p>
    <img alt="Python: PyTorch" src="https://img.shields.io/badge/Python-PyTorch-3776AB?style=for-the-badge&logo=python&logoColor=white" />
    <img alt="Game AI: MCTS" src="https://img.shields.io/badge/Game%20AI-MCTS-287866?style=for-the-badge" />
    <img alt="Training: self-play" src="https://img.shields.io/badge/Training-self--play-7d73b7?style=for-the-badge" />
  </p>
</div>

<p align="center">
  <img src=".github/assets/readme-hero.svg" alt="NeuralConnect4 overview image" width="100%" />
</p>

## Why This Exists

Connect Four is small enough for fast iteration but still rich enough to study search, policy/value networks, and self-play training loops inspired by AlphaZero-style learning.

## Workflow

- Play or test the local Pygame Connect Four board.
- Represent board states for neural-network evaluation.
- Use MCTS to explore moves with policy/value guidance.
- Generate self-play games and store training examples.
- Train checkpoints and inspect logs/plots.

## Features

- Playable Pygame Connect Four interface.
- ResNet-style policy/value network experiments.
- Neural-guided MCTS and self-play training loop.
- Training logs and visualization hooks.

## Quickstart

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

Start with small training settings for a smoke test, then increase iterations/simulations.

## Tech Stack

| Layer | Technology | Role |
| --- | --- | --- |
| Game | Pygame | Board UI and local play loop. |
| Learning | PyTorch | Policy/value neural network. |
| Search | MCTS | Move selection guided by network evaluation. |
| Analysis | matplotlib, TensorBoard | Training progress inspection. |

## Project Map

```text
game/main.py                 local game entry point
game/board.py                board logic
game/gui.py                  Pygame UI
game/ai_training/            neural net, MCTS, state, training scripts
requirements.txt             Python dependencies
```

## Notes

This is an AI learning project, not a polished released game. Human-vs-checkpoint inference is a natural future step.
