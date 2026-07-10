<div align="center">
  <h1>NeuralConnect4</h1>
  <p>A Connect Four AI experiment combining neural networks, Monte Carlo Tree Search, and self-play training.</p>

  <p>
    <a href="README.zh-CN.md">Chinese</a>
    &middot;
    <a href="#quickstart">Quickstart</a>
    &middot;
    <a href="#features">Features</a>
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

## Overview

Connect Four is small enough for fast iteration but still rich enough to study policy/value networks, search, and self-play loops.

This repo explores an AlphaZero-style learning path in a readable Python project: board state encoding, MCTS, neural evaluation, training logs, and a playable Pygame interface.

## Features

- Playable local Connect Four board built with Pygame.
- Board/state representation for search and model training.
- MCTS and neural-guided MCTS experiments.
- Policy/value neural network training through self-play data.
- Training logs and plotting hooks for progress inspection.

## How It Works

1. Run the local game to verify board logic and UI.
2. Use game states as inputs for search and neural evaluation.
3. Generate self-play examples with MCTS-guided decisions.
4. Train model checkpoints and inspect performance over time.

## Quickstart

Run the project locally with the commands below.

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

Start with small iteration/simulation counts for a smoke test, then scale training settings gradually.

## Configuration

| Item | Purpose |
| --- | --- |
| Training iterations | Increase only after the end-to-end loop runs correctly. |
| MCTS simulations | Higher values improve search quality but slow self-play. |
| Checkpoints | Keep model outputs separate from source when running longer experiments. |
| Device | PyTorch can use CPU by default; configure GPU only if available. |

## Tech Stack

| Layer | Technology | Role |
| --- | --- | --- |
| Game | Pygame | Board UI and local play loop. |
| Learning | PyTorch | Policy/value neural network. |
| Search | MCTS | Move selection guided by evaluation. |
| Analysis | matplotlib, TensorBoard | Training progress inspection. |

## Project Layout

```text
game/main.py                 local game entry point
game/board.py                board logic
game/gui.py                  Pygame UI
game/ai_training/            neural net, MCTS, state, training scripts
requirements.txt             Python dependencies
training.log                 sample training log
```

## Status

AI learning project, not a polished game release. Human-vs-checkpoint play and stronger evaluation tooling are natural next steps.

## License

No project-wide open-source license has been declared yet.
