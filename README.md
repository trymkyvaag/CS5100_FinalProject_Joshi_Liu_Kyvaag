
# Duo Ball

This project implements a custom 2v2 soccer game environment using PyGame and trains reinforcement learning agents (the blue team) using Proximal Policy Optimization (PPO) via Stable-Baselines3. The environment is Gym-compatible and built for flexible experimentation with reward shaping, curriculum learning, and hyperparameter tuning.

## Project Highlights

- Trained PPO agents using heuristic-based rewards
- Fully interactive 2v2 soccer simulation with PyGame rendering
- Replay module to visualize trained agents
- Reward logging, evaluation, and tuning via Optuna
- Curriculum learning and baselines explored

---

## Project Structure

```
```

---

## Requirements

- Python 3.8+

Install dependencies via:

```bash
pip install -r requirements.txt
```

---

## Running the Project

### 1. Train the Agent

```bash
python main.py
```

- Trains PPO agents using the custom environment.
- Saves models and reward stats periodically.

### 2. Replay Trained Agent

```bash
python replay.py
```

- Loads `soccer_agent_ppo.zip` and visualizes 5 evaluation episodes.
- Evaluates agent behavior against baseline red team.

---

## Logging & Evaluation

- Reward metrics (mean, median, std, etc.) are logged to `reward_stats.csv`
- Visualizations (e.g., reward curves, explained variance) can be generated for reporting.
- Evaluation mean rewards logged periodically during training.

---

## Key Features

- Multi-agent discrete control using `MultiDiscrete` action space
- Custom Gym-compatible environment
- Dense 15-D observation vector with player and ball state
- Manual and Optuna-based hyperparameter tuning
- Support for curriculum learning (configurable red team behavior)

---

## Reference Papers

- Kurach et al., [Google Research Football: A Novel RL Environment](https://arxiv.org/abs/1907.11180)
- Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- Matiisen et al., [Teacher-Student Curriculum Learning](https://arxiv.org/abs/1707.00183)
  
---

## Credits

Developed as part of a course project on 5100 Foundations of AI. Built and maintained by Sanjiv, Trym and Eason.

---

## TODOs / Future Work

- Self-play and red team training
- Imitation learning from expert trajectories
- Transition to continuous action space
- Curriculum learning enhancements

---
