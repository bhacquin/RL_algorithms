# rl_baselines
UncharTECH Baselines for Reinforcement Learning Algorithms

# Installation

```sh
pip install -e .  # at the root of this repo
```

# Notes

DQN on Cartpole only works when using MSE loss. This has to do with the reward structure of Cartpole, where the agent gains +1 for every time step it survives.

# Todo

Relire PPO + benchmarker PPO sur Atari

Refacto TD3

Refacto code "experiments" sur le modèle de DQN/train_cartpole.py

Faire une doc

Réflexion sur l'outil de log

Multiprocess + multiGPU pour DQN

Faire RAINBOW, QR-DQN, et C51


