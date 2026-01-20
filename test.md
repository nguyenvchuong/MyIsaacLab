
## Introduction

**Isaac Lab** is a GPU-accelerated, open-source framework designed to unify and simplify robotics research workflows, such as reinforcement learning, imitation learning, and motion planning. Built on [NVIDIA Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html), a detailed description of Isaac Lab, including installation instructions and documentation, can be found at [https://github.com/isaac-sim/IsaacLab] and in the accompanying [arXiv paper](https://arxiv.org/abs/2511.04831).

## Getting Started

For installation:
```./isaaclab.sh --install  # or "./isaaclab.sh -i"```


## Getting Started

For installation
```./isaaclab.sh --install # or "./isaaclab.sh -i" ```

Example: training a Digit robot using `rsl_rl`:

```python scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Velocity-Flat-Digit-V3 --headless```

For testing the policy:

```python scripts/reinforcement_learning/rsl_rl/play.py --task=Isaac-Velocity-Flat-Digit-V3 --num_env=5```

It will load the most recent logs from `logs/rsl_rl/digit_v3_flat`

