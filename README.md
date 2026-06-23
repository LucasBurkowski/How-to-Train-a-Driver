# How-to-Train-a-Driver

A reinforcement learning system for generating autonomous driving models for [FIRST Robotics Competition (FRC)](https://www.firstinspires.org/robotics/frc) robots. Uses **PPO** (Proximal Policy Optimization) via [StableBaselines3](https://stable-baselines3.readthedocs.io/) and [PyTorch](https://pytorch.org/), with optional integration into [WPILib simulation](https://docs.wpilib.org/en/stable/docs/software/wpilib-tools/robot-simulation/) via [maple-sim](https://github.com/Shenzhen-Robotics-Alliance/maple-sim).

---

## Overview

The final trained model observes the current robot pose and drive state and outputs high-level path targets for a path-following controller (e.g. PathPlannerLib) to execute. Training uses **self-play**: both full alliances (3 blue + 3 red = 6 robots) are controlled by the same evolving policy, which learns to win FRC matches through competition against itself.

### Reward shaping

| Event | Reward |
|---|---|
| Acquire game piece (intake) | +2.0 |
| Score game piece (× zone point value) | +1.0 × PV |
| Endgame climb | +5.0 |
| Failed score attempt | −0.5 |
| Robot-robot contact | −1.0 |
| Wall contact | −0.3 |
| Time (per step) | −0.01 |

---

## Project Structure

```
src/frc_rl/
├── env/
│   ├── frc_env.py        # Gymnasium environment (6-robot self-play)
│   ├── field.py          # FRC field geometry, scoring zones, intake zones
│   └── reward.py         # Reward computation
├── sim/
│   ├── robot_sim.py      # Standalone Python swerve-drive physics simulator
│   └── wpilib_bridge.py  # NetworkTables 4 bridge to WPILib/maple-sim
├── models/
│   └── policy.py         # Custom two-tower ActorCriticPolicy (PyTorch)
├── training/
│   ├── trainer.py        # PPOTrainer with checkpoint/resume support
│   └── self_play.py      # SelfPlayWrapper + SelfPlayManager
└── inference/
    └── path_generator.py # Converts policy output → Pose2d waypoints

scripts/
├── train.py              # Training entry point
└── evaluate.py           # Evaluation entry point

configs/
└── default_config.yaml   # All hyperparameters and environment settings

tests/                    # pytest unit tests (67 tests)
```

---

## Installation

**Python 3.10+ required.**

```bash
# Clone the repository
git clone https://github.com/LucasBurkowski/How-to-Train-a-Driver.git
cd How-to-Train-a-Driver

# Install core dependencies
pip install -e ".[dev]"

# (Optional) WPILib NetworkTables bridge for live maple-sim integration
pip install -e ".[wpilib]"
```

---

## Training

### Standalone (pure Python simulation — no WPILib required)

```bash
python scripts/train.py
```

### With custom config

```bash
python scripts/train.py --config configs/default_config.yaml
```

### Resume from checkpoint

```bash
python scripts/train.py --checkpoint snapshots/frc_ppo_final.zip
```

### Connected to a live WPILib / maple-sim simulation

Start your WPILib robot simulation (with maple-sim enabled) on the same machine, then:

```bash
python scripts/train.py --bridge-ip localhost
```

The Python trainer communicates with the simulation over **NetworkTables 4** (NT4). Robot state (pose, velocity, game-piece status) is read from NT topics; velocity commands are written back. See `src/frc_rl/sim/wpilib_bridge.py` for the full topic layout.

### Key training flags

| Flag | Default | Description |
|---|---|---|
| `--config` | built-in defaults | Path to YAML config |
| `--checkpoint` | none | Resume from `.zip` |
| `--bridge-ip` | none (standalone) | WPILib sim host IP |
| `--timesteps` | 10 000 000 | Total training steps |
| `--n-envs` | 4 | Parallel environments |

---

## Evaluation

```bash
python scripts/evaluate.py --model snapshots/frc_ppo_final.zip --episodes 20
```

---

## Inference / Deployment

After training, load the model and generate path waypoints for a path-follower:

```python
from frc_rl.inference.path_generator import PathGenerator

gen = PathGenerator("snapshots/frc_ppo_final.zip", n_waypoints=10, lookahead_dt=0.1)

# Build the current robot observation (27-dim vector) from your robot code
obs = build_observation(...)  # np.ndarray, shape (27,)

path = gen.generate_path(
    robot_id=0,
    observation=obs,
    current_x=robot.x,
    current_y=robot.y,
    current_theta=robot.theta,
)

for waypoint in path.waypoints:
    print(waypoint.x, waypoint.y, waypoint.theta_deg)
```

Each `Pose2d` waypoint is `lookahead_dt` seconds apart and can be fed directly into PathPlannerLib or any WPILib trajectory follower.

---

## Observation Space (27 dimensions, per robot)

| Slice | Dims | Description |
|---|---|---|
| Own state | 7 | `x, y, θ, vx, vy, ω, has_piece` (normalised) |
| Nearest game piece | 2 | `Δx, Δy` to nearest active piece |
| Nearest scoring zone | 2 | `Δx, Δy` to nearest own scoring zone |
| Other robots (×5) | 15 | `Δx, Δy, Δθ` for each of the 5 other robots |
| Time remaining | 1 | Normalised match time remaining |

## Action Space (3 dimensions, per robot)

`[vx_norm, vy_norm, omega_norm]` in `[-1, 1]`, field-relative velocity commands.

---

## Self-Play Architecture

Both alliances share the same policy throughout training. `SelfPlayWrapper` decomposes the joint 6-robot observation into per-robot slices and cycles them through the single policy. `SelfPlayManager` saves periodic snapshots (`snapshots/policy_step_N.zip`) so you can compare policy generations or use earlier snapshots as frozen opponents.

```
FRCEnv (6 robots, joint obs/action)
    └── SelfPlayWrapper  ← single-agent interface for SB3 PPO
            └── SB3 PPO (FRCPolicyNetwork)
                    └── FRCFeaturesExtractor (ego tower + context tower)
```

---

## Configuration (`configs/default_config.yaml`)

All training hyperparameters, reward weights, and environment settings are in `configs/default_config.yaml`. Pass `--config path/to/config.yaml` to any script.

---

## Testing

```bash
pytest tests/ -v
```

67 tests covering field geometry, reward computation, physics simulation, the Gymnasium environment, and the self-play wrapper.

---

## maple-sim Integration Notes

[maple-sim](https://github.com/Shenzhen-Robotics-Alliance/maple-sim) provides realistic FRC robot physics (swerve drive kinematics, bumper collisions, game piece intake/scoring) inside WPILib simulation. The Python RL environment connects to it via NT4:

- **Python → sim**: `frc_rl/robot/{id}/cmd/vx|vy|omega` (velocity commands)
- **Sim → Python**: `frc_rl/robot/{id}/state/x|y|theta|vx|vy|omega|has_piece`
- **Match control**: `frc_rl/match/reset`, `frc_rl/match/time_remaining`

When `--bridge-ip` is not set, a standalone Python physics approximation is used so training can run without any Java/WPILib dependencies.
