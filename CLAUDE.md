# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Thesis project comparing LSTM and Mamba SSM architectures for learning drone control policies via supervised imitation of a PID controller. Built on top of [gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones), a PyBullet-based quadrotor simulation framework.

## Setup

```sh
cd gym-pybullet-drones
pip install -e .
```

Requires Python 3.10+. Key dependencies: `pybullet`, `gymnasium`, `stable-baselines3`, `torch`, `mamba-ssm`, `numpy`, `scipy`.

## Common Commands

**Run tests:**
```sh
cd gym-pybullet-drones
pytest tests/
```

**Generate training data (PID demonstrations):**
```sh
python generar_datos_CF2P.py    # CF2P drone, DSL PID controller
python generar_datos_CF2X.py    # CF2X drone
python generar_datos_RACE.py    # RACE drone, MRAC controller
python generate_states.py       # Compute normalization statistics after data generation
```

**Train / test deep learning models:**
```sh
python lstm_simulador.py        # LSTM controller
python mamba_simulador.py       # Mamba SSM controller
python test_mamba.py            # Quick Mamba model test
```

**Run RL training example:**
```sh
python gym_pybullet_drones/examples/learn.py --multiagent false
```

**Run PID simulation example:**
```sh
python gym_pybullet_drones/examples/pid.py
```

## Architecture

### gym-pybullet-drones library (`gym_pybullet_drones/`)

- **`envs/BaseAviary.py`** — Core PyBullet integration: physics stepping, drone state extraction, camera rendering, and URDF loading. All environment classes inherit from this.
- **`envs/CtrlAviary.py`** — Used for data generation and PID control demos; wraps BaseAviary with action/observation spaces suited for external controllers.
- **`envs/BaseRLAviary.py` → `HoverAviary` / `MultiHoverAviary`** — RL-specific environments; used with Stable-Baselines3.
- **`control/DSLPIDControl.py`** — PID controller (DSL UTIAS reference implementation) that the thesis uses as the "teacher" policy to imitate.
- **`control/MRAC.py`** — Model Reference Adaptive Control, used for RACE drone data generation.
- **`utils/enums.py`** — All enumerations: `DroneModel` (CF2X, CF2P, RACE), `Physics`, `ActionType`, `ObservationType`.
- **`utils/Logger.py`** — Episode data logger used during data generation.

### Thesis-specific code (repo root)

- **`generar_datos_*.py`** — Run 400 episodes of PID/MRAC control, log 18-dimensional state vectors + 4 motor outputs → save to `results/`.
  - Input features: position (3), velocity (3), attitude RPY (3), angular velocity (3), target position (3), position error (3) = 18 signals.
- **`generate_states.py`** — Computes per-feature mean/std over the generated dataset for normalization.
- **`lstm_simulador.py`** — Loads trained LSTM (2-layer, 128 hidden units), closes the loop in PyBullet simulation with 50-step input window.
- **`mamba_simulador.py`** — Same closed-loop test but with Mamba SSM (d_model=128, 3 layers, ~99K parameters vs LSTM's 216K).
- **`Modelos_deep/tesis_drone_entrenamiento_LSTM.ipynb`** — Jupyter notebook for model training, hyperparameter search, and loss visualization.
- **`results/`** — Generated CSV datasets and saved model checkpoints (`.pt` files).
- **`Papers/`** — Reference papers and `Referencias.bib` bibliography.

### Data flow

```
generar_datos_*.py  →  results/datos_*.csv
                             ↓
                    generate_states.py  →  results/estados_normalizados.csv
                             ↓
              Jupyter notebook (training)  →  results/*.pt  (model weights)
                             ↓
              lstm_simulador.py / mamba_simulador.py  (closed-loop eval in PyBullet)
```
