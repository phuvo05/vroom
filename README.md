# Double DQN for Car Racing

A reinforcement learning project implementing and comparing **Tabular Q-Learning**, **Deep Q-Network (DQN)**, and **Double Deep Q-Network (Double DQN)** on the Gymnasium Car Racing environment.

The primary objective is to investigate how progressively more advanced value-based reinforcement learning methods perform on a visual control task. The repository includes a complete training pipeline, a custom environment wrapper, experiment logging, checkpoint management, and a standalone demo application. Only the Double DQN agent is included in this repository; the Q-Learning and DQN baselines are described in the report but their code and logs are not part of it.

---

## Features

- Double Deep Q-Network (Double DQN)
- Vanilla Deep Q-Network (DQN) baseline
- Tabular Q-Learning baseline
- CNN-based Q-network using PyTorch
- Experience replay and target network
- Custom Car Racing environment wrapper
- Frame preprocessing, stacking, and skipping
- Automatic checkpointing and logging
- Training, evaluation, and resume modes
- Standalone demo application

---

## Repository Structure

```text
.
├── dqn/
│   ├── agents/
│   ├── configs/
│   ├── environment/
│   ├── logs/
│   ├── models/
│   ├── utils/
│   ├── train.py
│   └── demo_ui.py
└── README.md
```

---

## Algorithms

| Algorithm | Purpose |
|-----------|---------|
| Q-Learning | Classical tabular RL baseline |
| DQN | Deep RL baseline |
| Double DQN | Main implementation |

---

## Project Architecture

```text
Gymnasium CarRacing-v3
        │
        ▼
Custom Environment Wrapper
        │
        ├── Observation preprocessing
        ├── Frame stacking
        ├── Frame skipping
        └── Reward processing
        │
        ▼
Double DQN Agent
        │
        ▼
CNN Q-Network
        │
        ▼
Discrete Action Selection
```

---

## Installation

```bash
git clone https://github.com/yomama2311/vroom.git
cd vroom/
pip install -r dqn/requirements.txt
```

---

## Usage

### Train

```bash
python dqn/train.py
```

### Resume Training

```bash
python dqn/train.py --resume <checkpoint>
```

### Evaluate

```bash
python dqn/train.py --eval <checkpoint>
```

### Launch Demo

```bash
python dqn/demo_ui.py
```

---

## Training Outputs

During training, the project generates:

- Model checkpoints
- Best-performing model
- Training logs
- Learning curves and visualizations

---

## Results

The training log and curves of the Double DQN run are in `dqn/logs/`. The full comparison with Q-Learning and DQN is reported in `report-paper.pdf`.

---

## Authors

- Vo Thien Phu
- Nguyen Truong Phuc Thinh

