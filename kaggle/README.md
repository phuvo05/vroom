# Kaggle GPU training

Import `VROOM_Kaggle_GPU.ipynb` into a new Kaggle notebook. It embeds the current
project (environment, agent, tests); cloning the older GitHub repository is not sufficient.
Rebuild after editing the project with `.venv/Scripts/python.exe kaggle/build_notebook.py`.

## What this version changes

- **Speed penalty:** in `forest` / `city` / `desert`, going faster than the upcoming bend
  allows costs reward (`speed_penalty` in `dqn/environment/challenge_racing.py`). The first
  model drove ~220 km/h even where the road turns more than 1 rad. `classic` keeps the
  original reward, so old weights stay comparable there.
- **Replay buffer:** 100,000 transitions (was 10,000), frames stored as uint8
  (lossless, about 5.6 GB at capacity).
- **Warm start:** the notebook trains from `best.pth` weights with `--warm-start`. The
  optimizer and replay buffer are fresh and epsilon restarts at 0.5, because the reward changed.
- `NUM_EPISODES=1500`; the final evaluation uses 10 unseen maps (seeds from 100042).

## Steps

1. Upload `best.pth` (the single file, not the unpacked folder) as a **private** Kaggle
   dataset with slug `vroom-obstacle-checkpoint`. Attach it to the notebook.
2. Choose an available GPU (P100 or T4) and enable Internet for dependencies. The agent
   uses a single CUDA GPU, not TPU.
3. Run the cells in order. The settings cell finds `best.pth` in that dataset, or accepts an
   explicit `WARM_START_CHECKPOINT` path. It refuses to silently start fresh; set
   `START_FROM_SCRATCH=True` only to discard previous learning.
4. Download the results archive from `/kaggle/working`, or save a notebook version with
   outputs, before releasing the session. Use `checkpoints/best.pth` from the run for the demo.

`MAX_MINUTES=None` omits `--max-minutes`; set e.g. `480` for an eight-hour budget. The
platform's own runtime/quota limits still apply. Evaluation is capped at two minutes.
Checkpoints are written every 120 seconds.

The notebook's Python cells and embedded sources were validated locally; actual Kaggle
execution and accelerator availability still need to be verified.

## Installation note

Setup pins `gymnasium==1.4.0`, `Box2D==2.3.10` and uses binary wheels for Box2D and
pygame-ce, which avoids building the older box2d-py package through SWIG on Kaggle's
Python 3.13.
