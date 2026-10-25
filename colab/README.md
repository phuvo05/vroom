# Colab training and GPU benchmarks

The notebooks embed a snapshot of the local source. They do not clone the older
GitHub version. Rebuild them after changing the environment or agent:

```powershell
.venv\Scripts\python.exe colab/build_notebook.py
.venv\Scripts\python.exe colab/build_benchmark.py --gpu T4 --minutes 1
.venv\Scripts\python.exe colab/build_benchmark.py --gpu A100 --minutes 10
```

Upload the desired `.ipynb` to Colab, select the matching GPU, and run all cells.
The setup verifies CUDA and runs the physical-environment regression tests.

## Train a new obstacle-driving model

`VROOM_Challenge_60min.ipynb` starts from new weights on forest/normal/dry. It
allows 58 minutes for training and up to two minutes for evaluation on unseen
map seeds. It saves an atomic `latest.pth` every two minutes, then downloads a
results archive containing `model.pth`, checkpoints, configuration and metrics.
The time budget starts after setup; model quality must be judged from evaluation.

## Compare T4 and A100

`VROOM_Benchmark_T4_1min.ipynb` and `VROOM_Benchmark_A100_10min.ipynb` use the same
forest/normal/dry environment, seed 42, batch size 32, two CPU threads and a
64-step warmup. Each exports `benchmark.json` with agent and optimizer steps per
second, environment/learning/reset times and peak CUDA memory. A benchmark
checkpoint is an experimental artifact, not a validated driving model.

Read the actual compute-unit consumption from Colab's Resources panel. With
multiple assigned runtimes the displayed account rate may be aggregate. Compare
`agent_steps_per_second / compute_units_per_hour` for relative efficiency; do
not substitute an old public price table for the current account rate.

All artifacts initially live on the Colab VM. Download them before releasing the
runtime. The notebooks do not mount Google Drive, buy units or keep sessions alive.
