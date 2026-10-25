"""Build a self-contained Colab notebook from the current working sources."""
import ast
import base64
import io
import json
from pathlib import Path
import textwrap
import zipfile

ROOT = Path(__file__).resolve().parents[1]
archive = io.BytesIO()
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
    for folder in ("dqn", "tests"):
        for path in sorted((ROOT / folder).rglob("*.py")):
            z.write(path, path.relative_to(ROOT).as_posix())
    z.write(ROOT / "dqn/requirements.txt", "dqn/requirements.txt")
payload = base64.b64encode(archive.getvalue()).decode("ascii")


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(source).strip().splitlines(keepends=True)}


def code(source):
    source = textwrap.dedent(source).strip() + "\n"
    ast.parse(source)
    return {"cell_type": "code", "execution_count": None, "outputs": [], "metadata": {}, "source": source.splitlines(keepends=True)}

setup = '''
import base64, io, os, sys, subprocess, zipfile
from pathlib import Path
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["PYTHONUNBUFFERED"] = "1"
import torch
assert torch.cuda.is_available(), "Select Runtime > Change runtime type > T4 GPU before running."
print("GPU:", torch.cuda.get_device_name(0), "| PyTorch:", torch.__version__, flush=True)
PROJECT = Path("/content/vroom_challenge")
PROJECT.mkdir(exist_ok=True)
ARCHIVE = "__PAYLOAD__"
with zipfile.ZipFile(io.BytesIO(base64.b64decode(ARCHIVE))) as z:
    z.extractall(PROJECT)
subprocess.check_call([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
    "--only-binary=Box2D,pygame-ce", "gymnasium==1.4.0", "Box2D==2.3.10",
    "pygame-ce>=2.5", "opencv-python-headless>=4.7", "matplotlib>=3.7", "Pillow>=10"])
subprocess.check_call([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], cwd=PROJECT)
print("Source and physical environment checks ready.", flush=True)
'''.replace("__PAYLOAD__", payload)

training = '''
# This is a NEW model. No weights or optimizer state from the old track.
MAX_MINUTES = 60
SCENARIO = "forest"
DIFFICULTY = "normal"
WEATHER = "dry"
SEED = 42
import datetime, json, shutil, time
RUN_DIR = PROJECT / "runs" / datetime.datetime.now().strftime("forest-%Y%m%d-%H%M%S")
BUDGET_STARTED = time.monotonic()
# Reserve two minutes for evaluation; keep the whole trial within 60 minutes.
TRAIN_MINUTES = max(0.1, MAX_MINUTES - 2)
train_command = [sys.executable, "-u", "dqn/train.py", "--scenario", SCENARIO,
    "--difficulty", DIFFICULTY, "--weather", WEATHER, "--seed", str(SEED),
    "--episodes", "100000", "--max-minutes", str(TRAIN_MINUTES),
    "--save-seconds", "120", "--output", str(RUN_DIR)]
print("STARTING NEW TRAINING:", " ".join(train_command), flush=True)
with (PROJECT / "training-console.log").open("w") as log:
    process = subprocess.Popen(train_command, cwd=PROJECT, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        result = process.wait()
    except KeyboardInterrupt:
        process.send_signal(2)
        process.wait(timeout=30)
        raise
if result:
    raise RuntimeError(f"Training process exited with code {result}; inspect training-console.log")
print((RUN_DIR / "summary.json").read_text(), flush=True)
MODEL = RUN_DIR / "checkpoints" / "model.pth"
best = RUN_DIR / "checkpoints" / "best.pth"
latest = RUN_DIR / "checkpoints" / "latest.pth"
shutil.copy2(best if best.exists() else latest, MODEL)
# Unseen map seeds. The notebook never claims a higher score without this check.
remaining = MAX_MINUTES * 60 - (time.monotonic() - BUDGET_STARTED)
if remaining > 5:
    evaluation = [sys.executable, "-u", "dqn/train.py", "--eval", str(MODEL),
        "--scenario", SCENARIO, "--difficulty", DIFFICULTY, "--weather", WEATHER,
        "--headless", "--seed", "100042", "--eval-episodes", "5"]
    try:
        with (RUN_DIR / "evaluation.txt").open("w") as out:
            subprocess.run(evaluation, cwd=PROJECT, stdout=out, stderr=subprocess.STDOUT,
                           text=True, timeout=max(1, remaining), check=True)
        print((RUN_DIR / "evaluation.txt").read_text(), flush=True)
    except subprocess.TimeoutExpired:
        print("Evaluation stopped at the trial time limit; partial log saved.", flush=True)
print("Model available:", MODEL, flush=True)
'''

export = '''
import shutil
from google.colab import files
shutil.copy2(PROJECT / "training-console.log", RUN_DIR / "training-console.log")
EXPORT_PATH = shutil.make_archive(str(RUN_DIR) + "-results", "zip", RUN_DIR)
print("Downloading model.pth, latest.pth, logs, metrics, configuration and evaluation:", EXPORT_PATH)
files.download(EXPORT_PATH)
'''

cells = [markdown('''
# VROOM — train a new obstacle-driving model (60-minute trial)

Self-contained snapshot of the modified project, not the old GitHub version.
Start with **T4 GPU** and use **Runtime → Run all**. The first trial trains a fresh
Dueling Double DQN on **forest / normal / dry**: solid barriers, parked cars and crates.

The budget is 58 minutes of training plus up to 2 minutes of evaluation on unseen map
seeds. Dependencies and environment checks happen first. A complete `latest.pth` is
saved every 120 seconds; the final download contains `model.pth`, optimizer checkpoints,
reward/collision/progress metrics and environment settings. A one-hour trial does not
guarantee the environment has been solved.

Keep this tab open for the automatic results download. Checkpoints reside on the Colab
VM until downloaded; they are not automatically saved to Google Drive.
'''), code(setup), code(training), code(export), markdown('''
## Use the trained model locally
Extract the results archive into a new folder. In the VROOM demo, select
**Chọn model .pth**, open `checkpoints/model.pth`, and choose **Rừng cây / Thường / dry**.
`latest.pth` retains optimizer and exploration state for a later `--resume` run.

After the results download has completed, disconnect this notebook's runtime to stop
using the GPU. No keep-alive scripts or additional compute purchases are used.
''')]
notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {
    "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
    "kernelspec": {"display_name": "Python 3", "name": "python3"},
    "language_info": {"name": "python"}}, "cells": cells}
path = ROOT / "colab/VROOM_Challenge_60min.ipynb"
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(path, path.stat().st_size, "bytes")
