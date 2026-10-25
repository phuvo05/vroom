"""Package the current obstacle project as a self-contained Kaggle GPU notebook."""
import ast
import base64
import io
import json
from pathlib import Path
import textwrap
import zipfile

ROOT = Path(__file__).resolve().parents[1]
buffer = io.BytesIO()
with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
    for folder in ("dqn", "tests"):
        for path in sorted((ROOT / folder).rglob("*.py")):
            z.write(path, path.relative_to(ROOT).as_posix())
payload = base64.b64encode(buffer.getvalue()).decode("ascii")


def cell(kind, source):
    source = textwrap.dedent(source).strip() + "\n"
    result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        ast.parse(source)
        result.update(execution_count=None, outputs=[])
    return result


setup = '''
import base64, io, json, os, subprocess, sys, zipfile
from pathlib import Path
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["PYTHONUNBUFFERED"] = "1"
import torch
assert torch.cuda.is_available(), "Enable a Kaggle GPU (P100 or T4); this notebook does not use TPU."
print("GPU:", torch.cuda.get_device_name(0), "| PyTorch:", torch.__version__, flush=True)
print("This DQN uses ONE GPU even if Kaggle assigns T4 x2.")
PROJECT = Path("/kaggle/working/vroom_challenge")
PROJECT.mkdir(exist_ok=True)
ARCHIVE = "__PAYLOAD__"
with zipfile.ZipFile(io.BytesIO(base64.b64decode(ARCHIVE))) as z:
    z.extractall(PROJECT)
# Pin the tested release: a preinstalled older Gymnasium may otherwise satisfy
# >=1.0 and request box2d-py, triggering a fragile source build on Kaggle Python 3.13.
# Box2D 2.3.10 and pygame-ce ship Linux cp313 wheels; no SWIG/compiler is needed.
subprocess.check_call([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
    "--only-binary=Box2D,pygame-ce", "gymnasium==1.4.0", "Box2D==2.3.10",
    "pygame-ce>=2.5", "opencv-python-headless>=4.7", "matplotlib>=3.7", "Pillow>=10"])
subprocess.check_call([sys.executable, "-c",
    "import sys, gymnasium, Box2D, pygame; print('Python', sys.version.split()[0], "
    "'Gymnasium', gymnasium.__version__, 'Box2D', Box2D.__version__, 'pygame', pygame.version.ver)"])
subprocess.check_call([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], cwd=PROJECT)
'''.replace("__PAYLOAD__", payload)
settings = '''
# Attach your private Kaggle dataset with this slug; it must contain best.pth.
# Only scan this project's dataset, never unrelated .pth files.
CHECKPOINT_DATASET_SLUG = "vroom-obstacle-checkpoint"
WARM_START_CHECKPOINT = ""  # Set an exact /kaggle/input/.../best.pth path if needed.
START_FROM_SCRATCH = False  # Change to True only to intentionally discard previous learning.
MAX_MINUTES = None  # None disables the notebook's wall-clock cap; 480 would allow 8 hours.
NUM_EPISODES = 1500  # Warm start with the new speed-penalty reward; evaluate before going longer.
SCENARIO, DIFFICULTY, WEATHER, SEED = "forest", "normal", "dry", 42
if not WARM_START_CHECKPOINT:
    candidates = [p for p in Path("/kaggle/input").rglob("best.pth")
                  if CHECKPOINT_DATASET_SLUG in p.parts]
    if len(candidates) == 1:
        WARM_START_CHECKPOINT = str(candidates[0])
    elif len(candidates) > 1:
        raise RuntimeError("Several project checkpoints found; set WARM_START_CHECKPOINT explicitly.")
if not START_FROM_SCRATCH:
    assert WARM_START_CHECKPOINT and Path(WARM_START_CHECKPOINT).is_file(), (
        "Attach your private vroom-obstacle-checkpoint dataset containing best.pth, "
        "or explicitly set START_FROM_SCRATCH=True to start a new model.")
print("Warm start:", WARM_START_CHECKPOINT if not START_FROM_SCRATCH else "NEW WEIGHTS", flush=True)
'''
training = '''
import datetime, shutil, time
assert MAX_MINUTES is None or MAX_MINUTES > 2, "Use None or a budget greater than 2 minutes."
assert isinstance(NUM_EPISODES, int) and NUM_EPISODES > 0, "NUM_EPISODES must be positive."
RUN_DIR = PROJECT / "runs" / datetime.datetime.now().strftime("forest-%Y%m%d-%H%M%S")
BUDGET_START = time.monotonic()
command = [sys.executable, "-u", "dqn/train.py", "--scenario", SCENARIO,
    "--difficulty", DIFFICULTY, "--weather", WEATHER, "--seed", str(SEED),
    "--episodes", str(NUM_EPISODES),
    "--save-seconds", "120", "--output", str(RUN_DIR)]
if MAX_MINUTES is not None:
    command += ["--max-minutes", str(MAX_MINUTES-2)]
if not START_FROM_SCRATCH:
    command += ["--warm-start", WARM_START_CHECKPOINT]
print("START:", " ".join(command), flush=True)
with (PROJECT / "training-console.log").open("w") as log:
    process = subprocess.Popen(command, cwd=PROJECT, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        returncode = process.wait()
    except KeyboardInterrupt:
        process.send_signal(2)
        process.wait(timeout=30)
        raise
if returncode:
    raise RuntimeError(f"Training failed ({returncode}); inspect the saved console log.")
print((RUN_DIR / "summary.json").read_text(), flush=True)
checkpoint_dir = RUN_DIR / "checkpoints"
latest = checkpoint_dir / "latest.pth"
best = checkpoint_dir / "best.pth"
MODEL = checkpoint_dir / "model.pth"
shutil.copy2(best if best.exists() else latest, MODEL)
remaining = 120 if MAX_MINUTES is None else MAX_MINUTES*60 - (time.monotonic()-BUDGET_START)
if remaining > 5:
    evaluation = [sys.executable, "-u", "dqn/train.py", "--eval", str(MODEL),
        "--scenario", SCENARIO, "--difficulty", DIFFICULTY, "--weather", WEATHER,
        "--headless", "--seed", "100042", "--eval-episodes", "10"]
    try:
        with (RUN_DIR / "evaluation.txt").open("w") as out:
            subprocess.run(evaluation, cwd=PROJECT, stdout=out, stderr=subprocess.STDOUT,
                text=True, check=True, timeout=max(1, remaining))
        print((RUN_DIR / "evaluation.txt").read_text(), flush=True)
    except subprocess.TimeoutExpired:
        print("Evaluation stopped at budget limit; partial log retained.", flush=True)
'''
export = '''
from IPython.display import FileLink, display
shutil.copy2(PROJECT / "training-console.log", RUN_DIR / "training-console.log")
EXPORT_PATH = shutil.make_archive(str(RUN_DIR) + "-results", "zip", RUN_DIR)
print("Results saved under /kaggle/working; download or Save Version with outputs.")
display(FileLink(EXPORT_PATH))
print("Demo checkpoint:", MODEL)
'''
notebook = {"nbformat":4,"nbformat_minor":5,"metadata":{
    "kernelspec":{"display_name":"Python 3","name":"python3"},
    "language_info":{"name":"python"}},"cells":[
    cell("markdown", '''
# VROOM — Kaggle GPU training

Self-contained snapshot of the obstacle-driving project. Use a **GPU**, not TPU.
Select an available P100 or T4 GPU, enable Internet for package installation,
and run the cells in order. The code uses one GPU, even on a T4 x2 allocation.
Setup pins Gymnasium 1.4.0 and installs prebuilt Box2D 2.3.10 wheels, including
Python 3.13 support, instead of compiling the older box2d-py package.

This version trains with a **speed penalty**: going faster than the upcoming bend allows
costs reward (forest/city/desert only; `classic` keeps the original reward). The replay
buffer is 100,000 transitions stored as uint8 (about 5.6 GB at capacity).

Upload the model you want to improve (`best.pth`) as a **private** Kaggle dataset named
`vroom-obstacle-checkpoint` and attach it to this notebook. Training **warm-starts** from
its weights: the optimizer and replay buffer are fresh and epsilon restarts at 0.5, because
the reward changed. The settings cell stops clearly if `best.pth` is missing. Only set
`START_FROM_SCRATCH=True` if you intentionally want new weights.

By default `MAX_MINUTES=None` and `NUM_EPISODES=1500`: training stops at the episode
target, a manual interrupt or the hosting platform's own session limit. Set e.g.
`MAX_MINUTES=480` for an eight-hour total budget instead. Evaluation runs 10 unseen maps
(seeds from 100042) and is limited to two minutes. Checkpoints save every 120 seconds.

Outputs are kept in `/kaggle/working`. Download them or save a notebook version
with outputs before releasing the runtime. A short run does not guarantee mastery.
'''),cell("code",setup),cell("code",settings),cell("code",training),cell("code",export)]}
path = ROOT / "kaggle/VROOM_Kaggle_GPU.ipynb"
path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding="utf-8")
print(path, path.stat().st_size,"bytes")
