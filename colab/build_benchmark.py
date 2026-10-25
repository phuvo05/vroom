"""Build matched Colab benchmark notebooks from the current project."""
import argparse
import json
from pathlib import Path

from build_notebook import code, markdown, setup

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--gpu", choices=["A100", "T4"], default="A100")
p.add_argument("--minutes", type=float, default=10)
args = p.parse_args()
assert 0 < args.minutes <= 10
benchmark = '''
import datetime, json, shutil
EXPECTED_GPU = "__GPU__"
BENCHMARK_MINUTES = __MINUTES__
actual_gpu = torch.cuda.get_device_name(0)
assert EXPECTED_GPU in actual_gpu, f"Expected {EXPECTED_GPU}, got {actual_gpu}; change runtime type first."
RUN_DIR = PROJECT / "runs" / datetime.datetime.now().strftime("benchmark-__GPU__-%Y%m%d-%H%M%S")
command = [sys.executable, "-u", "dqn/benchmark.py", "--minutes", str(BENCHMARK_MINUTES),
    "--seed", "42", "--warmup-steps", "64", "--output", str(RUN_DIR)]
print("GPU:", actual_gpu, "| command:", " ".join(command), flush=True)
with (PROJECT / "benchmark-console.log").open("w") as log:
    process = subprocess.Popen(command, cwd=PROJECT, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        returncode = process.wait()
    except KeyboardInterrupt:
        process.terminate()
        process.wait(timeout=30)
        raise
if returncode:
    raise RuntimeError(f"Benchmark failed with code {returncode}; inspect the output.")
REPORT = json.loads((RUN_DIR / "benchmark.json").read_text())
print("MEASURED:", REPORT["agent_steps_per_second"], "agent steps/s;",
    REPORT["optimizer_steps_per_second"], "optimizer updates/s", flush=True)
# CU rate is account-specific: read it from Colab's Resources panel.
# Do not substitute an old public price table for the actual displayed rate.
'''.replace("__GPU__", args.gpu).replace("__MINUTES__", str(args.minutes))
export = '''
from google.colab import files
shutil.copy2(PROJECT / "benchmark-console.log", RUN_DIR / "benchmark-console.log")
EXPORT_PATH = shutil.make_archive(str(RUN_DIR) + "-results", "zip", RUN_DIR)
print("Download benchmark JSON and trial checkpoint:", EXPORT_PATH, flush=True)
files.download(EXPORT_PATH)
'''
cells = [markdown(f'''
# VROOM — {args.gpu} benchmark ({args.minutes:g} minutes)

This notebook embeds the current physical obstacle environment. It measures the complete
**forest / normal / dry, batch32, one environment** training pipeline, with a 64-step
warmup, matching the same settings used by the T4 reference benchmark.

Select **{args.gpu} GPU**, then **Run all**. Setup and tests run before the timed benchmark.
The benchmark itself stops within {args.minutes:g} minutes (plus one in-flight step and final save).
Record the actual **compute units/hour** from Colab's Resources panel for the cost comparison.

The final download contains `benchmark.json`, `benchmark_model.pth` and console logs.
This trial checkpoint is not presented as a fully trained obstacle-avoidance model.
After the download completes, release this benchmark runtime to stop further GPU use.
'''), code(setup.replace("T4 GPU", "T4/A100 GPU")), code(benchmark), code(export)]
notebook = {"nbformat":4,"nbformat_minor":5,"metadata":{
    "accelerator":"GPU","colab":{"provenance":[],"gpuType":args.gpu},
    "kernelspec":{"display_name":"Python 3","name":"python3"},
    "language_info":{"name":"python"}},"cells":cells}
path = ROOT / f"colab/VROOM_Benchmark_{args.gpu}_{args.minutes:g}min.ipynb"
path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding="utf-8")
print(path, path.stat().st_size, "bytes")
