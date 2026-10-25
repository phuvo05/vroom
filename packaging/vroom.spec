# PyInstaller spec for the VROOM demo (Windows, one-folder, CPU-only torch).
# Build: packaging\build_app.ps1   ->   dist\VROOM\VROOM.exe
# The checkpoint is NOT bundled: users put best.pth next to VROOM.exe or pick it in the app.
from pathlib import Path

ROOT = Path(SPECPATH).parent

a = Analysis(
    [str(ROOT / "dqn" / "demo_ui.py")],
    pathex=[str(ROOT / "dqn")],
    hiddenimports=["gymnasium.envs.box2d.car_racing"],
    excludes=["matplotlib", "IPython", "jupyter", "pytest", "torch.utils.tensorboard", "tensorboard",
              "torchvision", "torchaudio", "pandas", "scipy"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="VROOM",
    console=False,  # Tk window only; no terminal
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="VROOM", upx=False)
