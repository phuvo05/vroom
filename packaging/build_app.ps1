# Build the VROOM demo as a Windows app in dist\VROOM (needs uv: https://docs.astral.sh/uv/).
# A separate venv with CPU-only torch keeps the app free of CUDA libraries.
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$venv = Join-Path $root "build\venv"
if (-not (Test-Path "$venv\Scripts\python.exe")) {
    uv venv --python 3.11 $venv
    uv pip install --python "$venv\Scripts\python.exe" --index-url https://download.pytorch.org/whl/cpu `
        --extra-index-url https://pypi.org/simple "torch==2.14.1" "numpy==2.4.6" "gymnasium==1.4.0" `
        "Box2D==2.3.10" "pygame-ce==2.5.8" opencv-python-headless Pillow pyinstaller
}
& "$venv\Scripts\python.exe" -m PyInstaller --noconfirm --clean `
    --distpath (Join-Path $root "dist") --workpath (Join-Path $root "build\pyinstaller") `
    (Join-Path $PSScriptRoot "vroom.spec")
Copy-Item (Join-Path $PSScriptRoot "README-app.txt") (Join-Path $root "dist\VROOM\README.txt") -Force
Write-Host "Built $root\dist\VROOM\VROOM.exe"
