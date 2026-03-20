# setup_env.ps1
# Sets up the Poetry venv with Python 3.11, installs deps (incl. CUDA torch via Poetry source), and builds torch-mesh-isect CUDA extension.

param(
    [string]$PythonPath = "C:\Users\Zixuan Lu\AppData\Local\Programs\Python\Python311\python.exe",
    [string]$CudaHome = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition

Push-Location $RepoRoot
try {
    # --- 1. Create .venv with Python 3.11 ---
    if (Test-Path ".venv") {
        Write-Host "[1/4] .venv already exists, skipping creation" -ForegroundColor Yellow
    } else {
        Write-Host "[1/4] Creating .venv with $PythonPath ..." -ForegroundColor Cyan
        & $PythonPath -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw "Failed to create .venv" }
    }

    # --- 2. Activate the venv ---
    Write-Host "[2/4] Activating .venv ..." -ForegroundColor Cyan
    & ".venv\Scripts\Activate.ps1"

    # --- 3. Poetry install ---
    Write-Host "[2b/4] Running poetry install ..." -ForegroundColor Cyan
    $env:VIRTUAL_ENV = Join-Path $RepoRoot ".venv"
    $env:PATH = (Join-Path $RepoRoot ".venv\Scripts") + ";" + $env:PATH
    poetry install
    if ($LASTEXITCODE -ne 0) { throw "poetry install failed" }

    # --- 3b. Apply torch-mesh-isect compatibility patch ---
    Write-Host "[2c/4] Applying torch-mesh-isect patch ..." -ForegroundColor Cyan
    $patchFile = Join-Path $RepoRoot "patches\torch-mesh-isect-win-fix.patch"
    $isectDir = Join-Path $RepoRoot "externals\torch-mesh-isect"
    $patchNeeded = $true
    try {
        $ErrorActionPreference = "Continue"
        git -C $isectDir apply --check $patchFile 2>&1 | Out-Null
        $patchNeeded = ($LASTEXITCODE -eq 0)
    } finally {
        $ErrorActionPreference = "Stop"
    }
    if ($patchNeeded) {
        git -C $isectDir apply $patchFile
        Write-Host "  Patch applied successfully" -ForegroundColor DarkGray
    } else {
        Write-Host "  Patch already applied or not needed, skipping" -ForegroundColor DarkGray
    }

    # --- 4. Set CUDA env vars and build torch-mesh-isect ---
    Write-Host "[3/4] Building torch-mesh-isect CUDA extension ..." -ForegroundColor Cyan
    $env:CUDA_HOME = $CudaHome
    $env:CUDA_SAMPLES_INC = Join-Path $RepoRoot "externals\cuda-samples\Common"
    Write-Host "  CUDA_HOME = $env:CUDA_HOME" -ForegroundColor DarkGray
    Write-Host "  CUDA_SAMPLES_INC = $env:CUDA_SAMPLES_INC" -ForegroundColor DarkGray

    Push-Location (Join-Path $RepoRoot "externals\torch-mesh-isect")
    try {
        pip install -r requirements.txt
        if ($LASTEXITCODE -ne 0) { throw "pip install requirements failed" }

        python setup.py install
        if ($LASTEXITCODE -ne 0) { throw "torch-mesh-isect build failed" }
    } finally {
        Pop-Location
    }

    # --- 5. Verify ---
    Write-Host "[4/4] Verifying installation ..." -ForegroundColor Cyan
    python -c "import torch; print('torch', torch.__version__); import mesh_intersection; print('mesh_intersection OK'); import repair_factory; print('repair_factory OK')"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Verification failed - check CUDA toolchain and GPU drivers." -ForegroundColor Red
        exit 1
    }

    Write-Host ""
    Write-Host "Setup complete. Run examples with:" -ForegroundColor Green
    Write-Host "  .venv\Scripts\Activate.ps1" -ForegroundColor White
    Write-Host "  python repair_factory.py --config configs/misc.yaml" -ForegroundColor White
} finally {
    Pop-Location
}
