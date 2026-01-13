# Setup script for VideoRAG backend using uv (Windows PowerShell)

$ErrorActionPreference = "Stop"

Write-Host "Setting up VideoRAG backend with uv..." -ForegroundColor Cyan

# Check if uv is installed
try {
    $null = Get-Command uv -ErrorAction Stop
    Write-Host "uv is already installed" -ForegroundColor Green
} catch {
    Write-Host "uv is not installed. Installing uv..." -ForegroundColor Yellow
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    Write-Host "uv installed successfully" -ForegroundColor Green
    # Refresh PATH
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
}

# Check if Python 3.11 is available, install if not
Write-Host "Checking for Python 3.11..." -ForegroundColor Cyan
$pythonCheck = uv python list --only-installed 2>&1 | Select-String "3\.11"
if (-not $pythonCheck) {
    Write-Host "Installing Python 3.11..." -ForegroundColor Yellow
    uv python install 3.11
}

Write-Host "Creating virtual environment and installing dependencies..." -ForegroundColor Cyan
uv sync --python 3.11

Write-Host "Installing ImageBind from git (no-deps)..." -ForegroundColor Cyan
uv pip install --no-deps "git+https://github.com/facebookresearch/ImageBind.git@3fcf5c9039de97f6ff5528ee4a9dce903c5979b3"

Write-Host ""
Write-Host "Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "To start the server, run:" -ForegroundColor Cyan
Write-Host "  uv run python videorag_api.py" -ForegroundColor White
Write-Host ""
Write-Host "Or use the Vimo desktop app which will auto-start the backend." -ForegroundColor Gray
