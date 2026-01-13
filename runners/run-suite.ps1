# Run suite script for VideoRAG - starts both backend and frontend servers

$ErrorActionPreference = "Stop"

# Get the script directory and project root
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$BackendDir = Join-Path $ProjectRoot "Vimo-desktop" "python_backend"
$FrontendDir = Join-Path $ProjectRoot "Vimo-desktop"

# Track processes for cleanup
$global:BackendProcess = $null
$global:FrontendProcess = $null
$global:CleanupCalled = $false

function Write-Step {
    param([string]$Message, [string]$Color = "Cyan")
    Write-Host $Message -ForegroundColor $Color
}

function Cleanup {
    if ($global:CleanupCalled) {
        return
    }
    $global:CleanupCalled = $true
    
    Write-Step "`n🔔 Shutting down services..." "Yellow"
    
    # Terminate backend process
    if ($global:BackendProcess -and -not $global:BackendProcess.HasExited) {
        Write-Step "Stopping backend server..." "Yellow"
        try {
            Stop-Process -Id $global:BackendProcess.Id -Force -ErrorAction SilentlyContinue
            $global:BackendProcess.WaitForExit(5000)
        } catch {
            Write-Step "⚠️  Error stopping backend: $_" "Red"
        }
    }
    
    # Terminate frontend process
    if ($global:FrontendProcess -and -not $global:FrontendProcess.HasExited) {
        Write-Step "Stopping frontend server..." "Yellow"
        try {
            Stop-Process -Id $global:FrontendProcess.Id -Force -ErrorAction SilentlyContinue
            $global:FrontendProcess.WaitForExit(5000)
        } catch {
            Write-Step "⚠️  Error stopping frontend: $_" "Red"
        }
    }
    
    Write-Step "✅ Cleanup completed" "Green"
}

# Register cleanup on exit
Register-EngineEvent -SourceIdentifier PowerShell.Exiting -Action { Cleanup } | Out-Null

# Handle Ctrl+C
$null = Register-ObjectEvent -InputObject ([System.Console]) -EventName CancelKeyPress -Action {
    $Event.Sender.CancelKeyPress = $true
    Cleanup
    exit 0
}

Write-Step "🚀 Starting VideoRAG development suite..." "Cyan"
Write-Step ""

# Check prerequisites
Write-Step "Checking prerequisites..." "Cyan"

# Check for uv
try {
    $null = Get-Command uv -ErrorAction Stop
    Write-Step "✓ uv is installed" "Green"
} catch {
    Write-Step "❌ uv is not installed. Please run Vimo-desktop/python_backend/setup.ps1 first" "Red"
    exit 1
}

# Check for pnpm
try {
    $null = Get-Command pnpm -ErrorAction Stop
    Write-Step "✓ pnpm is installed" "Green"
} catch {
    Write-Step "❌ pnpm is not installed. Please install pnpm first" "Red"
    exit 1
}

Write-Step ""

# Start backend server in a new window
Write-Step "Starting backend API server..." "Cyan"
try {
    $backendJob = Start-Process -FilePath "powershell" `
        -ArgumentList "-NoExit", "-Command", "cd '$BackendDir'; uv run python videorag_api.py" `
        -PassThru `
        -WindowStyle Normal
    
    $global:BackendProcess = $backendJob
    Write-Step "✓ Backend server started (PID: $($backendJob.Id))" "Green"
    Write-Step "  Backend directory: $BackendDir" "Gray"
    Write-Step "  Backend logs visible in separate window" "Gray"
    
    # Wait a moment for backend to start
    Start-Sleep -Seconds 3
    
} catch {
    Write-Step "❌ Failed to start backend server: $_" "Red"
    Cleanup
    exit 1
}

Write-Step ""

# Start frontend dev server in a new window
Write-Step "Starting frontend dev server..." "Cyan"
try {
    $frontendJob = Start-Process -FilePath "powershell" `
        -ArgumentList "-NoExit", "-Command", "cd '$FrontendDir'; pnpm run dev" `
        -PassThru `
        -WindowStyle Normal
    
    $global:FrontendProcess = $frontendJob
    Write-Step "✓ Frontend dev server started (PID: $($frontendJob.Id))" "Green"
    Write-Step "  Frontend directory: $FrontendDir" "Gray"
    Write-Step "  Frontend logs visible in separate window" "Gray"
    
} catch {
    Write-Step "❌ Failed to start frontend server: $_" "Red"
    Cleanup
    exit 1
}

Write-Step ""
Write-Step "✅ Both servers are running!" "Green"
Write-Step ""
Write-Step "Backend API: http://localhost:64451 (default port)" "White"
Write-Step "Frontend: Check the Electron window that should open shortly" "White"
Write-Step ""
Write-Step "Press Ctrl+C to stop both servers" "Gray"
Write-Step ""

# Wait for processes to exit or keep running
try {
    while ($true) {
        # Check if processes are still running
        if ($global:BackendProcess.HasExited) {
            Write-Step "⚠️  Backend process has exited" "Yellow"
            break
        }
        if ($global:FrontendProcess.HasExited) {
            Write-Step "⚠️  Frontend process has exited" "Yellow"
            break
        }
        Start-Sleep -Seconds 1
    }
} catch {
    Write-Step "❌ Error monitoring processes: $_" "Red"
} finally {
    Cleanup
}
