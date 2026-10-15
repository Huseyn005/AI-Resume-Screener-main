$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$backend = Join-Path $root "backend"
$activate = ".\venv\Scripts\Activate.ps1"
$backendCommand = "Set-Location -LiteralPath '$backend'; if (Test-Path '$activate') { . '$activate' }; uvicorn app.main:app --port 8000"

Start-Process -FilePath "powershell.exe" `
  -WorkingDirectory $backend `
  -ArgumentList @("-NoExit", "-Command", $backendCommand)

& npm --prefix (Join-Path $root "frontend") run dev
if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}
