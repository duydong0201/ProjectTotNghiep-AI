# Quality gate: chạy TRƯỚC MỖI COMMIT. CI chạy bản bash tương đương (scripts/check.sh).
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check.ps1
# Tham số thêm được chuyển cho pytest, ví dụ: scripts/check.ps1 -k split
$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Set-Location (Join-Path $PSScriptRoot "..")

$py = if (Test-Path ".venv\Scripts\python.exe") { ".venv\Scripts\python.exe" } else { "python" }

function Step($title, [scriptblock]$cmd) {
    Write-Host "==> $title" -ForegroundColor Cyan
    & $cmd
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAIL: $title" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

Step "[1/3] ruff check (lỗi logic, import, style)" { & $py -m ruff check . }
Step "[2/3] ruff format --check (định dạng code)" { & $py -m ruff format --check . }
Step "[3/3] pytest" { & $py -m pytest -q @args }
Write-Host "==> Tất cả đều PASS" -ForegroundColor Green
