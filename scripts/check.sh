#!/usr/bin/env bash
# Quality gate: chạy TRƯỚC MỖI COMMIT. CI chạy đúng script này.
#   bash scripts/check.sh
set -euo pipefail
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ -x .venv/Scripts/python.exe ]]; then PY=.venv/Scripts/python.exe
elif [[ -x .venv/bin/python ]]; then PY=.venv/bin/python
else PY=python
fi

echo "==> [1/3] ruff check (lỗi logic, import, style)"
"$PY" -m ruff check .
echo "==> [2/3] ruff format --check (định dạng code)"
"$PY" -m ruff format --check .
echo "==> [3/3] pytest"
"$PY" -m pytest -q "$@"
echo "==> Tất cả đều PASS"
