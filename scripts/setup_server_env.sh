#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"

echo "[1/5] Creating virtual environment at ${VENV_DIR}"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"

echo "[2/5] Activating virtual environment"
source "${VENV_DIR}/bin/activate"

echo "[3/5] Upgrading pip"
python -m pip install --upgrade pip

echo "[4/5] Installing project requirements"
pip install -r requirements.txt

echo "[5/5] Installing CUDA-enabled PyTorch"
pip install torch --index-url "${TORCH_INDEX_URL}"

echo
echo "Environment ready. Verifying CUDA availability..."
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("cuda_name:", torch.cuda.get_device_name(0))
else:
    raise SystemExit("CUDA is not available. Check the server driver/CUDA wheel configuration.")
PY
