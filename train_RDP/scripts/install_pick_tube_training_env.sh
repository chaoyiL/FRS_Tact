#!/usr/bin/env bash
set -euo pipefail

# Create the Python environment used by pick-tube AT/LDP training on a
# single NVIDIA RTX PRO 6000 (Blackwell). The defaults mirror the versions
# already validated in this repository.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RDP_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${RDP_DIR}"

PYTHON_VERSION=${PYTHON_VERSION:-3.12}
VENV_DIR=${VENV_DIR:-${RDP_DIR}/.venv}
PYPI_INDEX_URL=${PYPI_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}
TORCH_INDEX_URL=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}
TORCH_VERSION=${TORCH_VERSION:-2.10.0}
TORCHVISION_VERSION=${TORCHVISION_VERSION:-0.25.0}
TORCHAUDIO_VERSION=${TORCHAUDIO_VERSION:-2.10.0}
HF_HUB_VERSION=${HF_HUB_VERSION:-0.35.3}
RECREATE_VENV=${RECREATE_VENV:-0}
DRY_RUN=${DRY_RUN:-0}

run() {
  printf '+ '
  printf '%q ' "$@"
  printf '\n'
  if [[ "${DRY_RUN}" != "1" ]]; then
    "$@"
  fi
}

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required to bootstrap uv." >&2
  exit 1
fi

if command -v uv >/dev/null 2>&1; then
  UV_BIN=$(command -v uv)
else
  echo "uv was not found; installing it in the current user's Python bin directory."
  run python3 -m pip install --user --index-url "${PYPI_INDEX_URL}" uv
  USER_BASE=$(python3 -m site --user-base)
  UV_BIN=${USER_BASE}/bin/uv
fi

if [[ "${DRY_RUN}" != "1" && ! -x "${UV_BIN}" ]]; then
  echo "uv installation failed: ${UV_BIN} is not executable." >&2
  exit 1
fi

venv_args=(venv --python "${PYTHON_VERSION}")
if [[ "${RECREATE_VENV}" == "1" ]]; then
  venv_args+=(--clear)
fi
venv_args+=("${VENV_DIR}")

if [[ ! -x "${VENV_DIR}/bin/python" || "${RECREATE_VENV}" == "1" ]]; then
  run "${UV_BIN}" "${venv_args[@]}"
else
  echo "Reusing existing environment: ${VENV_DIR}"
fi

PYTHON_BIN=${VENV_DIR}/bin/python
run "${UV_BIN}" pip install --python "${PYTHON_BIN}" \
  --index-url "${TORCH_INDEX_URL}" \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  "torchaudio==${TORCHAUDIO_VERSION}"

run "${UV_BIN}" pip install --python "${PYTHON_BIN}" \
  --index-url "${PYPI_INDEX_URL}" \
  -r requirements-rdp-training.txt

# Keep this explicit install aligned with requirements-rdp-training.txt. It
# provides the `hf` CLI without silently replacing the compatible library pin.
run "${UV_BIN}" pip install --python "${PYTHON_BIN}" \
  --index-url "${PYPI_INDEX_URL}" \
  "huggingface-hub==${HF_HUB_VERSION}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "Dry run complete; no environment changes were made."
  exit 0
fi

run "${UV_BIN}" pip check --python "${PYTHON_BIN}"

"${PYTHON_BIN}" - <<'PY'
import sys

import accelerate
import diffusers
import hydra
import huggingface_hub
import numpy
import pyarrow
import sklearn
import torch
import torch.nn.functional as F
import torchvision
import zarr

print(f"python={sys.version.split()[0]}")
print(f"torch={torch.__version__}, torchvision={torchvision.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"accelerate={accelerate.__version__}, diffusers={diffusers.__version__}")
print(f"huggingface_hub={huggingface_hub.__version__}")
print(f"numpy={numpy.__version__}, zarr={zarr.__version__}")
print(f"pyarrow={pyarrow.__version__}, sklearn={sklearn.__version__}")

if not torch.cuda.is_available():
    raise SystemExit(
        "CUDA is unavailable. Check the NVIDIA driver and TORCH_INDEX_URL "
        "before starting training."
    )

# Catch CUDA/cuBLAS incompatibilities before a multi-hour job. CUDA visibility
# alone did not catch the Blackwell linear-algebra failure seen on this project.
x = torch.randn(32, 128, device="cuda")
weight = torch.randn(64, 128, device="cuda")
y = F.linear(x, weight)
torch.cuda.synchronize()
if not torch.isfinite(y).all():
    raise SystemExit("CUDA linear smoke produced non-finite values")
print("cuda_linear_smoke=ok")

print(f"gpu_count={torch.cuda.device_count()}")
for index in range(torch.cuda.device_count()):
    name = torch.cuda.get_device_name(index)
    capability = torch.cuda.get_device_capability(index)
    print(f"gpu[{index}]={name}, compute_capability={capability[0]}.{capability[1]}")
    if "RTX PRO 6000" not in name.upper():
        print(f"warning: gpu[{index}] is not identified as an RTX PRO 6000")
PY

"${VENV_DIR}/bin/hf" version

printf '\nEnvironment ready.\n'
printf 'Activate: source %q/bin/activate\n' "${VENV_DIR}"
printf 'Train:    bash scripts/train_pick_tube_single_gpu.sh all\n'
