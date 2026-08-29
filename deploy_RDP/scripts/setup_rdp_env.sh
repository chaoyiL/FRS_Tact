#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RDP_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${RDP_DIR}"

PYTHON_VERSION=${PYTHON_VERSION:-3.12}
VENV_DIR=${VENV_DIR:-${RDP_DIR}/.venv}
PYPI_INDEX_URL=${PYPI_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}
TORCH_VERSION=${TORCH_VERSION:-2.10.0}
TORCHVISION_VERSION=${TORCHVISION_VERSION:-0.25.0}
TORCHAUDIO_VERSION=${TORCHAUDIO_VERSION:-2.10.0}
WITH_TACTILE_PRECOMPUTE=${WITH_TACTILE_PRECOMPUTE:-0}
JAX_VENV_DIR=${JAX_VENV_DIR:-${RDP_DIR}/.venv-jax}
DRY_RUN=${DRY_RUN:-0}

run() {
  printf '+ '
  printf '%q ' "$@"
  printf '\n'
  if [[ "${DRY_RUN}" != "1" ]]; then
    "$@"
  fi
}

if command -v uv >/dev/null 2>&1; then
  UV_BIN=$(command -v uv)
else
  UV_BIN=${UV_BIN:-${HOME}/.local/bin/uv}
  run python3 -m pip install --user \
    --index-url "${PYPI_INDEX_URL}" uv
fi

run "${UV_BIN}" venv --python "${PYTHON_VERSION}" "${VENV_DIR}"
PYTHON_BIN=${VENV_DIR}/bin/python
run "${UV_BIN}" pip install --python "${PYTHON_BIN}" \
  --index-url "${PYPI_INDEX_URL}" \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  "torchaudio==${TORCHAUDIO_VERSION}"
run "${UV_BIN}" pip install --python "${PYTHON_BIN}" \
  --index-url "${PYPI_INDEX_URL}" \
  -r requirements-rdp-deploy.txt

if [[ "${WITH_TACTILE_PRECOMPUTE}" == "1" ]]; then
  run "${UV_BIN}" venv --python "${PYTHON_VERSION}" "${JAX_VENV_DIR}"
  run "${UV_BIN}" pip install --python "${JAX_VENV_DIR}/bin/python" \
    --index-url "${PYPI_INDEX_URL}" \
    'jax[cuda12]==0.10.2' 'flax==0.12.7' 'numpy==2.5.0' \
    'pyarrow==24.0.0' 'pillow==12.3.0' 'pyyaml==6.0.3'
fi

if [[ "${DRY_RUN}" != "1" ]]; then
  "${PYTHON_BIN}" - <<'PY'
import torch
import accelerate
import diffusers
import zarr

print(f"torch={torch.__version__}")
print(f"accelerate={accelerate.__version__}")
print(f"diffusers={diffusers.__version__}")
print(f"zarr={zarr.__version__}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; check the NVIDIA driver and CUDA wheel flavor")
print(f"cuda={torch.version.cuda}, gpu_count={torch.cuda.device_count()}")
for index in range(torch.cuda.device_count()):
    print(f"gpu[{index}]={torch.cuda.get_device_name(index)}")
PY
fi

printf '\nEnvironment ready. Activate it with:\n  source %q/bin/activate\n' "${VENV_DIR}"
if [[ "${WITH_TACTILE_PRECOMPUTE}" == "1" ]]; then
  printf 'Tactile precompute environment: %q/bin/python\n' "${JAX_VENV_DIR}"
fi
