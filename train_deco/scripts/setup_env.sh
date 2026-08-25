#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PACKAGE_ROOT="${PACKAGE_ROOT:-${PROJECT_ROOT}/train_deco}"
PYTHON_COMMAND="${PYTHON_COMMAND:-python3}"
VENV_PATH="${VENV_PATH:-${PACKAGE_ROOT}/.venv}"
PYPI_INDEX="${PYPI_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
PYTORCH_INDEX="${PYTORCH_INDEX:-https://download.pytorch.org/whl/cu128}"

usage() {
  cat <<'EOF'
用法：
  bash scripts/pick_tube_vision/01_setup_env.sh [选项]

为纯视觉 DECO 创建独立 Python 环境。默认安装：
  - torch 2.11.0 + torchvision 0.26.0
  - CUDA wheel: cu128（适用于本机 RTX 4090 和服务器 RTX PRO 6000）
  - numpy、Pillow、pyarrow、einops、tqdm、pytest

选项：
  --python COMMAND   创建虚拟环境所用的 Python，默认 python3
  --venv PATH        虚拟环境目录，默认 <代码目录>/.venv
  -h, --help         显示帮助

环境变量：
  PROJECT_ROOT       代码目录
  PYPI_INDEX         普通 Python 包镜像，默认清华 PyPI
  PYTORCH_INDEX      PyTorch CUDA wheel 索引，默认官方 cu128
  ALLOW_CPU=1        允许 CUDA 不可用时环境检查仍然成功

脚本只安装 Python/CUDA wheels，不安装或修改 NVIDIA 驱动。
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      PYTHON_COMMAND="$2"
      shift 2
      ;;
    --venv)
      VENV_PATH="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "未知参数：$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

"${PYTHON_COMMAND}" -c 'import sys; assert (3, 10) <= sys.version_info < (3, 14), sys.version'
if [[ ! -x "${VENV_PATH}/bin/python" ]]; then
  "${PYTHON_COMMAND}" -m venv "${VENV_PATH}"
fi

VENV_PYTHON="${VENV_PATH}/bin/python"
"${VENV_PYTHON}" -m pip install --upgrade pip setuptools wheel --index-url "${PYPI_INDEX}"
"${VENV_PYTHON}" -m pip install \
  torch==2.11.0 torchvision==0.26.0 \
  --index-url "${PYTORCH_INDEX}"
"${VENV_PYTHON}" -m pip install \
  numpy==1.26.4 \
  Pillow==11.0.0 \
  pyarrow==18.1.0 \
  einops==0.8.1 \
  tqdm==4.67.1 \
  pytest==9.1.1 \
  --index-url "${PYPI_INDEX}"

ALLOW_CPU="${ALLOW_CPU:-0}" "${VENV_PYTHON}" - <<'PY'
import os
import platform

import numpy
import pyarrow
import torch
import torchvision
import tqdm
from PIL import Image

print(f"python={platform.python_version()}")
print(f"torch={torch.__version__}")
print(f"torchvision={torchvision.__version__}")
print(f"numpy={numpy.__version__} pyarrow={pyarrow.__version__} pillow={Image.__version__}")
print(f"tqdm={tqdm.__version__}")
print(f"cuda_available={torch.cuda.is_available()} cuda_runtime={torch.version.cuda}")
if torch.cuda.is_available():
    print(f"gpu_count={torch.cuda.device_count()}")
    for index in range(torch.cuda.device_count()):
        print(f"gpu[{index}]={torch.cuda.get_device_name(index)}")
elif os.environ.get("ALLOW_CPU") != "1":
    raise SystemExit("CUDA 不可用；请先检查 NVIDIA 驱动和 nvidia-smi，或用 ALLOW_CPU=1 仅验证 CPU 环境")
PY

echo "环境已安装：${VENV_PATH}"
echo "下一步：bash ${PACKAGE_ROOT}/scripts/prepare_data.sh --mode local"
