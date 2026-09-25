#!/usr/bin/env bash
set -euo pipefail

eval_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
venv_dir="$eval_dir/.venv"
python_base="$eval_dir/.python311"
backend="${CSGO_EVAL_TORCH_BACKEND:-cpu}"
install_environment=1
prepare_weights=1

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'HELP'
Create or update the independent Python 3.11 evaluator environment at .venv.

Usage: bash setup_env.sh [--download-weights | --skip-weights | --weights-only]

The script bootstraps a self-contained Python 3.11 runtime under .python311
with conda if python3.11 is unavailable on PATH. It never uses or changes a
model environment implicitly. Set CSGO_EVAL_PYTHON311 to a specific Python 3.11
interpreter if desired. The default PyTorch wheels are CPU-only; set
CSGO_EVAL_TORCH_BACKEND=cu128 explicitly for CUDA 12.8 wheels.

After setup, invoke run_eval.py using .venv/bin/python. Evaluator invocations
do not install or verify packages. Installed versions are recorded in
.venv/install-manifest.json and .venv/pip-freeze.txt.

Default / --download-weights: reuse valid user Torch cache weights first,
then UniLIP loaded_models, then this evaluator's loaded_models directory.
Download only weights missing from all sources into evaluator loaded_models.
Torch cache honors TORCH_HOME, then XDG_CACHE_HOME/torch, then ~/.cache/torch.
--skip-weights installs only the environment (e.g. localization-only use).
--weights-only prepares weights using an already installed evaluator .venv.
CSGO_UNILIP_ROOT can name a relocated UniLIP checkout. Sibling UniLIP and
UniLP directories are searched by default. Formal evaluation never downloads.
HELP
  exit 0
fi
if [[ $# -gt 1 ]]; then
  echo "Choose at most one weight preparation option. See --help." >&2
  exit 2
fi
case "${1:-}" in
  ""|--download-weights) ;;
  --skip-weights) prepare_weights=0 ;;
  --weights-only) install_environment=0 ;;
  *) echo "Unknown argument: $1. See --help." >&2; exit 2 ;;
esac

if (( ! install_environment )); then
  exec "$venv_dir/bin/python" "$eval_dir/prepare_metric_assets.py"
fi

case "$backend" in
  cpu|cu128) ;;
  *) echo "CSGO_EVAL_TORCH_BACKEND must be cpu or cu128 (got: $backend)" >&2; exit 2 ;;
esac

if [[ -x "$venv_dir/bin/python" ]]; then
  if [[ "$("$venv_dir/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" != 3.11 ]]; then
    echo "Existing $venv_dir is not Python 3.11; move it aside before setup." >&2
    exit 2
  fi
else
  if [[ -d "$venv_dir" ]]; then
    echo "Incomplete $venv_dir exists; move it aside before setup." >&2
    exit 2
  fi
  if [[ -n "${CSGO_EVAL_PYTHON311:-}" ]]; then
    base_python="$CSGO_EVAL_PYTHON311"
  elif command -v python3.11 >/dev/null 2>&1; then
    base_python="$(command -v python3.11)"
  else
    if [[ ! -x "$python_base/bin/python" ]]; then
      command -v conda >/dev/null 2>&1 || {
        echo "Python 3.11 or conda is required to create the evaluator venv." >&2
        exit 2
      }
      conda create --yes --prefix "$python_base" 'python=3.11' pip
    fi
    base_python="$python_base/bin/python"
  fi
  if [[ "$("$base_python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" != 3.11 ]]; then
    echo "$base_python must be Python 3.11." >&2
    exit 2
  fi
  "$base_python" -m venv "$venv_dir"
fi

venv_python="$venv_dir/bin/python"
"$venv_python" - "$venv_dir" <<'PY'
import sys
from pathlib import Path

expected = Path(sys.argv[1]).resolve()
actual = Path(sys.prefix).resolve()
base = Path(sys.base_prefix).resolve()
config = expected / 'pyvenv.cfg'
if actual != expected or base == actual or not config.is_file():
    raise SystemExit(f'Refusing to install into a non-isolated environment: {actual}')
if 'include-system-site-packages = false' not in config.read_text().lower():
    raise SystemExit(f'Refusing to install into a venv with system packages: {config}')
PY
"$venv_python" -m pip install --upgrade 'pip==25.2'
"$venv_python" -m pip install \
  --index-url "https://download.pytorch.org/whl/$backend" \
  "torch==2.7.1+$backend" "torchvision==0.22.1+$backend"
"$venv_python" -m pip install -r "$eval_dir/requirements.txt"
"$venv_python" -m pip freeze > "$venv_dir/pip-freeze.txt"

cd "$eval_dir"
CSGO_EVAL_BACKEND="$backend" CSGO_EVAL_DIR="$eval_dir" "$venv_python" - <<'PY'
import importlib.metadata as metadata
import json
import os
from pathlib import Path

import cv2
import numpy
import PIL
import requests
import scipy
import torch
import torchvision
import yaml
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import metrics_localization
import metrics_images
import metrics_continuous
import fvd_metric

backend = os.environ['CSGO_EVAL_BACKEND']
expected_suffix = '+cpu' if backend == 'cpu' else '+cu128'
if not torch.__version__.startswith('2.7.1' + expected_suffix):
    raise SystemExit(f'Unexpected torch build: {torch.__version__}; expected {expected_suffix}')
if not torchvision.__version__.startswith('0.22.1' + expected_suffix):
    raise SystemExit(f'Unexpected torchvision build: {torchvision.__version__}; expected {expected_suffix}')

versions = {name: metadata.version(name) for name in (
    'pip', 'torch', 'torchvision', 'numpy', 'Pillow', 'torchmetrics',
    'torch-fidelity', 'scipy', 'opencv-python-headless', 'PyYAML', 'requests',
)}
manifest = {
    'backend': backend,
    'python': __import__('sys').version.split()[0],
    'packages': versions,
    'imports_checked': ['metrics_localization', 'metrics_images', 'metrics_continuous', 'fvd_metric'],
}
manifest_path = Path(os.environ['CSGO_EVAL_DIR']) / '.venv' / 'install-manifest.json'
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
print(f'Evaluator ready: {manifest_path.parent / "bin" / "python"}')
print(f'Torch backend: {backend}; torch {torch.__version__}; torchvision {torchvision.__version__}')
print(f'Installed version manifest: {manifest_path}')
PY

if (( prepare_weights )); then
  "$venv_python" "$eval_dir/prepare_metric_assets.py"
fi
