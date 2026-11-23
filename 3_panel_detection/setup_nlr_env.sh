#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

python - <<'PY'
import os
import sys
import torch
if not torch.cuda.is_available():
    raise SystemExit("A CUDA-enabled PyTorch installation is required")
print(f"torch={torch.__version__} cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0)}")
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
PY

if python -c "import sys; raise SystemExit(0 if sys.prefix != sys.base_prefix else 1)"; then
  PIP_SCOPE=()
else
  PIP_SCOPE=(--user)
fi
python -m pip install "${PIP_SCOPE[@]}" --upgrade-strategy only-if-needed \
  "setuptools==80.9.0" wheel "numpy>=1.26,<2" "Pillow>=10,<12" \
  "opencv-python-headless>=4.8,<5" "mmengine==0.10.7" "seaborn==0.13.2" \
  "tensorboard>=2.16,<3" addict packaging pyyaml yapf ninja

if ! python - <<'PY'
try:
    import mmcv
    from mmcv.ops import nms  # noqa: F401
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if mmcv.__version__ == "2.1.0" else 1)
PY
then
  command -v nvcc >/dev/null || { echo "nvcc is required to build MMCV" >&2; exit 1; }
  MMCV_WITH_OPS=1 MAX_JOBS=4 python -m pip install "${PIP_SCOPE[@]}" -v \
    --no-build-isolation --no-deps "git+https://github.com/open-mmlab/mmcv.git@v2.1.0"
fi
python -m pip install "${PIP_SCOPE[@]}" --upgrade-strategy only-if-needed "mmdet==3.3.0"

