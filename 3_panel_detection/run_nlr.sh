#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
COUNT=""
CONFIDENCE="0.25"

usage() {
  echo "Usage: bash run_nlr.sh [--count N] [--confidence 0..1]"
}

while (($#)); do
  case "$1" in
    --count) COUNT="${2:?--count requires N}"; shift 2 ;;
    --confidence) CONFIDENCE="${2:?--confidence requires a value}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -z "$COUNT" || "$COUNT" =~ ^[1-9][0-9]*$ ]] || { echo "--count must be positive" >&2; exit 2; }
python - "$CONFIDENCE" <<'PY'
import sys
value = float(sys.argv[1])
if not 0 <= value <= 1:
    raise SystemExit("--confidence must be between 0 and 1")
PY

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
  addict packaging pyyaml yapf ninja

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
MODEL_ARGS=()
MODEL_KEY="original"
if [[ -n "${NLR_RUN:-}" ]]; then
  [[ -s "$NLR_RUN/best_model.pth" && -s "$NLR_RUN/inference_config.py" ]] || {
    echo "NLR_RUN must point to a completed training folder" >&2; exit 2;
  }
  MODEL_ARGS=(--model "$NLR_RUN/best_model.pth" --config "$NLR_RUN/inference_config.py")
  MODEL_KEY="finetuned_$(basename "$NLR_RUN")"
else
  python -m src.download nlr
fi

SLUG="${CONFIDENCE//./p}"
COMMAND=(python -m src.nlr
  --input ../data/bitgaram/raw_tiles/z19
  --output "outputs/nlr/${MODEL_KEY}/conf_${SLUG}"
  --conf "$CONFIDENCE" --device cuda:0 --resume --hard-exit-after-success)
COMMAND+=("${MODEL_ARGS[@]}")
[[ -z "$COUNT" ]] || COMMAND+=(--max-images "$COUNT")
"${COMMAND[@]}"
