#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
COUNT=""
CONFIDENCE="0.50"

usage() {
  echo "Usage: bash run_deeplabv3.sh [--count N] [--confidence 0..1]"
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

python -m pip install --upgrade-strategy only-if-needed \
  "numpy>=1.26,<3" "Pillow>=10,<13" "onnxruntime-gpu==1.26.0"
python -m src.download deeplabv3

SLUG="${CONFIDENCE//./p}"
COMMAND=(python -m src.deeplabv3
  --input ../data/bitgaram/raw_tiles/z19
  --output "outputs/deeplabv3/conf_${SLUG}"
  --conf "$CONFIDENCE" --device cuda --resume)
[[ -z "$COUNT" ]] || COMMAND+=(--max-images "$COUNT")
"${COMMAND[@]}"
