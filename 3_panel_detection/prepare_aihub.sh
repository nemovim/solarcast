#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

AIHUB_DIR="$PWD/../data/aihub"
ZIPS_DIR="$AIHUB_DIR/zips"
OUTPUT_DIR="$AIHUB_DIR/coco_256"
SKIP_UNZIP=0

usage() {
  cat <<'EOF'
Usage: bash prepare_aihub.sh [options]

Extracts AI-Hub raw zip files and converts them into 256px COCO dataset.

Options:
  --output <dir>    Output directory for converted COCO dataset (default: ../data/aihub/coco_256)
  --skip-unzip      Skip zip extraction and run COCO conversion directly
  -h, --help        Show this help message
EOF
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output) OUTPUT_DIR="$2"; shift 2 ;;
    --skip-unzip) SKIP_UNZIP=1; shift ;;
    -h|--help) usage ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

unzip_if_needed() {
  local zip_file="$1"
  local target_dir="$2"
  local label="$3"

  if [ ! -f "$zip_file" ]; then
    echo "[!] $label not found at $zip_file (skipping)"
    return
  fi

  if [ -d "$target_dir" ] && [ "$(ls -A "$target_dir" 2>/dev/null)" ]; then
    echo "[✓] $label already extracted to $target_dir"
  else
    echo "[*] Extracting $label..."
    mkdir -p "$target_dir"
    python - <<PY
import zipfile, sys
with zipfile.ZipFile("$zip_file", 'r') as z:
    z.extractall("$target_dir")
print("    Extraction complete.")
PY
  fi
}

if [ "$SKIP_UNZIP" -eq 0 ]; then
  echo "=== Step 1: Extracting AI-Hub zip archives ==="
  unzip_if_needed "$ZIPS_DIR/train_images.zip" "$AIHUB_DIR/train_data" "Train Images (TIFF)"
  unzip_if_needed "$ZIPS_DIR/train_labels.zip" "$AIHUB_DIR/train_label" "Train Labels (JSON)"
  unzip_if_needed "$ZIPS_DIR/train_meta.zip"   "$AIHUB_DIR/train_meta"  "Train Metadata"
  unzip_if_needed "$ZIPS_DIR/valid_images.zip" "$AIHUB_DIR/valid_data" "Validation Images (TIFF)"
  unzip_if_needed "$ZIPS_DIR/valid_labels.zip" "$AIHUB_DIR/valid_label" "Validation Labels (JSON)"
  unzip_if_needed "$ZIPS_DIR/valid_meta.zip"   "$AIHUB_DIR/valid_meta"  "Validation Metadata"
  echo ""
fi

echo "=== Step 2: Running prepare_aihub (COCO 256px conversion) ==="
exec python -u -m src.prepare_aihub \
  --aihub-root "$AIHUB_DIR" \
  --output "$OUTPUT_DIR"
