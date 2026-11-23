#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# Reuse the old NLR installation, but never install CUDA dependencies for a CPU preview.
PREPARE=0
for argument in "$@"; do
  case "$argument" in --prepare-only|-h|--help) PREPARE=1 ;; esac
done
if ((!PREPARE)); then
  bash setup_nlr_env.sh
fi
exec python -u -m src.finetune_masking "$@"
