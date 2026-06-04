#!/usr/bin/env bash
set -Eeuo pipefail

CONFIG_FILE="${CONFIG_FILE:-run_config.example.json}"
RESUME_RUN="${RESUME_RUN:-}"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

command -v python >/dev/null 2>&1 || fail "python is not available"
if [[ -n "$RESUME_RUN" ]]; then
  [[ -d "$RESUME_RUN" ]] || fail "resume directory not found: $RESUME_RUN"
  CONFIG_FILE="$RESUME_RUN/run_config.json"
fi
[[ -f "$CONFIG_FILE" ]] || fail "config file not found: $CONFIG_FILE"

MODEL_PATH="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["model_path"])' "$CONFIG_FILE")"
USE_STRUCTURED="$(python -c 'import json,sys; print(str(json.load(open(sys.argv[1])).get("use_structured", False)).lower())' "$CONFIG_FILE")"
OUTPUT_DIR="$(python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("output_dir", "output"))' "$CONFIG_FILE")"

[[ -d "$MODEL_PATH" ]] || fail "model directory not found: $MODEL_PATH"
[[ -f "$MODEL_PATH/model_index.json" ]] || fail "model_index.json not found in: $MODEL_PATH"
python -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'" || fail "PyTorch CUDA check failed"
if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "WARNING: no active Conda environment detected" >&2
else
  echo "Conda:  $CONDA_PREFIX"
fi

if [[ "$USE_STRUCTURED" == "true" && -z "${DEEPSEEK_API_KEY:-}" ]]; then
  fail "DEEPSEEK_API_KEY is required when use_structured=true"
fi

ARGS=(--config "$CONFIG_FILE")
if [[ -n "$RESUME_RUN" ]]; then
  ARGS+=(--resume_run "$RESUME_RUN")
  RUN_DIR="$RESUME_RUN"
else
  RUN_NAME="${RUN_NAME:-run_$(date +%Y%m%d%H%M%S)}"
  ARGS+=(--run_name "$RUN_NAME")
  RUN_DIR="$OUTPUT_DIR/$RUN_NAME"
fi

echo "Config: $CONFIG_FILE"
echo "Model:  $MODEL_PATH"
echo "Run:    $RUN_DIR"

PYTHONPATH=. python run_structured.py "${ARGS[@]}"
python -c 'import json,sys,pathlib; p=pathlib.Path(sys.argv[1]); c=json.load(open(p/"checkpoint.json")); assert c["status"]=="completed"; n=len(c["prompts"]); assert len(list((p/"baseline_images").glob("*.png")))==n; assert len(list((p/"improved_images").glob("*.png")))==n; assert (p/"results.json").is_file(); print(f"Verified completed run with {n} prompts: {p}")' "$RUN_DIR"
