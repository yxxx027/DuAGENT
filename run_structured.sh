#!/bin/bash

API_KEY="${DEEPSEEK_API_KEY:-YOUR_API_KEY_HERE}"
BASE_URL="https://api.deepseek.com"
API_MODEL="deepseek-chat"
CUDA_DEVICE="cuda:0"
MODEL_NAME="sd1"
MODEL_PATH="/home/user5/models/sd1.5"
INPUT_FILE="evaluation/test_prompts.txt"
OUTPUT_DIR="output"

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo " Step 1: Baseline Image Generation Only"
echo "=========================================="
PYTHONPATH=. python run_structured.py \
  --input_file "$INPUT_FILE" \
  --cuda "$CUDA_DEVICE" \
  --model_name "$MODEL_NAME" \
  --model_path "$MODEL_PATH" \
  --output_dir "$OUTPUT_DIR"

echo ""
echo "=========================================="
echo " Step 2: With Structured + Negative Prompt"
echo "=========================================="
PYTHONPATH=. python run_structured.py \
  --input_file "$INPUT_FILE" \
  --cuda "$CUDA_DEVICE" \
  --model_name "$MODEL_NAME" \
  --model_path "$MODEL_PATH" \
  --api_key "$API_KEY" \
  --url "$BASE_URL" \
  --api_model "$API_MODEL" \
  --use_structured \
  --use_negative \
  --output_dir "$OUTPUT_DIR"

echo ""
echo "=========================================="
echo " Step 3: Full Pipeline with CLIP Scoring"
echo "=========================================="
PYTHONPATH=. python run_structured.py \
  --input_file "$INPUT_FILE" \
  --cuda "$CUDA_DEVICE" \
  --model_name "$MODEL_NAME" \
  --model_path "$MODEL_PATH" \
  --api_key "$API_KEY" \
  --url "$BASE_URL" \
  --api_model "$API_MODEL" \
  --use_structured \
  --use_negative \
  --use_clip \
  --clip_device "cuda:1" \
  --output_dir "$OUTPUT_DIR"

echo ""
echo "All steps completed!"
