#!/bin/bash

# 公共参数
API_KEY="${DEEPSEEK_API_KEY:-YOUR_API_KEY_HERE}"
BASE_URL="https://api.deepseek.com"
API_MODEL="deepseek-chat"
CUDA_DEVICE="cuda:0"
MODEL_NAME="sd1"
MODEL_PATH="/home/user5/models/sd1.5"

INPUT_FOLDER="test"
OUTPUT_FOLDER="test"
OUTPUT_IMG_FOLDER="test/ori_img"
# 确保输出目录存在
mkdir -p "$OUTPUT_FOLDER" "$OUTPUT_IMG_FOLDER"

# 1. decom_prompt.py
PYTHONPATH=. python error_analysis/decom_prompt.py \
  --input_folder "$INPUT_FOLDER" \
  --output_folder "$OUTPUT_FOLDER" \
  --api_key "$API_KEY" \
  --url "$BASE_URL" \
  --api_model "$API_MODEL"

# 2. reform_decomposed_prompt.py
PYTHONPATH=. python utils_all/reform_decomposed_prompt.py \
  --input_file "${OUTPUT_FOLDER}/decomposed_prompt.jsonl"

# 3. generate_image.py
PYTHONPATH=. python error_analysis/generate_image.py \
  --cuda "$CUDA_DEVICE" \
  --input_folder "$INPUT_FOLDER" \
  --model_name "$MODEL_NAME" \
  --model_path "$MODEL_PATH" \
  --api_key "$API_KEY" \
  --url "$BASE_URL" \
  --api_model "$API_MODEL"

# 4. caption.py
PYTHONPATH=. python error_analysis/caption.py \
  --input_folder "$INPUT_FOLDER" \
  --output_folder "$OUTPUT_FOLDER" \
  --api_key "$API_KEY" \
  --url "$BASE_URL" \
  --api_model "$API_MODEL"

# 5. check_captions.py
PYTHONPATH=. python error_analysis/check_captions.py \
  --input_folder "$INPUT_FOLDER" \
  --output_folder "$OUTPUT_FOLDER" \
  --api_key "$API_KEY" \
  --url "$BASE_URL" \
  --api_model "$API_MODEL"

# 6. reform_jsonl.py
PYTHONPATH=. python utils_all/reform_jsonl.py \
  --input_file "${OUTPUT_FOLDER}/check_captions.jsonl"

# 7. question.py
PYTHONPATH=. python error_analysis/question.py \
  --input_folder "$INPUT_FOLDER" \
  --output_folder "$OUTPUT_FOLDER" \
  --api_key "$API_KEY" \
  --url "$BASE_URL" \
  --api_model "$API_MODEL"

# 8. qa.py
PYTHONPATH=. python error_analysis/qa.py \
  --input_folder "$INPUT_FOLDER" \
  --output_folder "$OUTPUT_FOLDER" \
  --api_key "$API_KEY" \
  --url "$BASE_URL" \
  --api_model "$API_MODEL"

# 9. error_integration.py
PYTHONPATH=. python error_analysis/error_integration.py \
  --input_folder "$INPUT_FOLDER" \
  --output_folder "$OUTPUT_FOLDER" \
  --api_key "$API_KEY" \
  --url "$BASE_URL" \
  --api_model "$API_MODEL"

# 10. reform_jsonl.py
PYTHONPATH=. python utils_all/reform_jsonl.py \
  --input_file "${OUTPUT_FOLDER}/errors.jsonl"

# 11. error_mapping.py
PYTHONPATH=. python error_analysis/error_mapping.py \
  --input_folder "$INPUT_FOLDER" \
  --output_folder "$OUTPUT_FOLDER" \
  --api_key "$API_KEY" \
  --url "$BASE_URL" \
  --api_model "$API_MODEL"

# 12. reform_jsonl.py (again)
PYTHONPATH=. python utils_all/reform_jsonl.py \
  --input_file "${OUTPUT_FOLDER}/find_error.jsonl"
