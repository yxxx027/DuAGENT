# 首次完整管线运行记录

**日期**：2026-05-23
**实例**：ModelScope GPU 实例
**环境**：Python 3.11 venv，PyTorch 2.9.1+cu128，diffusers 0.37.0，transformers 4.57.6

## 运行命令

```bash
cd /mnt/workspace/GenPilot-light
PYTHONPATH=. python run_structured.py \
  --input_file evaluation/test_prompts.txt \
  --cuda cuda:0 --model_name sd1 --model_path /mnt/workspace/models/sd1.5 \
  --api_key <key> --url https://api.deepseek.com --api_model deepseek-chat \
  --use_structured --use_negative --use_clip --clip_device cuda:0 \
  --clip_model_path /mnt/workspace/models/clip \
  --output_dir output
```

## 结果

| 指标 | 值 |
|------|-----|
| Avg Baseline CLIP | 29.64 |
| Avg Improved CLIP | 25.84 |
| Delta | **-3.81** |

Improved 反而低于 Baseline。

## 诊断

根因在 `agents/structured_parser.py` 的 `reassemble()` 方法。

LLM 返回的结构化 JSON 字段丰富（`quantity`、`other`、`material`、`shape` 等），但 `reassemble()` 只是把所有非空值盲目拼接，导致：

- `other: ["parked"]` → 拼成 Python list 字面量 `['parked']`
- `quantity: 1` → 拼成裸数字 `1`
- `quantity: 3` → 仍然是 `3` 而非 "three"

重组后的 prompt 形如 `"red 1 apple, blue 1 ['parked'] car, green 1 tree, 1 ['sunny'] afternoon..."`，损害了自然语言质量，SD1.5 无法理解。

## 待修复

- `reassemble()` 需自然语言化重组
- `prompts/structured_parse.txt` 需收紧输出协议
