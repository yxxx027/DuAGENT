# GenPilotLight — 项目初始化和 Baseline 运行指南

***

## 一、项目概述

GenPilotLight 是基于 GenPilot（EMNLP 2025 Findings）的轻量改进项目，核心思路：

1. **结构化 Prompt 解析**：用 LLM 将自由文本解析为 `{objects, attributes, spatial_relations}` JSON，再重组为更精确的 Prompt
2. **Negative Prompt 生成**：基于结构化结果，规则生成"不要画什么"，SD1.5 原生支持
3. **CLIP 评测**：本地 CLIP 模型自动评分，替代需要 VLM 的评分机制

**LLM 调用量**：每条 Prompt 仅 1 次（结构化解析），Negative Prompt 由规则生成无需 LLM。

***

## 二、环境搭建

### 2.1 创建 Conda 环境

```bash
conda create -n genpilot python=3.12
conda activate genpilot
```

### 2.2 安装依赖

```bash
# 用 conda 安装需要编译的包（服务器 GCC 版本低时必须如此）
conda install -y -c conda-forge numpy scipy scikit-learn cmake compilers sentencepiece

# 安装 PyTorch（CUDA 12.2 兼容 cu121）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装其余依赖
pip install accelerate annotated-types anyio diffusers distro h11 httpcore httpx \
  huggingface-hub importlib_metadata jiter joblib packaging protobuf psutil pydantic \
  regex safetensors setuptools sniffio sympy tenacity threadpoolctl tokenizers tqdm \
  typing-inspection openai pillow transformers matplotlib
```

### 2.3 确认 GPU 可用

```bash
nvidia-smi
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| Devices:', torch.cuda.device_count())"
```

**预期输出**：`CUDA: True | Devices: 4`

***

## 三、初始化验证（7 项全过才算成功）

### 3.1 模块导入验证

```bash
cd ~/GenPilot-main
PYTHONPATH=. python -c "
from agents.structured_parser import StructuredParser
from agents.negative_prompt_generator import NegativePromptGenerator
from evaluation.clip_scorer import CLIPScorer
from evaluation.visualize_results import plot_score_comparison_bar
from utils_all.api import APIClient
print('All modules imported successfully!')
"
```

**通过标准**：输出 `All modules imported successfully!`

### 3.2 Negative Prompt 生成验证

```bash
PYTHONPATH=. python -c "
from agents.negative_prompt_generator import NegativePromptGenerator
npg = NegativePromptGenerator()
test = {
    'objects': [
        {'id': 1, 'name': 'apple', 'attributes': {'color': 'red'}},
        {'id': 2, 'name': 'car', 'attributes': {'color': 'blue'}}
    ],
    'spatial_relations': [
        {'subject': 'apple', 'relation': 'next to', 'object': 'car'}
    ]
}
neg = npg.generate(test)
print(f'Negative Prompt: {neg}')
assert len(neg) > 0, 'Empty!'
print('Negative prompt generation: OK')
"
```

**通过标准**：输出非空的否定词列表

### 3.3 SD1.5 图像生成验证

```bash
PYTHONPATH=. python -c "
import torch
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained('/home/user5/models/sd1.5', torch_dtype=torch.float16, local_files_only=True)
pipe.to('cuda:0')
img = pipe('a red apple', width=512, height=512, num_inference_steps=20).images[0]
img.save('/tmp/test_sd15.png')
print('SD1.5 image generation: OK')
"
```

**通过标准**：输出 `SD1.5 image generation: OK`，`/tmp/test_sd15.png` 存在

### 3.4 CLIP 评分验证

```bash
PYTHONPATH=. python -c "
from evaluation.clip_scorer import CLIPScorer
scorer = CLIPScorer(device='cuda:1')
score = scorer.score('a red apple', '/tmp/test_sd15.png')
print(f'CLIP Score: {score:.4f}')
assert score > 0, 'CLIP score is 0!'
print('CLIP scoring: OK')
"
```

**通过标准**：输出正浮点数（如 `15.xxxx`）

### 3.5 DeepSeek API 连通验证

```bash
python -c "
from openai import OpenAI
c = OpenAI(api_key='你的DeepSeek密钥', base_url='https://api.deepseek.com')
r = c.chat.completions.create(model='deepseek-chat', messages=[{'role':'user','content':'Say hi'}])
print('API Response:', r.choices[0].message.content)
print('DeepSeek API: OK')
"
```

**通过标准**：返回正常文本响应

### 3.6 Baseline 图像生成验证

```bash
PYTHONPATH=. python run_structured.py \
  --input_file evaluation/test_prompts.txt \
  --cuda cuda:0 --model_name sd1 --model_path /home/user5/models/sd1.5 \
  --output_dir output
```

**通过标准**：`output/run_*/baseline_images/` 下有 10 张 PNG 图像

### 3.7 改进版图像生成验证

```bash
PYTHONPATH=. python run_structured.py \
  --input_file evaluation/test_prompts.txt \
  --cuda cuda:0 --model_name sd1 --model_path /home/user5/models/sd1.5 \
  --api_key 你的DeepSeek密钥 --url https://api.deepseek.com --api_model deepseek-chat \
  --use_structured --use_negative \
  --output_dir output
```

**通过标准**：`output/run_*/improved_images/` 下有 10 张 PNG 图像

***

## 四、初始化成功判定清单

| 序号 | 验证项 | 通过标准 |
|------|--------|----------|
| 1 | 模块导入 | 无报错 |
| 2 | Negative Prompt 生成 | 生成非空否定词列表 |
| 3 | SD1.5 图像生成 | 生成测试图像 |
| 4 | CLIP 评分 | 返回正浮点数 |
| 5 | DeepSeek API | 返回正常响应 |
| 6 | Baseline 图像 | 10 张 PNG |
| 7 | 改进版图像 | 10 张 PNG |

**全部 7 项通过 = 初始化成功 ✅**

***

## 五、完整管线运行

### 5.1 逐步运行

```bash
# Step 1: Baseline（无需 LLM）
PYTHONPATH=. python run_structured.py \
  --input_file evaluation/test_prompts.txt \
  --cuda cuda:0 --model_name sd1 --model_path /home/user5/models/sd1.5 \
  --output_dir output

# Step 2: 结构化 + Negative Prompt
PYTHONPATH=. python run_structured.py \
  --input_file evaluation/test_prompts.txt \
  --cuda cuda:0 --model_name sd1 --model_path /home/user5/models/sd1.5 \
  --api_key 你的DeepSeek密钥 --url https://api.deepseek.com --api_model deepseek-chat \
  --use_structured --use_negative \
  --output_dir output

# Step 3: 完整管线（含 CLIP 评分）
PYTHONPATH=. python run_structured.py \
  --input_file evaluation/test_prompts.txt \
  --cuda cuda:0 --model_name sd1 --model_path /home/user5/models/sd1.5 \
  --api_key 你的DeepSeek密钥 --url https://api.deepseek.com --api_model deepseek-chat \
  --use_structured --use_negative --use_clip --clip_device cuda:1 \
  --output_dir output
```

### 5.2 一键运行

```bash
bash run_structured.sh
```

> ⚠️ 运行前需编辑 `run_structured.sh`，填入你的 DeepSeek API Key。

***

## 六、输出目录结构

```
output/
└── run_20260517XXXXXX/
    ├── baseline_images/          # 原始 Prompt 生成的图像
    ├── improved_images/          # 结构化+负向 Prompt 生成的图像
    ├── structured_results.json   # 结构化解析结果
    ├── negative_prompts.json     # Negative Prompt 列表
    ├── results.json              # 完整结果（含 CLIP 分数）
    ├── score_comparison.png      # CLIP 分数对比条形图
    └── report_table.md           # Markdown 格式结果表格
```

***

## 七、常见问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| `ModuleNotFoundError: No module named 'agents'` | 未设置 PYTHONPATH 或不在项目目录 | `cd ~/GenPilot-main && PYTHONPATH=.` |
| `CUDA out of memory` | GPU 显存不足 | 换 `--cuda cuda:1` 或释放 GPU |
| `model_index.json not found` | 模型路径错误 | 检查 `--model_path` |
| CLIP 下载慢 | HuggingFace 网络问题 | `export HF_ENDPOINT=https://hf-mirror.com` |
| DeepSeek API 429 | 速率限制 | 增加重试间隔或减少并发 |
| `Negative prompt is empty` | `negative_rules.json` 未找到 | 确认 `agents/negative_rules.json` 存在 |
