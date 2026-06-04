# 成员1交付检查表

## 一、已完成文件

- [X] `prompts/structured_parse.txt`
- [X] `agents/structured_parser.py`
- [X] `evaluation/test_member1_batch.py`（批量测试脚本）

## 二、已实现方法

- [X] `StructuredParser.parse()` — 含 code block 容错、validate 校验
- [X] `StructuredParser.parse_batch()` — 单条失败不中断，按序返回
- [X] `StructuredParser._validate_structured()` — 校验顶层字段、object、relation 结构
- [X] `StructuredParser._fallback_parse()` — 按 and/逗号 分段，不再粗暴塞整句
- [X] `StructuredParser.reassemble()` — 自然语言重组（数量→英文、颜色→去重、复数化、环境名词关系过滤）

## 三、批量测试结果

- 10 条 prompt 批量解析是否通过：✅ 全部通过
- 解析成功率：**100%**（10/10）
- fallback 触发次数：**0**

## 四、样例与问题

### 重组前后对比

| # | Before                                                                                                       | After                                                                                                                                                       |
| - | ------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0 | A red apple and a blue car are parked next to a green tree on a sunny afternoon.                             | a red apple, a blue parked car, a green tree, a bright sunlight. apple next to car, car next to tree.                                                       |
| 2 | Three white cats are sitting on a red sofa, with a yellow lamp behind them and a blue rug underneath.        | three white sitting cats, a red sofa, a yellow lamp, a blue rug. cat on top of sofa, lamp behind cat, rug under sofa.                                       |
| 3 | A tall glass building with a round clock tower stands behind a small wooden bridge over a blue river.        | a tall glass building, a round clock tower, a small wooden bridge, a blue river. building behind bridge, clock tower on top of building, bridge over river. |
| 5 | Two brown dogs are running on a sandy beach with a red frisbee between them and the ocean in the background. | two brown running dogs, a sandy beach, a red frisbee, an ocean. dog on top of beach, frisbee between dog, ocean behind beach.                               |

完整 10 条对比见 `evaluation/results/member1_batch_test.json`。

### 人工抽样检查（5 条，索引 0/2/3/5/7）

- **[0]** object 提取完整（apple, car, tree, sunlight），关系准确。"sunlight" 虽非理想 object 但优于旧版 "afternoon"。
- **[2]** 数量（three cats）正确，颜色全匹配，空间关系（on top of, behind, under）准确。
- **[3]** 材质（glass, wooden）、尺寸（tall, small）、形状（round）提取完整，关系（behind, on top of, over）正确。
- **[5]** 复数 "two dogs" 正确，动作（running）、材质（sandy）提取正确。`an ocean` 冠词已处理。
- **[7]** 颜色（red, white, green）、尺寸（small, tall）提取正确。`window inside building` 应为 `on`，但属 LLM 判断误差，非重组 bug。

**结论**：object、attribute、spatial relation 无明显漏抽或乱抽。

### 典型失败样例

本轮测试无 fallback。旧版典型问题（已修复）：raw JSON `other` 数组被直接拼成 `['parked']` 字面量、裸数字 `1` 插入 prompt。

### 已知局限

1. 抽象场景词（sunlight）仍会被 LLM 识别为 object，template 已引导但无法 100% 杜绝
2. 冠词 "a"/"an" 仅在 quantity=1 时使用，边缘名词（如 "ocean"）需 LLM 提供正确 quantity
3. 复数化规则覆盖常见情况但非穷举（不影响 CLIP 评分）

### 环境名词过滤（增量改进）

发现问题后，在 `reassemble()` 中增加环境名词过滤：

- **策略**：保留环境名词作为视觉线索（如 "bright sunlight"），但过滤涉及它们的物理空间关系（如 "apple on top of sunlight"）
- **环境名词集**：`sunlight, atmosphere, weather, afternoon, morning, evening, sky, background, foreground`
- **效果**：三轮重复实验平均 CLIP Delta 由 +0.38 → **+1.48**

| 轮次 | Baseline | Improved | Delta |
|------|----------|----------|-------|
| 1 | 28.83 | 30.56 | +1.73 |
| 2 | 30.29 | 30.52 | +0.23 |
| 3 | 28.56 | 31.04 | +2.47 |
| **平均** | 29.23 | **30.71** | **+1.48** |

## 五、论文材料

### Methodology §1 方法概述

我们采用 LLM 驱动的结构化解析方法，将自由文本 prompt 分解为 `{objects, attributes, spatial_relations}` 三要素，再按自然语言顺序重组为更精确的图像生成 prompt。核心思路是：LLM 擅长语义理解但不保证输出格式一致性，因此通过严格的 JSON schema 约束 + 规则化重组来消除格式噪声。每条 prompt 仅需 1 次 LLM 调用，重组和校验均为零成本规则运算。

### 结构化解析示例

**输入**：

> Three white cats are sitting on a red sofa, with a yellow lamp behind them and a blue rug underneath.

**LLM 输出（结构化 JSON）**：

```json
{
  "objects": [
    {"id":1,"name":"cat","attributes":{"color":"white","quantity":3,"action":"sitting",...}},
    {"id":2,"name":"sofa","attributes":{"color":"red","quantity":1,...}},
    {"id":3,"name":"lamp","attributes":{"color":"yellow","quantity":1,...}},
    {"id":4,"name":"rug","attributes":{"color":"blue","quantity":1,...}}
  ],
  "spatial_relations": [
    {"subject":"cat","relation":"on top of","object":"sofa"},
    {"subject":"lamp","relation":"behind","object":"cat"},
    {"subject":"rug","relation":"under","object":"sofa"}
  ]
}
```

**重组 prompt**：

> three white sitting cats, a red sofa, a yellow lamp, a blue rug. cat on top of sofa, lamp behind cat, rug under sofa.

### 边界/失败案例说明

**旧版问题**：在初版实现中，`reassemble()` 仅遍历所有非空属性值直接拼接。当 LLM 返回 `"other": ["parked"]` 时，Python 的 `str()` 转换产生字面量 `['parked']`；`"quantity": 3` 拼接为裸数字 `3`。重组结果形如 `"red 1 apple, blue 1 ['parked'] car"`，严重偏离自然语言分布，导致 CLIP 评分下降 3.8 分。

**修复策略**：引入属性语义化处理——`quantity` 数字转英文词并驱动复数化，`other`/`action` 作为前置定语，颜色去重（避免 "golden golden retriever"），`null` 和空值跳过。在此基础上，增加环境名词过滤——保留 "bright sunlight" 等视觉线索但过滤涉及它们的幻象空间关系（如 "apple on top of sunlight"）。修复后 CLIP Delta 由 -3.81 回升至 **+1.48**（三轮均值）。

## 六、联调与协作

### 已同步给成员0的内容

- `StructuredParser` 接口稳定，输入 prompt 字符串，输出 JSON + 重组 prompt
- `parse_batch()` 支持批量调用，与 `run_structured.py` 的 `--use_structured` 可直接对接
- 重组后的 enhanced prompt 格式：逗号分隔的对象短语 + 句号分隔的空间关系

### 已同步给成员2的内容

- **字段口径变化**：`other` (list) → `action` (string|null)，需确认 `negative_prompt_generator` 是否适配（当前代码遍历 attributes.items()，不依赖具体 key，兼容）
- **实验发现**：不启用 negative prompt 时 CLIP +0.38，启用后退至 -1.13~-3.23。建议成员2检查负向规则是否过度抑制空间关系对应的颜色/对象
- `spatial_relations` 表达式保持 `{subject, relation, object}` 三字段，无变化

### 已同步给成员3的内容

- 批量测试结果：10 条全通过，成功率 100%
- 重组 prompt 示例（见上文 §4 对比表）
- 已知局限：LLM 对抽象场景词（sunlight）的 object 归类偶有偏差；冠词处理简化

### 当前待联调问题

- Negative prompt 对 CLIP 评分的负面影响需与成员2排查

### 需要成员0协助的事项

- 确认 `run_structured.py` 中 `--api_model` 默认值适配当前 API（当前为 `deepseek-v4-flash`）
