"""成员1批量解析测试脚本。输出统计、样例、重组前后对比。"""
import json
import sys
from agents.structured_parser import StructuredParser
from utils_all.api import APIClient

template = open("prompts/structured_parse.txt").readlines()
client = APIClient(
    api_key="sk-a1e95a020c3e4f3eb9aaea6ddadc95a3",
    url="https://api.deepseek.com",
    api_name="deepseek-chat",
)
sp = StructuredParser(client, template)

prompts = [l.strip() for l in open("evaluation/test_prompts.txt") if l.strip()]
print(f"Testing {len(prompts)} prompts...\n")

success = 0
fallback = 0
records = []

for i, p in enumerate(prompts):
    result = sp.parse(p)
    objs = result.get("objects", [])
    rels = result.get("spatial_relations", [])
    reassembled = sp.reassemble(result)

    # 判定 fallback：单个 object 且 name 过长（≥60 char 说明是整句塞入）
    is_fallback = len(objs) == 1 and len(objs[0].get("name", "")) >= 60

    if is_fallback:
        fallback += 1
        status = "FALLBACK"
    else:
        success += 1
        status = "OK"

    records.append({
        "index": i,
        "status": status,
        "original": p,
        "structured": result,
        "reassembled": reassembled,
    })

    print(f"[{i}] {status} | objects={len(objs)} relations={len(rels)}")
    print(f"    Original:    {p[:120]}")
    print(f"    Reassembled: {reassembled[:200]}")
    print()

# ── 汇总 ──
print(f"{'='*60}")
print(f"Summary")
print(f"  Success:  {success}/{len(prompts)}")
print(f"  Fallback: {fallback}/{len(prompts)}")
print(f"  Rate:     {success/len(prompts)*100:.0f}%")
print()

# ── 重组前后对比 ──
print("=== Before / After (first 5) ===")
for r in records[:5]:
    print(f"[{r['index']}] {r['status']}")
    print(f"  Before: {r['original'][:150]}")
    print(f"  After:  {r['reassembled'][:200]}")
    print()

# ── 失败样例 ──
failed = [r for r in records if r["status"] == "FALLBACK"]
if failed:
    print("=== Failure Cases ===")
    for r in failed:
        print(f"[{r['index']}] Original: {r['original'][:150]}")
        print(f"  Raw result: {json.dumps(r['structured'], ensure_ascii=False)[:300]}")
        print()
else:
    print("=== Failure Cases: NONE ===")

# ── 输出 JSON 供文档引用 ──
with open("output/member1_batch_test.json", "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)
print("Full results saved to output/member1_batch_test.json")
