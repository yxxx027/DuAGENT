import json
import re
from utils_all.api import APIClient


class StructuredParser:
    """
    将自由文本 Prompt 解析为结构化 JSON，再重组为更精确的 Prompt。
    仅需 1 次 LLM 调用。

    输入: "A red apple and a blue car next to a green tree"
    输出: {
        "objects": [
            {"id": 1, "name": "apple", "attributes": {"color": "red", ...}},
            {"id": 2, "name": "car", "attributes": {"color": "blue", ...}},
            {"id": 3, "name": "tree", "attributes": {"color": "green", ...}}
        ],
        "spatial_relations": [
            {"subject": "apple", "relation": "next to", "object": "car"},
            {"subject": "car", "relation": "next to", "object": "tree"}
        ]
    }
    """

    _NUMBER_WORDS = {
        1: "a", 2: "two", 3: "three", 4: "four", 5: "five",
        6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
    }

    @staticmethod
    def _pluralize(name: str, qty: int) -> str:
        if qty <= 1:
            return name
        # already plural
        if name.endswith("s") and not name.endswith(("ss", "us", "is")):
            return name
        if name.endswith(("s", "sh", "ch", "x", "z", "o")):
            return name + "es"
        if name.endswith("y") and len(name) > 1 and name[-2] not in "aeiou":
            return name[:-1] + "ies"
        return name + "s"

    def __init__(self, client: APIClient, template: list[str]):
        self.client = client
        self.template = template

    # ── parse ──────────────────────────────────────────────

    def parse(self, prompt: str) -> dict:
        textprompt = f"{' '.join(self.template)}\nInput: {prompt}"
        gen_text = self.client.request_gpt(textprompt)

        # Try code block first, then raw JSON extraction
        json_str = gen_text
        code_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', gen_text, re.DOTALL)
        if code_match:
            json_str = code_match.group(1).strip()
        else:
            json_start = gen_text.find("{")
            json_end = gen_text.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                json_str = gen_text[json_start:json_end]

        try:
            structured = json.loads(json_str)
            is_valid, warnings = self._validate_structured(structured)
            if not is_valid:
                print(f"Validation failed: {'; '.join(warnings)}")
                return self._fallback_parse(prompt)
            return structured
        except json.JSONDecodeError:
            return self._fallback_parse(prompt)

    def parse_batch(self, prompts: list[str]) -> list[dict]:
        results = []
        for i, prompt in enumerate(prompts):
            try:
                results.append(self.parse(prompt))
            except Exception as e:
                print(f"Batch parse failed for prompt {i}: {e}")
                results.append(self._fallback_parse(prompt))
        return results

    # ── reassemble ─────────────────────────────────────────

    def reassemble(self, structured: dict) -> str:
        """
        将结构化 JSON 重组为自然语言 prompt。
        跳过 null/空值，quantity 数字转英文，action 作为定语。
        """
        object_phrases = []
        for obj in structured.get("objects", []):
            name = obj.get("name", "")
            if not name:
                continue
            attrs = obj.get("attributes", {})

            parts = []

            # quantity → word
            qty = attrs.get("quantity") or 1
            qty_word = self._NUMBER_WORDS.get(qty, str(qty))
            parts.append(qty_word)

            # size
            size = attrs.get("size")
            if size:
                parts.append(size)

            # shape
            shape = attrs.get("shape")
            if shape:
                parts.append(shape)

            # color (skip if already in name, e.g. "golden retriever")
            color = attrs.get("color")
            if color and color.lower() not in name.lower():
                parts.append(color)

            # material
            material = attrs.get("material")
            if material:
                parts.append(material)

            # action / other (backward compat)
            action = attrs.get("action") or attrs.get("other")
            if action:
                if isinstance(action, list):
                    action = " ".join(str(a) for a in action if a)
                if action:
                    parts.append(str(action))

            # pluralize name when quantity > 1
            final_name = self._pluralize(name, qty)
            parts.append(final_name)
            object_phrases.append(" ".join(parts))

        # spatial relations
        relation_phrases = []
        for rel in structured.get("spatial_relations", []):
            subj = rel.get("subject", "")
            relation = rel.get("relation", "")
            obj = rel.get("object", "")
            if subj and relation and obj:
                relation_phrases.append(f"{subj} {relation} {obj}")

        result = ", ".join(object_phrases)
        if relation_phrases:
            result += ". " + ", ".join(relation_phrases) + "."
        return result

    # ── validate ───────────────────────────────────────────

    def _validate_structured(self, structured: dict) -> tuple:
        """返回 (is_valid, warnings)。仅顶层字段缺失视为 invalid。"""
        warnings = []
        if not isinstance(structured, dict):
            return False, ["structured result is not a dict"]
        if "objects" not in structured or not isinstance(structured["objects"], list):
            return False, ["missing or invalid 'objects' field"]
        if "spatial_relations" not in structured:
            structured["spatial_relations"] = []

        for obj in structured["objects"]:
            for key in ("id", "name"):
                if key not in obj:
                    warnings.append(f"object missing '{key}': {obj.get('name', str(obj)[:40])}")

        for rel in structured.get("spatial_relations", []):
            for key in ("subject", "relation", "object"):
                if key not in rel:
                    warnings.append(f"relation missing '{key}': {rel}")

        # objects 有条目就视为 valid；仅有 warning 时仍然可用
        return len(structured["objects"]) > 0, warnings

    # ── fallback ───────────────────────────────────────────

    def _fallback_parse(self, prompt: str) -> dict:
        """
        解析失败时的降级策略。
        按 and / 逗号 / with 等连词分段，每段作为一个基础 object。
        """
        chunks = re.split(r',\s*|\s+and\s+|\s+with\s+|\s+while\s+', prompt)
        objects = []
        for i, chunk in enumerate(chunks):
            chunk = chunk.strip().rstrip('.')
            if not chunk:
                continue
            chunk = re.sub(r'^(a|an|the)\s+', '', chunk, flags=re.IGNORECASE)
            objects.append({
                "id": i + 1,
                "name": chunk,
                "attributes": {
                    "color": None, "material": None, "shape": None,
                    "size": None, "quantity": 1, "action": None,
                }
            })
        return {"objects": objects, "spatial_relations": []}
