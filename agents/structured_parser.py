import json
from utils_all.api import APIClient


class StructuredParser:
    """
    将自由文本 Prompt 解析为结构化 JSON，再重组为更精确的 Prompt。
    仅需 1 次 LLM 调用。

    输入: "A red apple and a blue car next to a green tree"
    输出: {
        "objects": [
            {"id": 1, "name": "apple", "attributes": {"color": "red"}},
            {"id": 2, "name": "car", "attributes": {"color": "blue"}},
            {"id": 3, "name": "tree", "attributes": {"color": "green"}}
        ],
        "spatial_relations": [
            {"subject": "apple", "relation": "next to", "object": "car"},
            {"subject": "car", "relation": "next to", "object": "tree"}
        ]
    }
    """

    def __init__(self, client: APIClient, template: list[str]):
        self.client = client
        self.template = template

    def parse(self, prompt: str) -> dict:
        textprompt = f"{' '.join(self.template)}\nInput: {prompt}"
        gen_text = self.client.request_gpt(textprompt)
        json_start = gen_text.find("{")
        json_end = gen_text.rfind("}") + 1
        if json_start != -1 and json_end > json_start:
            try:
                structured = json.loads(gen_text[json_start:json_end])
                return structured
            except json.JSONDecodeError:
                pass
        return self._fallback_parse(prompt)

    def _fallback_parse(self, prompt: str) -> dict:
        return {
            "objects": [{"id": 1, "name": prompt, "attributes": {}}],
            "spatial_relations": []
        }

    def reassemble(self, structured: dict) -> str:
        parts = []
        for obj in structured.get("objects", []):
            desc = obj.get("name", "")
            attrs = obj.get("attributes", {})
            attr_parts = [f"{v}" for v in attrs.values() if v]
            if attr_parts:
                desc = " ".join(attr_parts) + " " + desc
            parts.append(desc)
        for rel in structured.get("spatial_relations", []):
            parts.append(f"{rel.get('subject', '')} {rel.get('relation', '')} {rel.get('object', '')}")
        return ", ".join(parts) if parts else ""
