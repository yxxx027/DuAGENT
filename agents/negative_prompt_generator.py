import json


class NegativePromptGenerator:
    """
    基于结构化解析结果，规则生成 Negative Prompt。
    无需 LLM 调用。

    策略：
    1. 对每个属性生成否定（如 color=red → "green, blue, yellow"）
    2. 对每个空间关系生成否定（如 "left of" → "right of, behind"）
    3. 通用否定词（blurry, deformed, extra limbs...）
    """

    def __init__(self, rules_path: str = None):
        if rules_path is None:
            import os
            rules_path = os.path.join(os.path.dirname(__file__), "negative_rules.json")
        with open(rules_path, "r", encoding="utf-8") as f:
            self.rules = json.load(f)

    def generate(self, structured: dict) -> str:
        negations = []
        negations.extend(self.generate_attribute_negations(structured.get("objects", [])))
        negations.extend(self.generate_spatial_negations(structured.get("spatial_relations", [])))
        negations.extend(self.rules.get("general_negations", []))
        seen = set()
        unique = []
        for n in negations:
            if n.lower() not in seen:
                seen.add(n.lower())
                unique.append(n)
        return ", ".join(unique)

    def generate_attribute_negations(self, objects: list[dict]) -> list[str]:
        negations = []
        attr_rules = self.rules.get("attribute_negations", {})
        for obj in objects:
            for attr_key, attr_val in obj.get("attributes", {}).items():
                if not attr_val:
                    continue
                key_lower = attr_key.lower()
                if key_lower in attr_rules:
                    negations.extend(attr_rules[key_lower])
                else:
                    val_lower = str(attr_val).lower()
                    for rule_key, rule_vals in attr_rules.items():
                        if val_lower in rule_vals or val_lower in rule_key:
                            negations.extend([v for v in rule_vals if v.lower() != val_lower])
                            break
        return negations

    def generate_spatial_negations(self, relations: list[dict]) -> list[str]:
        negations = []
        spatial_rules = self.rules.get("spatial_negations", {})
        for rel in relations:
            relation = rel.get("relation", "").lower()
            for rule_key, rule_vals in spatial_rules.items():
                if rule_key in relation or relation in rule_key:
                    negations.extend(rule_vals)
                    break
        return negations
