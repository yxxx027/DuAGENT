import json
import os


# 优先级定义：数字越小优先级越高（越先被保留）
PRIORITY_MAP = {
    "attribute": 0,
    "spatial": 1,
    "style": 2,
    "context": 3,
    "quantity": 4,
    "general": 5,
}


def _estimate_tokens(words):
    """估算一个单词列表的 token 数量（近似：以空格分词的单词数）。"""
    if not words:
        return 0
    joined = ", ".join(words)
    # 近似：以空格分词后数量
    return len(joined.split())


class NegativePromptGenerator:
    """
    基于结构化解析结果，规则生成 Negative Prompt。
    无需 LLM 调用。

    策略：
    1. 对每个属性生成否定（如 color=red -> "green, blue, yellow"）
    2. 对每个空间关系生成否定（如 "left of" -> "right of, behind"）
    3. 对风格/上下文/数量分别生成否定
    4. 通用否定词
    5. 去重，按优先级剪枝到 <=75 tokens
    """

    def __init__(self, rules_path=None):
        if rules_path is None:
            rules_path = os.path.join(
                os.path.dirname(__file__), "negative_rules.json"
            )
        with open(rules_path, "r", encoding="utf-8") as f:
            self.rules = json.load(f)

    def generate(self, structured):
        """
        基于完整的结构化 JSON 生成 Negative Prompt 字符串。

        Args:
            structured (dict): 结构化解析结果，可包含 objects、spatial_relations、
                style、mood、context、setting、background、quantity_error 等字段。

        Returns:
            str: 英文逗号分隔的 Negative Prompt（例如 "blue, green, cartoon, ..."）。

        Example:
            >>> npg = NegativePromptGenerator()
            >>> structured = {
            ...     "objects": [{"name":"apple","attributes":{"color":"red"}}],
            ...     "spatial_relations": [{"subject":"apple","relation":"left of","object":"car"}],
            ...     "style": "realistic",
            ...     "context": "indoor"
            ... }
            >>> prompt = npg.generate(structured)
            >>> isinstance(prompt, str) and len(prompt) > 0
            True
        """
        if not structured or not isinstance(structured, dict):
            structured = {}

        negations = []
        # 1. 属性否定
        negations.extend(
            self.generate_attribute_negations(structured.get("objects", []))
        )
        # 2. 空间否定
        negations.extend(
            self.generate_spatial_negations(structured.get("spatial_relations", []))
        )
        # 3. 风格否定
        negations.extend(self.generate_style_negations(structured))
        # 4. 上下文否定
        negations.extend(self.generate_context_negations(structured))
        # 5. 数量否定
        negations.extend(self.generate_quantity_negations(structured))
        # 6. 通用否定
        negations.extend(self.rules.get("general_negations", []))

        # 7. 去重（大小写不敏感，保持首次出现顺序）
        seen = set()
        unique = []
        for n in negations:
            if not isinstance(n, str) or not n.strip():
                continue
            key = n.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(n.strip())

        # 8. 剪枝
        pruned = self.prune_negations(unique, max_tokens=75)
        return ", ".join(pruned)

    def generate_attribute_negations(self, objects):
        """
        基于物体属性生成否定词。

        Args:
            objects (list[dict]): 物体列表，每个 dict 含 "attributes" 字段。

        Returns:
            list[str]: 否定词列表。
        """
        if not objects:
            return []
        negations = []
        attr_rules = self.rules.get("attribute_negations", {})
        if not attr_rules:
            return []

        for obj in objects:
            if not isinstance(obj, dict):
                continue
            attributes = obj.get("attributes", {}) or {}
            for attr_key, attr_val in attributes.items():
                if attr_val is None or (isinstance(attr_val, str) and not attr_val.strip()):
                    continue
                key_lower = attr_key.lower() if isinstance(attr_key, str) else ""
                val_lower = str(attr_val).lower().strip()

                # 优先按属性类别查找（如 "color" -> general color negations）
                if key_lower in attr_rules and isinstance(attr_rules[key_lower], list):
                    negations.extend(attr_rules[key_lower])

                # 再按具体属性值查找（如 "red" -> blue/green/...）
                if val_lower in attr_rules and isinstance(attr_rules[val_lower], list):
                    negations.extend(attr_rules[val_lower])
                else:
                    # 包含式匹配
                    for rule_key, rule_vals in attr_rules.items():
                        if isinstance(rule_vals, list) and (
                            val_lower in rule_key or rule_key in val_lower
                        ):
                            negations.extend(
                                [v for v in rule_vals if v.lower() != val_lower]
                            )
                            break
        return negations

    def generate_spatial_negations(self, relations):
        """
        基于空间关系生成否定词。

        Args:
            relations (list[dict]): 每个 dict 含 "relation" 字段。

        Returns:
            list[str]: 否定词列表。
        """
        if not relations:
            return []
        negations = []
        spatial_rules = self.rules.get("spatial_negations", {})
        if not spatial_rules:
            return []

        for rel in relations:
            if not isinstance(rel, dict):
                continue
            relation = str(rel.get("relation", "")).lower().strip()
            if not relation:
                continue
            # 精确匹配优先
            if relation in spatial_rules:
                negations.extend(spatial_rules[relation])
                continue
            # 包含式匹配
            matched = False
            for rule_key, rule_vals in spatial_rules.items():
                if rule_key in relation or relation in rule_key:
                    negations.extend(rule_vals)
                    matched = True
                    break
        return negations

    def generate_style_negations(self, structured):
        """
        基于风格描述生成否定词。

        Args:
            structured (dict): 结构化结果。支持顶层 style/mood，
                以及 objects[*].attributes.style。

        Returns:
            list[str]: 否定词列表。
        """
        if not structured or not isinstance(structured, dict):
            return []

        style_rules = self.rules.get("style_negations", {})
        if not style_rules:
            return []

        negations = []
        # 顶层风格
        for key in ("style", "mood"):
            val = structured.get(key)
            if isinstance(val, str) and val.strip():
                self._match_into_negations(val.strip(), style_rules, negations)
        # 对象层风格
        for obj in structured.get("objects", []) or []:
            if not isinstance(obj, dict):
                continue
            val = (obj.get("attributes") or {}).get("style")
            if isinstance(val, str) and val.strip():
                self._match_into_negations(val.strip(), style_rules, negations)
        return negations

    def generate_context_negations(self, structured):
        """
        基于场景描述生成否定词。

        Args:
            structured (dict): 结构化结果。支持 context/setting/background。

        Returns:
            list[str]: 否定词列表。
        """
        if not structured or not isinstance(structured, dict):
            return []

        context_rules = self.rules.get("context_negations", {})
        if not context_rules:
            return []

        parts = []
        for key in ("context", "setting", "background"):
            val = structured.get(key)
            if isinstance(val, str) and val.strip():
                parts.append(val.strip())
        combined = " ".join(parts).lower()
        if not combined:
            return []

        negations = []
        for rule_key, rule_vals in context_rules.items():
            if rule_key in combined:
                negations.extend(rule_vals)
        return negations

    def generate_quantity_negations(self, structured):
        """
        基于数量描述生成否定词。

        Args:
            structured (dict): 结构化结果。支持顶层 quantity_error，
                以及 objects[*].attributes.quantity。

        Returns:
            list[str]: 否定词列表。
        """
        if not structured or not isinstance(structured, dict):
            return []

        quantity_rules = self.rules.get("quantity_negations", {})
        if not quantity_rules:
            return []

        parts = []
        val = structured.get("quantity_error")
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())
        # 从每个 object 的 quantity 属性推断
        for obj in structured.get("objects", []) or []:
            if not isinstance(obj, dict):
                continue
            qv = (obj.get("attributes") or {}).get("quantity")
            if qv is not None and str(qv).strip():
                parts.append(str(qv).strip())
        combined = " ".join(parts).lower()
        if not combined:
            return []

        negations = []
        for rule_key, rule_vals in quantity_rules.items():
            if rule_key in combined:
                negations.extend(rule_vals)
        return negations

    def prune_negations(self, negations, max_tokens=75):
        """
        按优先级剪枝否定词列表，控制总 token 数。

        优先级：attribute > spatial > style > context > quantity > general。

        Args:
            negations (list[str]): 原始否定词列表。
            max_tokens (int): 最大 token 数，默认 75。

        Returns:
            list[str]: 剪枝后的否定词列表（保持输入顺序的同时优先保留高优先级词）。

        Example:
            >>> npg = NegativePromptGenerator()
            >>> npg.prune_negations(["a", "b"], max_tokens=10)
            ['a', 'b']
        """
        if not negations:
            return []
        if _estimate_tokens(negations) <= max_tokens:
            return list(negations)

        # 为每个词计算优先级标签
        attr_rules = self.rules.get("attribute_negations", {})
        spatial_rules = self.rules.get("spatial_negations", {})
        style_rules = self.rules.get("style_negations", {})
        context_rules = self.rules.get("context_negations", {})
        quantity_rules = self.rules.get("quantity_negations", {})
        general_set = set(self.rules.get("general_negations", []))

        def _flat_values(d):
            s = set()
            for v in d.values():
                if isinstance(v, list):
                    s.update([x.lower() for x in v])
            return s

        attr_values = _flat_values(attr_rules)
        spatial_values = _flat_values(spatial_rules)
        style_values = _flat_values(style_rules)
        context_values = _flat_values(context_rules)
        quantity_values = _flat_values(quantity_rules)

        def _priority(word):
            w = word.lower()
            if w in attr_values:
                return PRIORITY_MAP["attribute"]
            if w in spatial_values:
                return PRIORITY_MAP["spatial"]
            if w in style_values:
                return PRIORITY_MAP["style"]
            if w in context_values:
                return PRIORITY_MAP["context"]
            if w in quantity_values:
                return PRIORITY_MAP["quantity"]
            return PRIORITY_MAP["general"]

        # 按 (priority, 原始索引) 排序以保留同优先级原始顺序
        ranked = sorted(
            enumerate(negations),
            key=lambda pair: (_priority(pair[1]), pair[0]),
        )

        # 逐步添加直到 token 数达到上限
        selected = []
        for _, word in ranked:
            trial = selected + [word]
            if _estimate_tokens(trial) > max_tokens:
                break
            selected.append(word)
        return selected

    @staticmethod
    def _match_into_negations(description, rules, out_list):
        """在 rules dict 中按 description 查找，命中则加入 out_list。"""
        desc_lower = description.lower()
        if desc_lower in rules and isinstance(rules[desc_lower], list):
            out_list.extend(rules[desc_lower])
            return
        for rule_key, rule_vals in rules.items():
            if isinstance(rule_vals, list) and (
                rule_key in desc_lower or desc_lower in rule_key
            ):
                out_list.extend(rule_vals)
                return
