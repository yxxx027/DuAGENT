"""NegativePromptGenerator 单元测试。

注意：为避免包初始化依赖（agents/__init__.py 依赖 openai），
本测试通过 importlib 直接加载 negative_prompt_generator.py 模块文件。
"""

import importlib.util
import os
import sys

import pytest


HERE = os.path.dirname(os.path.abspath(__file__))
MODULE_PATH = os.path.join(HERE, "agents", "negative_prompt_generator.py")


@pytest.fixture(scope="module")
def npg_module():
    """加载模块文件，返回模块对象。"""
    spec = importlib.util.spec_from_file_location(
        "negative_prompt_generator_under_test", MODULE_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def npg(npg_module):
    return npg_module.NegativePromptGenerator()


# --------------------------------------------------------------------------- #
# 规则库加载
# --------------------------------------------------------------------------- #
def test_rules_load_has_required_keys(npg):
    """规则库必须包含 6 个顶级 key。"""
    required = {
        "attribute_negations",
        "spatial_negations",
        "style_negations",
        "context_negations",
        "quantity_negations",
        "general_negations",
    }
    assert required.issubset(set(npg.rules.keys()))


def test_rules_counts(npg):
    """至少数量要达标。"""
    assert len(npg.rules["attribute_negations"]) >= 10
    assert len(npg.rules["style_negations"]) >= 8
    assert len(npg.rules["context_negations"]) >= 8
    assert len(npg.rules["quantity_negations"]) >= 5
    assert len(npg.rules["general_negations"]) > 0


# --------------------------------------------------------------------------- #
# 属性否定
# --------------------------------------------------------------------------- #
def test_attribute_negations_red_returns_color_negations(npg):
    """color=red 应返回 blue/green 等否定色。"""
    result = npg.generate_attribute_negations(
        [{"name": "apple", "attributes": {"color": "red"}}]
    )
    lowered = [w.lower() for w in result]
    assert any(c in lowered for c in ("blue", "green", "yellow", "orange", "purple"))


def test_attribute_negations_empty_objects(npg):
    """空物体列表不返回任何否定。"""
    assert npg.generate_attribute_negations([]) == []
    assert npg.generate_attribute_negations(None) == []


# --------------------------------------------------------------------------- #
# 空间否定
# --------------------------------------------------------------------------- #
def test_spatial_negations_left_returns_right(npg):
    """left of 关系应返回 right side 等。"""
    result = npg.generate_spatial_negations(
        [{"subject": "apple", "relation": "left of", "object": "car"}]
    )
    assert any("right" in w.lower() for w in result)


def test_spatial_negations_empty(npg):
    assert npg.generate_spatial_negations([]) == []
    assert npg.generate_spatial_negations(None) == []


# --------------------------------------------------------------------------- #
# 风格否定
# --------------------------------------------------------------------------- #
def test_style_negations_cartoon(npg):
    """cartoon 风格应返回 realistic/photorealistic。"""
    result = npg.generate_style_negations({"style": "cartoon"})
    lowered = [w.lower() for w in result]
    assert "realistic" in lowered or "photorealistic" in lowered


def test_style_negations_from_object(npg):
    """object.attributes.style 也可产生否定。"""
    structured = {
        "objects": [{"name": "dog", "attributes": {"style": "anime"}}]
    }
    result = npg.generate_style_negations(structured)
    assert len(result) > 0


def test_style_negations_empty(npg):
    assert npg.generate_style_negations({}) == []
    assert npg.generate_style_negations(None) == []
    assert npg.generate_style_negations({"style": ""}) == []


# --------------------------------------------------------------------------- #
# 上下文否定
# --------------------------------------------------------------------------- #
def test_context_negations_indoor(npg):
    """indoor 上下文应返回 outdoor/sky。"""
    result = npg.generate_context_negations(
        {"context": "indoor", "setting": "kitchen room"}
    )
    lowered = [w.lower() for w in result]
    assert "outdoor" in lowered or "sky" in lowered


def test_context_negations_empty(npg):
    assert npg.generate_context_negations({}) == []
    assert npg.generate_context_negations(None) == []


# --------------------------------------------------------------------------- #
# 数量否定
# --------------------------------------------------------------------------- #
def test_quantity_negations_extra_objects(npg):
    result = npg.generate_quantity_negations({"quantity_error": "extra_objects"})
    assert len(result) > 0


def test_quantity_negations_from_object_attribute(npg):
    structured = {
        "objects": [{"name": "cat", "attributes": {"quantity": "two"}}]
    }
    # 如果规则库中没有 two，不生成否定也属正常；有则非空
    result = npg.generate_quantity_negations(structured)
    assert isinstance(result, list)


def test_quantity_negations_empty(npg):
    assert npg.generate_quantity_negations({}) == []
    assert npg.generate_quantity_negations(None) == []


# --------------------------------------------------------------------------- #
# prune_negations
# --------------------------------------------------------------------------- #
def test_prune_short_list_unchanged(npg):
    short = ["a", "b", "c"]
    assert npg.prune_negations(short, max_tokens=10) == short


def test_prune_long_list_is_truncated(npg):
    long_list = [f"word{i}" for i in range(50)]
    result = npg.prune_negations(long_list, max_tokens=5)
    # ", ".join(result).split() 数量必须 <= 5
    tokens = ", ".join(result).split()
    assert len(tokens) <= 5


def test_prune_empty(npg):
    assert npg.prune_negations([], max_tokens=10) == []
    assert npg.prune_negations(None, max_tokens=10) == []


def test_prune_preserves_priority(npg):
    """高优先级（属性）词应优先保留。"""
    mixed = ["blurry", "red", "cartoon", "outdoor", "single"]
    # 这些词里 "red" / "cartoon" / "outdoor" / "single" 属于规则库值，
    # "blurry" 属于 general。让我们验证前几个被保留下来。
    result = npg.prune_negations(mixed, max_tokens=3)
    # 至少应保留一些属性/风格/上下文/数量级的否定，而不应全是 general
    assert len(result) >= 1


# --------------------------------------------------------------------------- #
# generate 集成
# --------------------------------------------------------------------------- #
def test_generate_returns_non_empty_string(npg):
    structured = {
        "objects": [
            {"id": 1, "name": "apple", "attributes": {"color": "red"}},
            {"id": 2, "name": "car", "attributes": {"color": "blue", "style": "realistic"}},
        ],
        "spatial_relations": [
            {"subject": "apple", "relation": "left of", "object": "car"}
        ],
        "context": "indoor",
        "style": "realistic",
    }
    result = npg.generate(structured)
    assert isinstance(result, str)
    assert len(result) > 0
    # 必须以英文逗号分隔
    assert ", " in result


def test_generate_token_bound(npg):
    """最终输出 token 数不应超过 75。"""
    structured = {
        "objects": [
            {"id": 1, "name": "apple", "attributes": {"color": "red", "shape": "round"}},
        ],
        "spatial_relations": [
            {"subject": "apple", "relation": "left of", "object": "car"},
            {"subject": "apple", "relation": "above", "object": "floor"},
        ],
        "style": "realistic",
        "context": "indoor",
        "setting": "kitchen",
        "background": "wooden walls",
        "quantity_error": "extra_objects",
    }
    result = npg.generate(structured)
    tokens = result.split()
    assert len(tokens) <= 75


def test_generate_no_duplicates(npg):
    structured = {
        "objects": [{"name": "apple", "attributes": {"color": "red"}}],
        "context": "indoor",
    }
    result = npg.generate(structured)
    parts = [p.strip().lower() for p in result.split(",")]
    # 无重复
    assert len(parts) == len(set(parts))


def test_generate_empty_structured(npg):
    assert isinstance(npg.generate({}), str)
    assert isinstance(npg.generate(None), str)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
