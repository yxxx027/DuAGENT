from typing import Any

import torch
from PIL import Image
from tqdm import tqdm


class CLIPScorer:
    """
    CLIP-based evaluator for text-image alignment.

    Features:
    1. Return normalized 0-10 CLIP scores.
    2. Support batch scoring.
    3. Compare baseline and improved images.
    4. Score object attributes and spatial relations from structured JSON.
    5. Support local model_path to skip HuggingFace download.
    """

    def __init__(
        self,
        device: str = "cuda:0",
        model_name: str = "openai/clip-vit-large-patch14",
        model_path: str = None,
        normalize: bool = True,
    ):
        self.device = device
        self.model_name = model_path or model_name
        self.normalize = normalize
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        from transformers import CLIPModel, CLIPProcessor

        self.model = CLIPModel.from_pretrained(self.model_name)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def normalize_score(raw_score: float) -> float:
        normalized = (raw_score - 10.0) / 2.5
        normalized = max(0.0, min(10.0, normalized))
        return round(float(normalized), 4)

    def _raw_score(self, prompt: str, image_path: str) -> float:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(
            text=[prompt],
            images=[image],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            similarity = outputs.logits_per_image[0][0].item()

        return float(similarity)

    def score(self, prompt: str, image_path: str) -> float:
        raw = self._raw_score(prompt, image_path)
        if self.normalize:
            return self.normalize_score(raw)
        return float(raw)

    def score_batch(self, prompts: list[str], image_paths: list[str]) -> list[float]:
        if len(prompts) != len(image_paths):
            raise ValueError(
                f"prompts and image_paths must have the same length, "
                f"got {len(prompts)} and {len(image_paths)}"
            )

        results = []
        for prompt, img_path in tqdm(
            list(zip(prompts, image_paths)),
            desc="CLIP batch scoring",
            total=len(prompts),
        ):
            results.append(self.score(prompt, img_path))
        return results

    def compare_baseline_improved(
        self,
        baseline_prompt: str,
        improved_prompt: str,
        baseline_img: str,
        improved_img: str,
        tie_threshold: float = 0.15,
    ) -> dict[str, Any]:
        baseline_score = self.score(baseline_prompt, baseline_img)
        improved_score = self.score(improved_prompt, improved_img)
        delta = improved_score - baseline_score

        if abs(delta) <= tie_threshold:
            winner = "Tie"
        elif delta > 0:
            winner = "Improved"
        else:
            winner = "Baseline"

        return {
            "baseline_score": round(float(baseline_score), 4),
            "improved_score": round(float(improved_score), 4),
            "delta": round(float(delta), 4),
            "winner": winner,
        }

    def compare_batch(
        self,
        baseline_prompts: list[str],
        improved_prompts: list[str],
        baseline_imgs: list[str],
        improved_imgs: list[str],
        tie_threshold: float = 0.15,
    ) -> list[dict[str, Any]]:
        n = len(baseline_prompts)
        if not (
            len(improved_prompts) == n
            and len(baseline_imgs) == n
            and len(improved_imgs) == n
        ):
            raise ValueError("All input lists must have the same length.")

        results = []
        for i in tqdm(range(n), desc="Comparing baseline vs improved"):
            item = self.compare_baseline_improved(
                baseline_prompts[i],
                improved_prompts[i],
                baseline_imgs[i],
                improved_imgs[i],
                tie_threshold=tie_threshold,
            )
            item["index"] = i
            results.append(item)
        return results

    def score_per_attribute(self, structured: dict, image_path: str) -> dict:
        results = {}

        for obj in structured.get("objects", []):
            obj_name = obj.get("name", "unknown")
            attrs = obj.get("attributes", {}) or {}

            attr_texts = []

            for attr_key, attr_val in attrs.items():
                if attr_val is None or attr_val == "":
                    continue

                if isinstance(attr_val, list):
                    values = [str(v) for v in attr_val if v]
                else:
                    values = [str(attr_val)]

                for value in values:
                    attr_texts.append(f"{value} {obj_name}")

            if not attr_texts:
                attr_texts.append(obj_name)

            scores = []
            for text in attr_texts:
                scores.append(self.score(text, image_path))

            results[obj_name] = {
                "attributes": {t: s for t, s in zip(attr_texts, scores)},
                "average": round(sum(scores) / len(scores), 4) if scores else 0.0,
            }

        return results

    def score_spatial_relations(self, structured: dict, image_path: str) -> dict:
        objects = structured.get("objects", []) or []
        relations = structured.get("spatial_relations", []) or []

        id_to_name = {}
        for obj in objects:
            obj_id = obj.get("id")
            obj_name = obj.get("name", "object")
            if obj_id is not None:
                id_to_name[str(obj_id)] = obj_name

        relation_scores = {}

        for rel in relations:
            subject = (
                rel.get("subject")
                or rel.get("subject_name")
                or id_to_name.get(str(rel.get("subject_id")), "object")
            )
            obj = (
                rel.get("object")
                or rel.get("object_name")
                or id_to_name.get(str(rel.get("object_id")), "object")
            )
            relation = rel.get("relation", "")

            if not relation:
                continue

            query = f"{subject} {relation} {obj}"
            relation_scores[query] = self.score(query, image_path)

        average = (
            round(sum(relation_scores.values()) / len(relation_scores), 4)
            if relation_scores
            else 0.0
        )

        return {
            "relations": relation_scores,
            "average": average,
        }

    @staticmethod
    def compute_statistics(comparison_results: list[dict]) -> dict:
        if not comparison_results:
            return {
                "num_samples": 0,
                "baseline_wins": 0,
                "improved_wins": 0,
                "ties": 0,
                "average_baseline": 0.0,
                "average_improved": 0.0,
                "average_delta": 0.0,
            }

        baseline_wins = sum(1 for r in comparison_results if r["winner"] == "Baseline")
        improved_wins = sum(1 for r in comparison_results if r["winner"] == "Improved")
        ties = sum(1 for r in comparison_results if r["winner"] == "Tie")

        baseline_scores = [r["baseline_score"] for r in comparison_results]
        improved_scores = [r["improved_score"] for r in comparison_results]
        deltas = [r["delta"] for r in comparison_results]

        return {
            "num_samples": len(comparison_results),
            "baseline_wins": baseline_wins,
            "improved_wins": improved_wins,
            "ties": ties,
            "average_baseline": round(sum(baseline_scores) / len(baseline_scores), 4),
            "average_improved": round(sum(improved_scores) / len(improved_scores), 4),
            "average_delta": round(sum(deltas) / len(deltas), 4),
        }
