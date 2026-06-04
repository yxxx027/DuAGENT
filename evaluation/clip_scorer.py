import torch
from PIL import Image


class CLIPScorer:
    """
    使用 CLIP 模型计算文本-图像对齐度。
    本地运行，无需 API。
    """

    def __init__(self, device: str = "cuda:0", model_path: str = None):
        self.device = device
        self.model_path = model_path or "openai/clip-vit-large-patch14"
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        from transformers import CLIPModel, CLIPProcessor
        self.model = CLIPModel.from_pretrained(self.model_path)
        self.processor = CLIPProcessor.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()

    def score(self, prompt: str, image_path: str) -> float:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(
            text=[prompt], images=[image],
            return_tensors="pt", padding=True
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            similarity = outputs.logits_per_image[0][0].item()
        return float(similarity)

    def score_batch(self, prompts: list[str], image_paths: list[str]) -> list[float]:
        results = []
        for prompt, img_path in zip(prompts, image_paths):
            results.append(self.score(prompt, img_path))
        return results

    def score_per_attribute(self, structured: dict, image_path: str) -> dict:
        image = Image.open(image_path).convert("RGB")
        results = {}
        for obj in structured.get("objects", []):
            obj_name = obj.get("name", "unknown")
            attrs = obj.get("attributes", {})
            attr_texts = []
            for attr_key, attr_val in attrs.items():
                if attr_val:
                    attr_texts.append(f"{attr_val} {obj_name}")
            if not attr_texts:
                attr_texts.append(obj_name)
            inputs = self.processor(
                text=attr_texts, images=[image] * len(attr_texts),
                return_tensors="pt", padding=True
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = outputs.logits_per_image.diagonal().tolist()
            results[obj_name] = {
                "attributes": {t: s for t, s in zip(attr_texts, scores)},
                "average": sum(scores) / len(scores) if scores else 0.0
            }
        return results
