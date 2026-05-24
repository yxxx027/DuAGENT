import argparse
import json
import os
import random
import importlib
from datetime import datetime
from tqdm import tqdm

from utils_all.api import APIClient


def load_model_module(name):
    if name == "flux":
        module_name = "utils_all.image_flux"
    elif name in ["sd1", "sd2", "sd3"]:
        module_name = f"utils_all.image_{name}"
    else:
        raise ValueError(f"Unknown model name: {name}")
    return importlib.import_module(module_name)


def load_prompts(prompt_file: str) -> list[str]:
    with open(prompt_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_template(template_path: str) -> list[str]:
    with open(template_path, "r", encoding="utf-8") as f:
        return f.readlines()


def generate_images_for_prompts(model_module, pipeline, prompts: list[str],
                                 output_dir: str, negative_prompt: str = "") -> list[str]:
    os.makedirs(output_dir, exist_ok=True)
    image_paths = []
    for idx, prompt in enumerate(tqdm(prompts, desc="Generating images")):
        if negative_prompt:
            image = pipeline(
                prompt, negative_prompt=negative_prompt,
                width=512, height=512, num_inference_steps=50,
                generator=torch.Generator("cpu").manual_seed(random.randint(1, 1000))
            ).images[0]
        else:
            image = model_module.t2i_slow(
                pipeline, prompt, idx,
                width=512, height=512, output_dir=output_dir
            )
            image_paths.append(os.path.join(output_dir, f"{idx}.png"))
            continue
        save_path = os.path.join(output_dir, f"{idx}.png")
        image.save(save_path)
        image_paths.append(save_path)
    return image_paths


def run_clip_scoring(clip_scorer, prompts, image_paths) -> list[float]:
    scores = []
    for prompt, img_path in tqdm(zip(prompts, image_paths), desc="CLIP scoring", total=len(prompts)):
        score = clip_scorer.score(prompt, img_path)
        scores.append(score)
    return scores


def main():
    parser = argparse.ArgumentParser(description="Structured Prompt + Negative Prompt Agent")
    parser.add_argument("--input_file", type=str, default="evaluation/test_prompts.txt",
                        help="Path to prompt file (one prompt per line)")
    parser.add_argument("--cuda", type=str, default="cuda:0", help="CUDA device")
    parser.add_argument("--model_name", type=str, default="sd1",
                        help="Image model name: sd1, sd2, sd3, flux")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to image model weights")
    parser.add_argument("--api_key", type=str, default="", help="LLM API key")
    parser.add_argument("--url", type=str, default="", help="LLM API base URL")
    parser.add_argument("--api_model", type=str, default="	deepseek-v4-flash", help="LLM model name")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Output directory")
    parser.add_argument("--use_structured", action="store_true",
                        help="Enable structured prompt parsing")
    parser.add_argument("--use_negative", action="store_true",
                        help="Enable negative prompt generation")
    parser.add_argument("--use_clip", action="store_true",
                        help="Enable CLIP scoring")
    parser.add_argument("--clip_device", type=str, default="cuda:0",
                        help="Device for CLIP model")
    parser.add_argument("--clip_model_path", type=str, default=None,
                        help="Local path to CLIP model (skip HuggingFace download)")
    args = parser.parse_args()

    import torch
    global torch

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    run_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    prompts = load_prompts(args.input_file)
    print(f"Loaded {len(prompts)} prompts from {args.input_file}")

    config = {
        "timestamp": timestamp,
        "num_prompts": len(prompts),
        "use_structured": args.use_structured,
        "use_negative": args.use_negative,
        "use_clip": args.use_clip,
        "model_name": args.model_name,
        "input_file": args.input_file,
    }

    model_module = load_model_module(args.model_name)
    print(f"Loading {args.model_name} model from {args.model_path}...")
    pipeline = model_module.get_pipe_slow(args.model_path, args.cuda)
    print("Model loaded.")

    structured_results = []
    enhanced_prompts = prompts[:]
    negative_prompts = [""] * len(prompts)

    if args.use_structured:
        if not args.api_key:
            print("WARNING: --use_structured requires --api_key. Skipping structured parsing.")
        else:
            print("Initializing StructuredParser...")
            client = APIClient(args.api_key, args.url, args.api_model)
            template = load_template("prompts/structured_parse.txt")
            from agents.structured_parser import StructuredParser
            sp = StructuredParser(client, template)

            print("Parsing prompts into structured format...")
            for i, prompt in enumerate(tqdm(prompts, desc="Structured parsing")):
                structured = sp.parse(prompt)
                structured_results.append(structured)
                reassembled = sp.reassemble(structured)
                if reassembled:
                    enhanced_prompts[i] = reassembled

            structured_file = os.path.join(run_dir, "structured_results.json")
            with open(structured_file, "w", encoding="utf-8") as f:
                json.dump(structured_results, f, ensure_ascii=False, indent=2)
            print(f"Structured results saved to {structured_file}")

    if args.use_negative:
        print("Initializing NegativePromptGenerator...")
        from agents.negative_prompt_generator import NegativePromptGenerator
        npg = NegativePromptGenerator()

        if structured_results:
            for i, structured in enumerate(structured_results):
                negative_prompts[i] = npg.generate(structured)
        else:
            for i, prompt in enumerate(prompts):
                fallback = {"objects": [{"id": 1, "name": prompt, "attributes": {}}], "spatial_relations": []}
                negative_prompts[i] = npg.generate(fallback)

        neg_file = os.path.join(run_dir, "negative_prompts.json")
        with open(neg_file, "w", encoding="utf-8") as f:
            json.dump({"prompts": enhanced_prompts, "negative_prompts": negative_prompts},
                      f, ensure_ascii=False, indent=2)
        print(f"Negative prompts saved to {neg_file}")

    print("Generating BASELINE images...")
    baseline_dir = os.path.join(run_dir, "baseline_images")
    baseline_paths = generate_images_for_prompts(model_module, pipeline, prompts, baseline_dir)
    print(f"Baseline images saved to {baseline_dir}")

    print("Generating IMPROVED images...")
    improved_dir = os.path.join(run_dir, "improved_images")
    improved_paths = []
    os.makedirs(improved_dir, exist_ok=True)
    for idx, (enh_prompt, neg_prompt) in enumerate(
        tqdm(zip(enhanced_prompts, negative_prompts), desc="Generating improved images", total=len(prompts))
    ):
        image = pipeline(
            enh_prompt,
            negative_prompt=neg_prompt if neg_prompt else None,
            width=512, height=512, num_inference_steps=50,
            generator=torch.Generator("cpu").manual_seed(random.randint(1, 1000))
        ).images[0]
        save_path = os.path.join(improved_dir, f"{idx}.png")
        image.save(save_path)
        improved_paths.append(save_path)
    print(f"Improved images saved to {improved_dir}")

    baseline_scores = []
    improved_scores = []

    if args.use_clip:
        print("Initializing CLIP scorer...")
        from evaluation.clip_scorer import CLIPScorer
        clip = CLIPScorer(device=args.clip_device, model_path=args.clip_model_path)

        print("Scoring baseline images...")
        baseline_scores = run_clip_scoring(clip, prompts, baseline_paths)

        print("Scoring improved images...")
        improved_scores = run_clip_scoring(clip, enhanced_prompts, improved_paths)

        from evaluation.visualize_results import plot_score_comparison_bar, generate_report_table
        plot_score_comparison_bar(
            baseline_scores, improved_scores,
            os.path.join(run_dir, "score_comparison.png")
        )
        generate_report_table(
            baseline_scores, improved_scores,
            os.path.join(run_dir, "report_table.md")
        )
        print(f"Visualizations saved to {run_dir}/")

    results = {
        "config": config,
        "prompts": prompts,
        "enhanced_prompts": enhanced_prompts,
        "negative_prompts": negative_prompts,
        "baseline_scores": baseline_scores,
        "improved_scores": improved_scores,
    }
    results_file = os.path.join(run_dir, "results.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"Run complete! Results saved to: {run_dir}/")
    print(f"  Baseline images:  {baseline_dir}/")
    print(f"  Improved images:  {improved_dir}/")
    print(f"  Results JSON:     {results_file}")
    if baseline_scores and improved_scores:
        avg_b = sum(baseline_scores) / len(baseline_scores)
        avg_i = sum(improved_scores) / len(improved_scores)
        print(f"  Avg Baseline CLIP: {avg_b:.4f}")
        print(f"  Avg Improved CLIP: {avg_i:.4f}")
        print(f"  Delta: {avg_i - avg_b:+.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
