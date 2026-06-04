import argparse
import importlib
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import torch
from PIL import Image
from tqdm import tqdm

from utils_all.api import APIClient


DEFAULT_CONFIG = {
    "input_file": "evaluation/test_prompts.txt",
    "cuda": "cuda:0",
    "model_name": "sd1",
    "model_path": "",
    "api_key": "",
    "url": "https://api.deepseek.com",
    "api_model": "deepseek-chat",
    "output_dir": "output",
    "template_path": "prompts/structured_parse.txt",
    "use_structured": False,
    "use_negative": False,
    "use_clip": False,
    "clip_device": "cuda:0",
    "clip_model_path": None,
    "cache": True,
    "max_retries": 3,
    "retry_delay": 2.0,
    "retry_backoff": 2.0,
    "seed": 42,
    "width": 512,
    "height": 512,
    "num_inference_steps": 50,
}

LOGGER = logging.getLogger("genpilot")


def atomic_write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    os.replace(temp_path, path)


def load_json(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def retry_call(
    operation: Callable[[], Any],
    description: str,
    max_retries: int,
    retry_delay: float,
    retry_backoff: float,
) -> Any:
    attempts = max(1, max_retries + 1)
    delay = max(0.0, retry_delay)
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception:
            if attempt >= attempts:
                LOGGER.exception("%s failed after %d attempt(s)", description, attempt)
                raise
            LOGGER.exception(
                "%s failed on attempt %d/%d; retrying in %.1fs",
                description,
                attempt,
                attempts,
                delay,
            )
            time.sleep(delay)
            delay *= max(1.0, retry_backoff)
    raise RuntimeError(f"Unreachable retry state for {description}")


def setup_logging(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    LOGGER.addHandler(console)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
    return log_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Structured Prompt + Negative Prompt Agent")
    parser.add_argument("--config", help="JSON configuration file")
    parser.add_argument("--resume_run", help="Existing run directory to resume")
    parser.add_argument("--run_name", help="Optional fixed run directory name")
    parser.add_argument("--input_file")
    parser.add_argument("--cuda")
    parser.add_argument("--model_name", choices=["sd1", "sd2", "sd3", "flux"])
    parser.add_argument("--model_path")
    parser.add_argument("--api_key")
    parser.add_argument("--url")
    parser.add_argument("--api_model")
    parser.add_argument("--output_dir")
    parser.add_argument("--template_path")
    parser.add_argument("--clip_device")
    parser.add_argument("--clip_model_path")
    parser.add_argument("--max_retries", type=int)
    parser.add_argument("--retry_delay", type=float)
    parser.add_argument("--retry_backoff", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--num_inference_steps", type=int)
    parser.add_argument("--use_structured", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use_negative", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use_clip", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=None)
    return parser


def resolve_config(argv: list[str] | None = None) -> tuple[dict, argparse.Namespace]:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = dict(DEFAULT_CONFIG)

    if args.resume_run:
        saved_config_path = Path(args.resume_run) / "run_config.json"
        saved_config = load_json(saved_config_path)
        if not isinstance(saved_config, dict):
            parser.error(f"Resume config not found or invalid: {saved_config_path}")
        config.update(saved_config)

    if args.config:
        loaded = load_json(args.config)
        if not isinstance(loaded, dict):
            parser.error("--config must contain a JSON object")
        unknown = sorted(set(loaded) - set(DEFAULT_CONFIG))
        if unknown:
            parser.error(f"Unknown config key(s): {', '.join(unknown)}")
        config.update(loaded)

    for key in DEFAULT_CONFIG:
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value

    config["api_key"] = config["api_key"] or os.getenv("DEEPSEEK_API_KEY", "")
    if not config["model_path"]:
        parser.error("--model_path is required, either on the command line or in --config")
    if config["use_structured"] and not config["api_key"]:
        parser.error("--use_structured requires --api_key or DEEPSEEK_API_KEY")
    if config["max_retries"] < 0:
        parser.error("--max_retries must be >= 0")
    return config, args


def public_config(config: dict) -> dict:
    return {key: value for key, value in config.items() if key != "api_key"}


def validate_resume_config(run_dir: str | Path, config: dict) -> None:
    saved = load_json(Path(run_dir) / "run_config.json")
    if not isinstance(saved, dict) or not config["cache"]:
        return
    cache_sensitive = {
        "input_file",
        "model_name",
        "model_path",
        "template_path",
        "use_structured",
        "use_negative",
        "clip_model_path",
        "seed",
        "width",
        "height",
        "num_inference_steps",
    }
    changed = sorted(
        key for key in cache_sensitive if saved.get(key) != config.get(key)
    )
    if changed:
        raise ValueError(
            "Cannot resume with cache after changing: "
            f"{', '.join(changed)}. Use --no-cache to recompute the run."
        )


def create_run_dir(config: dict, args: argparse.Namespace) -> Path:
    if args.resume_run:
        run_dir = Path(args.resume_run)
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Resume directory does not exist: {run_dir}")
        return run_dir
    name = args.run_name or f"run_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    run_dir = Path(config["output_dir"]) / name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def load_model_module(name: str):
    if name == "flux":
        module_name = "utils_all.image_flux"
    elif name in {"sd1", "sd2", "sd3"}:
        module_name = f"utils_all.image_{name}"
    else:
        raise ValueError(f"Unknown model name: {name}")
    return importlib.import_module(module_name)


def load_prompts(prompt_file: str) -> list[str]:
    with open(prompt_file, "r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def load_template(template_path: str) -> list[str]:
    with open(template_path, "r", encoding="utf-8") as file:
        return [line.rstrip("\n") for line in file]


def truncate_prompt_for_pipeline(pipeline, prompt: str, max_tokens: int = 75) -> str:
    tokenizer = pipeline.tokenizer
    token_ids = tokenizer.encode(prompt, truncation=True, max_length=max_tokens + 2)
    if len(token_ids) <= max_tokens + 2:
        return prompt
    return tokenizer.decode(token_ids[1:max_tokens + 1], skip_special_tokens=True).strip()


def is_valid_image(path: str | Path) -> bool:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def generate_image(
    pipeline,
    prompt: str,
    output_path: str | Path,
    config: dict,
    seed: int,
    negative_prompt: str = "",
) -> str:
    output_path = Path(output_path)
    if config["cache"] and is_valid_image(output_path):
        LOGGER.info("Cache hit: %s", output_path)
        return str(output_path)

    kwargs = {
        "width": config["width"],
        "height": config["height"],
        "num_inference_steps": 4 if config["model_name"] == "flux" else config["num_inference_steps"],
        "generator": torch.Generator("cpu").manual_seed(seed),
    }
    if config["model_name"] == "flux":
        kwargs["guidance_scale"] = 0.0
    elif negative_prompt:
        kwargs["negative_prompt"] = negative_prompt

    def operation():
        image = pipeline(prompt, **kwargs).images[0]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = output_path.with_suffix(".tmp.png")
        image.save(temp_path)
        os.replace(temp_path, output_path)
        return str(output_path)

    return retry_call(
        operation,
        f"Generate image {output_path.name}",
        config["max_retries"],
        config["retry_delay"],
        config["retry_backoff"],
    )


def empty_checkpoint(prompts: list[str]) -> dict:
    size = len(prompts)
    return {
        "status": "running",
        "prompts": prompts,
        "structured_results": [None] * size,
        "enhanced_prompts": [None] * size,
        "negative_prompts": [None] * size,
        "baseline_paths": [None] * size,
        "improved_paths": [None] * size,
        "baseline_scores": [None] * size,
        "improved_scores": [None] * size,
    }


def load_checkpoint(path: str | Path, prompts: list[str]) -> dict:
    checkpoint = load_json(path, empty_checkpoint(prompts))
    if checkpoint.get("prompts") != prompts:
        raise ValueError("Prompt file changed; refusing to resume an incompatible run")
    expected = len(prompts)
    for key, default in empty_checkpoint(prompts).items():
        checkpoint.setdefault(key, default)
        if isinstance(checkpoint[key], list) and len(checkpoint[key]) != expected:
            raise ValueError(f"Checkpoint field {key!r} has an incompatible length")
    return checkpoint


def save_checkpoint(path: str | Path, checkpoint: dict) -> None:
    atomic_write_json(path, checkpoint)


def cached_value(values: list, index: int, cache: bool) -> Any:
    return values[index] if cache and values[index] is not None else None


def score_images(clip, prompts: list[str], paths: list[str], scores: list, config: dict, checkpoint_path, checkpoint):
    for index, (prompt, image_path) in enumerate(
        tqdm(list(zip(prompts, paths)), desc="CLIP scoring")
    ):
        if cached_value(scores, index, config["cache"]) is not None:
            LOGGER.info("CLIP cache hit: item %d", index)
            continue
        scores[index] = retry_call(
            lambda p=prompt, image=image_path: clip.score(p, image),
            f"CLIP score item {index}",
            config["max_retries"],
            config["retry_delay"],
            config["retry_backoff"],
        )
        save_checkpoint(checkpoint_path, checkpoint)


def main(argv: list[str] | None = None) -> None:
    config, args = resolve_config(argv)
    run_dir = create_run_dir(config, args)
    log_path = setup_logging(run_dir)
    checkpoint_path = run_dir / "checkpoint.json"
    if args.resume_run:
        validate_resume_config(run_dir, config)
    atomic_write_json(run_dir / "run_config.json", public_config(config))

    prompts = load_prompts(config["input_file"])
    checkpoint = load_checkpoint(checkpoint_path, prompts)
    if not config["cache"]:
        checkpoint = empty_checkpoint(prompts)
    checkpoint["status"] = "running"
    save_checkpoint(checkpoint_path, checkpoint)
    LOGGER.info("Loaded %d prompts from %s", len(prompts), config["input_file"])
    LOGGER.info("Run directory: %s", run_dir)

    structured_results = checkpoint["structured_results"]
    enhanced_prompts = checkpoint["enhanced_prompts"]
    negative_prompts = checkpoint["negative_prompts"]
    for index, prompt in enumerate(prompts):
        enhanced_prompts[index] = enhanced_prompts[index] or prompt
        negative_prompts[index] = negative_prompts[index] or ""

    if config["use_structured"]:
        from agents.structured_parser import StructuredParser

        parser = StructuredParser(
            APIClient(config["api_key"], config["url"], config["api_model"]),
            load_template(config["template_path"]),
        )
        for index, prompt in enumerate(tqdm(prompts, desc="Structured parsing")):
            if cached_value(structured_results, index, config["cache"]) is not None:
                LOGGER.info("Structured cache hit: item %d", index)
                enhanced_prompts[index] = parser.reassemble(structured_results[index]) or prompt
                continue
            structured_results[index] = retry_call(
                lambda p=prompt: parser.parse(p),
                f"Structured parse item {index}",
                config["max_retries"],
                config["retry_delay"],
                config["retry_backoff"],
            )
            enhanced_prompts[index] = parser.reassemble(structured_results[index]) or prompt
            save_checkpoint(checkpoint_path, checkpoint)
        atomic_write_json(run_dir / "structured_results.json", structured_results)

    if config["use_negative"]:
        from agents.negative_prompt_generator import NegativePromptGenerator

        generator = NegativePromptGenerator()
        for index, prompt in enumerate(prompts):
            if cached_value(negative_prompts, index, config["cache"]):
                LOGGER.info("Negative prompt cache hit: item %d", index)
                continue
            structured = structured_results[index] or {
                "objects": [{"id": 1, "name": prompt, "attributes": {}}],
                "spatial_relations": [],
            }
            negative_prompts[index] = generator.generate(structured)
            save_checkpoint(checkpoint_path, checkpoint)
        atomic_write_json(
            run_dir / "negative_prompts.json",
            {"prompts": enhanced_prompts, "negative_prompts": negative_prompts},
        )

    baseline_dir = run_dir / "baseline_images"
    improved_dir = run_dir / "improved_images"
    expected_images = [
        *(baseline_dir / f"{index}.png" for index in range(len(prompts))),
        *(improved_dir / f"{index}.png" for index in range(len(prompts))),
    ]
    needs_pipeline = not config["cache"] or any(
        not is_valid_image(path) for path in expected_images
    )
    pipeline = None
    if needs_pipeline:
        model_module = load_model_module(config["model_name"])
        pipeline = retry_call(
            lambda: model_module.get_pipe_slow(config["model_path"], config["cuda"]),
            f"Load {config['model_name']} model",
            config["max_retries"],
            config["retry_delay"],
            config["retry_backoff"],
        )
    else:
        LOGGER.info("All images are cached; skipping image model loading")

    for index, prompt in enumerate(tqdm(prompts, desc="Baseline images")):
        path = baseline_dir / f"{index}.png"
        checkpoint["baseline_paths"][index] = generate_image(
            pipeline, prompt, path, config, config["seed"] + index
        )
        save_checkpoint(checkpoint_path, checkpoint)

    for index, prompt in enumerate(tqdm(enhanced_prompts, desc="Improved images")):
        path = improved_dir / f"{index}.png"
        if config["cache"] and is_valid_image(path):
            LOGGER.info("Cache hit: %s", path)
            checkpoint["improved_paths"][index] = str(path)
            save_checkpoint(checkpoint_path, checkpoint)
            continue
        truncated = truncate_prompt_for_pipeline(pipeline, prompt)
        if truncated != prompt:
            LOGGER.info("Prompt %d truncated from %d to %d characters", index, len(prompt), len(truncated))
        checkpoint["improved_paths"][index] = generate_image(
            pipeline,
            truncated,
            path,
            config,
            config["seed"] + index,
            negative_prompts[index],
        )
        save_checkpoint(checkpoint_path, checkpoint)

    if config["use_clip"]:
        from evaluation.clip_scorer import CLIPScorer
        from evaluation.visualize_results import generate_report_table, plot_score_comparison_bar

        clip = retry_call(
            lambda: CLIPScorer(device=config["clip_device"], model_path=config["clip_model_path"]),
            "Load CLIP model",
            config["max_retries"],
            config["retry_delay"],
            config["retry_backoff"],
        )
        score_images(
            clip, prompts, checkpoint["baseline_paths"], checkpoint["baseline_scores"],
            config, checkpoint_path, checkpoint,
        )
        score_images(
            clip, enhanced_prompts, checkpoint["improved_paths"], checkpoint["improved_scores"],
            config, checkpoint_path, checkpoint,
        )
        plot_score_comparison_bar(
            checkpoint["baseline_scores"],
            checkpoint["improved_scores"],
            str(run_dir / "score_comparison.png"),
        )
        generate_report_table(
            checkpoint["baseline_scores"],
            checkpoint["improved_scores"],
            str(run_dir / "report_table.md"),
            prompts=prompts,
        )

    results = {
        "config": public_config(config),
        "prompts": prompts,
        "enhanced_prompts": enhanced_prompts,
        "negative_prompts": negative_prompts,
        "baseline_scores": checkpoint["baseline_scores"] if config["use_clip"] else [],
        "improved_scores": checkpoint["improved_scores"] if config["use_clip"] else [],
    }
    atomic_write_json(run_dir / "results.json", results)
    checkpoint["status"] = "completed"
    save_checkpoint(checkpoint_path, checkpoint)
    LOGGER.info("Run complete. Results: %s", run_dir)
    LOGGER.info("Log file: %s", log_path)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        LOGGER.exception("Run failed")
        raise
