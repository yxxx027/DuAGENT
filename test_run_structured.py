import json
from pathlib import Path

import pytest
from PIL import Image

import run_structured


def minimal_config(**overrides):
    config = dict(run_structured.DEFAULT_CONFIG)
    config.update(
        {
            "model_name": "sd1",
            "cache": True,
            "max_retries": 0,
            "retry_delay": 0,
            "retry_backoff": 1,
            "width": 8,
            "height": 8,
            "num_inference_steps": 1,
        }
    )
    config.update(overrides)
    return config


def test_resolve_config_file_and_cli_override(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"model_path": "/model", "seed": 10, "cache": False}),
        encoding="utf-8",
    )

    config, _ = run_structured.resolve_config(
        ["--config", str(config_path), "--seed", "99", "--cache"]
    )

    assert config["model_path"] == "/model"
    assert config["seed"] == 99
    assert config["cache"] is True


def test_resume_loads_saved_config_without_model_path(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    run_structured.atomic_write_json(
        run_dir / "run_config.json",
        {"model_path": "/saved/model", "model_name": "sd2"},
    )

    config, _ = run_structured.resolve_config(["--resume_run", str(run_dir)])

    assert config["model_path"] == "/saved/model"
    assert config["model_name"] == "sd2"


def test_resume_rejects_cache_sensitive_config_change(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    saved = minimal_config(model_path="/model", seed=1)
    run_structured.atomic_write_json(run_dir / "run_config.json", saved)

    with pytest.raises(ValueError, match="seed"):
        run_structured.validate_resume_config(
            run_dir, minimal_config(model_path="/model", seed=2)
        )

    run_structured.validate_resume_config(
        run_dir, minimal_config(model_path="/model", seed=2, cache=False)
    )


def test_retry_call_retries_until_success(monkeypatch):
    attempts = []
    monkeypatch.setattr(run_structured.time, "sleep", lambda _: None)

    def operation():
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError("temporary")
        return "ok"

    result = run_structured.retry_call(operation, "test", 2, 0, 1)

    assert result == "ok"
    assert len(attempts) == 3


def test_checkpoint_resume_and_prompt_mismatch(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint = run_structured.empty_checkpoint(["one", "two"])
    checkpoint["enhanced_prompts"][0] = "cached"
    run_structured.save_checkpoint(checkpoint_path, checkpoint)

    resumed = run_structured.load_checkpoint(checkpoint_path, ["one", "two"])
    assert resumed["enhanced_prompts"][0] == "cached"

    with pytest.raises(ValueError, match="Prompt file changed"):
        run_structured.load_checkpoint(checkpoint_path, ["different"])


def test_generate_image_uses_valid_cached_file(tmp_path):
    output_path = tmp_path / "cached.png"
    Image.new("RGB", (8, 8), "red").save(output_path)

    class Pipeline:
        def __call__(self, *_args, **_kwargs):
            raise AssertionError("pipeline should not run for a cache hit")

    result = run_structured.generate_image(
        Pipeline(), "prompt", output_path, minimal_config(), seed=42
    )

    assert result == str(output_path)


def test_generate_image_retries_and_saves_atomically(tmp_path, monkeypatch):
    output_path = tmp_path / "generated.png"
    attempts = []
    monkeypatch.setattr(run_structured.time, "sleep", lambda _: None)

    class Result:
        images = [Image.new("RGB", (8, 8), "blue")]

    class Pipeline:
        def __call__(self, *_args, **_kwargs):
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError("temporary")
            return Result()

    result = run_structured.generate_image(
        Pipeline(),
        "prompt",
        output_path,
        minimal_config(max_retries=1),
        seed=42,
    )

    assert result == str(output_path)
    assert run_structured.is_valid_image(output_path)
    assert len(attempts) == 2
    assert not Path(str(output_path) + ".tmp").exists()


def test_main_resumes_completed_images_without_regeneration(tmp_path, monkeypatch):
    prompts_path = tmp_path / "prompts.txt"
    prompts_path.write_text("one\ntwo\n", encoding="utf-8")
    calls = []
    model_loads = []

    class Tokenizer:
        @staticmethod
        def encode(_prompt, **_kwargs):
            return [0, 1]

        @staticmethod
        def decode(_tokens, **_kwargs):
            return "decoded"

    class Result:
        images = [Image.new("RGB", (8, 8), "green")]

    class Pipeline:
        tokenizer = Tokenizer()

        def __call__(self, prompt, **_kwargs):
            calls.append(prompt)
            return Result()

    class ModelModule:
        @staticmethod
        def get_pipe_slow(_model_path, _cuda):
            model_loads.append(1)
            return Pipeline()

    monkeypatch.setattr(run_structured, "load_model_module", lambda _name: ModelModule())

    output_dir = tmp_path / "output"
    run_structured.main(
        [
            "--input_file",
            str(prompts_path),
            "--model_path",
            "/fake/model",
            "--output_dir",
            str(output_dir),
            "--run_name",
            "integration",
            "--width",
            "8",
            "--height",
            "8",
            "--num_inference_steps",
            "1",
        ]
    )
    run_dir = output_dir / "integration"
    assert len(calls) == 4
    assert len(model_loads) == 1
    assert run_structured.load_json(run_dir / "checkpoint.json")["status"] == "completed"

    run_structured.main(["--resume_run", str(run_dir)])

    assert len(calls) == 4
    assert len(model_loads) == 1
    assert run_structured.is_valid_image(run_dir / "baseline_images/0.png")
    assert run_structured.is_valid_image(run_dir / "improved_images/1.png")
