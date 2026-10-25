from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import huggingface_hub
import joblib
import pytest
from sklearn.isotonic import IsotonicRegression

from app import config, fetch_model, serve
from app.config import Settings
from app.scoring import load_calibrator

BACKEND_ROOT = Path(__file__).resolve().parents[1]
FLAT_MODULES = [
    {
        "idx": 0,
        "name": "0",
        "path": "",
        "type": "sentence_transformers.models.Transformer",
    },
    {
        "idx": 1,
        "name": "1",
        "path": "1_Pooling",
        "type": "sentence_transformers.models.Pooling",
    },
    {
        "idx": 2,
        "name": "2",
        "path": "2_Normalize",
        "type": "sentence_transformers.models.Normalize",
    },
]


def _write_model_folder(
        model_dir: Path,
        transformer_path: str = "",
        include_weights: bool = True,
        include_calibrator: bool = True,
) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    modules = json.loads(json.dumps(FLAT_MODULES))
    modules[0]["path"] = transformer_path
    if transformer_path:
        modules[1]["path"] = "1_Pooling"
        modules[2]["path"] = "2_Normalize"
    (model_dir / "modules.json").write_text(
        json.dumps(modules), encoding="utf-8"
    )
    (model_dir / "config_sentence_transformers.json").write_text(
        "{}", encoding="utf-8"
    )
    transformer_dir = model_dir / transformer_path
    transformer_dir.mkdir(parents=True, exist_ok=True)
    for filename in (
        "config.json",
        "sentence_bert_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        (transformer_dir / filename).write_text("{}", encoding="utf-8")
    if include_weights:
        (transformer_dir / "model.safetensors").write_bytes(b"weights")
    for module_dir in ("1_Pooling", "2_Normalize"):
        path = model_dir / module_dir
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "README.md").write_text("Model", encoding="utf-8")
    if include_calibrator:
        calibrator = IsotonicRegression(
            y_min=0.0, y_max=1.0, out_of_bounds="clip"
        ).fit([0.1, 0.2], [0, 1])
        joblib.dump(calibrator, model_dir / config.CALIBRATOR_PATH.name)


def _patch_model_paths(monkeypatch, model_dir: Path) -> None:
    monkeypatch.setattr(config, "FINETUNED_DIR", model_dir)
    monkeypatch.setattr(
        config, "CALIBRATOR_PATH", model_dir / "calibrator.pkl"
    )


def test_flat_model_folder_passes_exact_manifest_validation(tmp_path) -> None:
    _write_model_folder(tmp_path)

    assert fetch_model.validate_model_folder(tmp_path) == []


def test_transformer_subfolder_model_folder_passes_validation(
        tmp_path,
) -> None:
    _write_model_folder(tmp_path, transformer_path="0_Transformer")

    assert fetch_model.validate_model_folder(tmp_path) == []


def test_missing_weights_are_reported(tmp_path) -> None:
    _write_model_folder(tmp_path, include_weights=False)

    issues = fetch_model.validate_model_folder(tmp_path)

    assert any("model.safetensors" in issue for issue in issues)
    assert any("pytorch_model.bin" in issue for issue in issues)


def test_missing_calibrator_is_reported(tmp_path) -> None:
    _write_model_folder(tmp_path, include_calibrator=False)

    assert "calibrator.pkl" in fetch_model.validate_model_folder(tmp_path)


def test_invalid_modules_json_is_reported(tmp_path) -> None:
    (tmp_path / "modules.json").write_text("{invalid", encoding="utf-8")

    assert "modules.json (invalid JSON)" in (
        fetch_model.validate_model_folder(tmp_path)
    )


def test_missing_module_path_is_reported(tmp_path) -> None:
    _write_model_folder(tmp_path)
    modules_path = tmp_path / "modules.json"
    modules = json.loads(modules_path.read_text(encoding="utf-8"))
    modules[1]["path"] = "missing_pooling"
    modules_path.write_text(json.dumps(modules), encoding="utf-8")

    assert any(
        "missing_pooling" in issue
        for issue in fetch_model.validate_model_folder(tmp_path)
    )


def test_empty_repo_does_not_download_and_valid_local_model_is_used(
        tmp_path,
        monkeypatch,
) -> None:
    model_dir = tmp_path / "finetuned"
    _patch_model_paths(monkeypatch, model_dir)
    _write_model_folder(model_dir)
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda **_kwargs: pytest.fail("download should not be called"),
    )
    settings = Settings(finetuned_repo="", embed_model="base-model")

    assert fetch_model.fetch_model(settings) is None
    assert settings.model_path() == str(model_dir)
    assert settings.model_source() == "local-finetuned"
    assert load_calibrator(settings) is not None


def test_empty_or_incomplete_local_model_folder_uses_base(
        tmp_path,
        monkeypatch,
) -> None:
    model_dir = tmp_path / "finetuned"
    _patch_model_paths(monkeypatch, model_dir)
    model_dir.mkdir()
    settings = Settings(finetuned_repo="", embed_model="base-model")

    assert settings.model_path() == "base-model"
    assert load_calibrator(settings) is None

    (model_dir / "modules.json").write_text("[]", encoding="utf-8")
    assert settings.model_path() == "base-model"
    assert load_calibrator(settings) is None


def test_private_repo_download_loads_flat_model_and_calibrator(
        tmp_path,
        monkeypatch,
) -> None:
    model_dir = tmp_path / "finetuned"
    _patch_model_paths(monkeypatch, model_dir)
    settings = Settings(
        finetuned_repo="owner/private-model",
        finetuned_revision="approved-commit",
        hf_token="test-token",
    )

    def download(**kwargs):
        assert kwargs == {
            "repo_id": "owner/private-model",
            "revision": "approved-commit",
            "token": "test-token",
            "local_dir": str(model_dir),
        }
        _write_model_folder(model_dir)
        return str(model_dir)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)

    assert fetch_model.fetch_model(settings) == model_dir
    assert settings.model_path() == str(model_dir)
    assert load_calibrator(settings) is not None


def test_download_error_has_clear_failure_message(
        tmp_path,
        monkeypatch,
) -> None:
    _patch_model_paths(monkeypatch, tmp_path / "finetuned")

    def fail_download(**_kwargs):
        raise OSError("hub unavailable")

    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", fail_download
    )

    with pytest.raises(RuntimeError, match="Could not download"):
        fetch_model.fetch_model(Settings(finetuned_repo="owner/model"))


def test_fetch_fails_when_calibrator_is_missing(tmp_path, monkeypatch) -> None:
    model_dir = tmp_path / "finetuned"
    _patch_model_paths(monkeypatch, model_dir)
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda **_kwargs: _write_model_folder(
            model_dir, include_calibrator=False
        ),
    )

    with pytest.raises(RuntimeError, match="calibrator.pkl"):
        fetch_model.fetch_model(Settings(finetuned_repo="owner/model"))


def test_local_folder_checker_passes_and_fails_without_loading_model(
        tmp_path,
) -> None:
    valid_dir = tmp_path / "valid"
    _write_model_folder(valid_dir)
    command = [
        sys.executable, "-m", "app.fetch_model", "--check", str(valid_dir)
    ]
    passed = subprocess.run(
        command,
        cwd=BACKEND_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert passed.returncode == 0
    assert passed.stdout.strip() == "PASS"

    missing_calibrator_dir = tmp_path / "missing-calibrator"
    _write_model_folder(missing_calibrator_dir, include_calibrator=False)
    failed = subprocess.run(
        [
            sys.executable, "-m", "app.fetch_model", "--check",
            str(missing_calibrator_dir),
        ],
        cwd=BACKEND_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert failed.returncode == 1
    assert "calibrator.pkl" in failed.stderr


def test_importing_local_checker_does_not_import_model_stack() -> None:
    code = (
        "import sys, app.fetch_model; "
        "print(sorted(set(sys.modules).intersection({"
        "'huggingface_hub', 'sentence_transformers', 'transformers', "
        "'torch'})))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=BACKEND_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.strip() == "[]"


def test_importing_serve_does_not_import_model_stack() -> None:
    code = (
        "import sys, app.serve; "
        "print(sorted(set(sys.modules).intersection({"
        "'huggingface_hub', 'sentence_transformers', 'transformers', "
        "'torch', 'app.main'})))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=BACKEND_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.strip() == "[]"


def test_download_subprocess_failure_exits_cleanly_without_token(
        monkeypatch,
        capsys,
) -> None:
    token = "hf-secret-test-token"
    monkeypatch.setattr(
        serve, "get_settings",
        lambda: Settings(finetuned_repo="owner/private", hf_token=token),
    )

    def fail_run(*_args, **_kwargs):
        raise subprocess.CalledProcessError(
            1,
            ["python", "-m", "app.fetch_model", token],
            output=token,
            stderr=token,
        )

    monkeypatch.setattr(serve.subprocess, "run", fail_run)

    with pytest.raises(SystemExit) as exc:
        serve.main()

    assert exc.value.code == 1
    error = capsys.readouterr().err
    assert token not in error
    assert "Fine-tuned model download failed" in error
