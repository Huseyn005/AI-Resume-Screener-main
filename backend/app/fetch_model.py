from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from app import config
from app.config import Settings, get_settings
from app.model_files import model_folder_issues


def validate_model_folder(model_dir: Path) -> list[str]:
    return model_folder_issues(
        model_dir, config.CALIBRATOR_PATH.name
    )


def fetch_model(settings: Settings | None = None) -> Path | None:
    settings = settings or get_settings()
    if not settings.finetuned_repo:
        return None

    model_dir = config.FINETUNED_DIR
    if model_dir.is_symlink():
        raise RuntimeError(
            f"Refusing to download a model into symlink {model_dir}"
        )
    if model_dir.exists():
        if not model_dir.is_dir():
            raise RuntimeError(
                f"Fine-tuned model destination is not a directory: {model_dir}"
            )
        shutil.rmtree(model_dir)

    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=settings.finetuned_repo,
            revision=settings.finetuned_revision or None,
            token=settings.hf_token or None,
            local_dir=str(model_dir),
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not download the configured fine-tuned model from "
            f"{settings.finetuned_repo!r}."
        ) from exc

    missing = validate_model_folder(model_dir)
    if missing:
        raise RuntimeError(
            "Downloaded fine-tuned model is incomplete; missing or invalid "
            "items: " + ", ".join(missing)
        )
    return model_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        type=Path,
        metavar="FOLDER",
        help="validate a local model folder without downloading or loading it",
    )
    args = parser.parse_args()
    if args.check is not None:
        missing = validate_model_folder(args.check)
        if missing:
            print("FAIL: " + ", ".join(missing), file=sys.stderr)
            return 1
        print("PASS")
        return 0

    try:
        fetch_model()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
