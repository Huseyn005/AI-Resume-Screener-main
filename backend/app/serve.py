from __future__ import annotations

import os
import subprocess
import sys

import uvicorn

from app.config import get_settings


def main() -> None:
    settings = get_settings()
    if settings.finetuned_repo:
        download_env = os.environ.copy()
        download_env.pop("HF_HUB_OFFLINE", None)
        download_env.pop("TRANSFORMERS_OFFLINE", None)
        try:
            subprocess.run(
                [sys.executable, "-m", "app.fetch_model"],
                check=True,
                env=download_env,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError:
            print(
                "Fine-tuned model download failed; check the private repo "
                "settings, token, and model files.",
                file=sys.stderr,
            )
            raise SystemExit(1) from None

    settings.hf_token = ""
    os.environ.pop("HF_TOKEN", None)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        workers=1,
        proxy_headers=False,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
