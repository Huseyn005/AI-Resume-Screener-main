from __future__ import annotations

import json
from pathlib import Path

WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")
TOKENIZER_FILES = ("tokenizer.json", "vocab.txt")


def model_folder_issues(
        model_dir: Path,
        calibrator_name: str,
) -> list[str]:
    issues: list[str] = []
    modules_path = model_dir / "modules.json"
    if not modules_path.is_file():
        issues.append("modules.json")
        modules = None
    else:
        try:
            modules = json.loads(modules_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            issues.append("modules.json (invalid JSON)")
            modules = None

    if modules is not None:
        if not isinstance(modules, list):
            issues.append("modules.json (expected a list of modules)")
            modules = None

    transformer_found = False
    if modules is not None:
        for index, module in enumerate(modules):
            if not isinstance(module, dict):
                issues.append(f"modules.json module {index} (invalid entry)")
                continue
            module_type = module.get("type", "")
            is_transformer = (
                isinstance(module_type, str)
                and module_type.rsplit(".", maxsplit=1)[-1].casefold()
                == "transformer"
            )
            transformer_found = transformer_found or is_transformer
            module_path = module.get("path")
            if not isinstance(module_path, str):
                issues.append(
                    f"modules.json module {index} path (missing or invalid)"
                )
                continue
            location = (model_dir / module_path).resolve()
            try:
                location.relative_to(model_dir.resolve())
            except ValueError:
                issues.append(
                    f"modules.json module {index} path "
                    f"{module_path!r} (outside model folder)"
                )
                continue
            if not location.is_dir():
                label = module_path or "."
                issues.append(
                    f"modules.json module {index} path {label!r}"
                )
                continue
            if is_transformer:
                if not (location / "config.json").is_file():
                    issues.append(f"{module_path or '.'}/config.json")
                if not any(
                    (location / filename).is_file()
                    for filename in WEIGHT_FILES
                ):
                    issues.append(
                        f"{module_path or '.'}/one of: "
                        + ", ".join(WEIGHT_FILES)
                    )
                if not any(
                    (location / filename).is_file()
                    for filename in TOKENIZER_FILES
                ):
                    issues.append(
                        f"{module_path or '.'}/one of: "
                        + ", ".join(TOKENIZER_FILES)
                    )

    if modules is not None and not transformer_found:
        issues.append("modules.json (Transformer module not found)")
    if not (model_dir / calibrator_name).is_file():
        issues.append(calibrator_name)
    return issues
