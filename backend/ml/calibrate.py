from __future__ import annotations
import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from app.config import CALIBRATOR_PATH, FINETUNED_DIR, get_settings
from app.embed import load_model
from ml.common import (
    DATA_DIR,
    encode_chunks,
    read_jsonl,
    semantic_scores,
)
from ml.evaluate import REPORT_PATH

def reliability_rows(
        predicted: np.ndarray,
        observed: np.ndarray,
) -> list[tuple[str, int, float, float]]:
    edges = np.linspace(0.0, 1.0, 11)
    rows: list[tuple[str, int, float, float]] = []
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (
            (predicted >= low)
            & ((predicted <= high) if index == 9 else (predicted < high))
        )
        if not np.any(mask):
            continue
        rows.append((
            f"{low:.1f}-{high:.1f}",
            int(mask.sum()),
            float(predicted[mask].mean()),
            float(observed[mask].mean()),
        ))
    return rows


def update_report(rows: list[tuple[str, int, float, float]]) -> None:
    if not REPORT_PATH.is_file():
        raise FileNotFoundError(
            f"{REPORT_PATH} is missing; run ml/evaluate.py first"
        )
    report = REPORT_PATH.read_text(encoding="utf-8")
    section = [
        "## Calibration reliability",
        "",
        "Validation pairs are binned by calibrated probability.",
        "",
        "| Predicted probability bin | Pairs | Mean predicted | "
        "Observed rate |",
        "| --- | ---: | ---: | ---: |",
    ]
    section.extend(
        f"| {label} | {count} | {pred:.4f} | "
        f"{rate:.4f} |"
        for label, count, pred, rate in rows
    )
    replacement = "\n".join(section) + "\n\n"
    marker = "## Calibration reliability\n"
    start = report.find(marker)
    if start >= 0:
        end = report.find("\n## ", start + len(marker))
        if end < 0:
            report = report[:start]
        else:
            report = report[:start] + report[end + 1:]
    limitation = report.find("## Limitations\n")
    if limitation < 0:
        report = report.rstrip() + "\n\n" + replacement
    else:
        report = report[:limitation] + replacement + report[limitation:]
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    val = read_jsonl(DATA_DIR / "val.jsonl")
    queries = read_jsonl(DATA_DIR / "val_jds.jsonl")
    if not val or not queries:
        raise ValueError("Validation resumes and JDs are required")
    model = load_model(get_settings(), model_path=str(FINETUNED_DIR))
    _chunks, embeddings = encode_chunks(model, val)
    semantic: list[float] = []
    observed: list[int] = []
    for query in queries:
        semantic.extend(semantic_scores(model, query["text"], embeddings))
        observed.extend(
            int(row["category"] == query["category"]) for row in val
        )
    x = np.asarray(semantic, dtype=np.float64)
    y = np.asarray(observed, dtype=np.int8)
    if len(np.unique(y)) < 2:
        raise ValueError(
            "Calibration requires relevant and irrelevant pairs"
        )
    calibrator = IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        out_of_bounds="clip",
    ).fit(x, y)
    predicted = np.asarray(calibrator.predict(x), dtype=np.float64)
    CALIBRATOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrator, CALIBRATOR_PATH)
    update_report(reliability_rows(predicted, y))
    print(
        f"Saved calibrator to {CALIBRATOR_PATH}; "
        f"fit on {len(x)} validation pairs."
    )


if __name__ == "__main__":
    main()
