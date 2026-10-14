from __future__ import annotations
import json
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from app.config import get_settings
from app.embed import load_model
from ml.common import (
    BACKEND_ROOT,
    DATA_DIR,
    ML_ROOT,
    encode_chunks,
    jd_chunks,
    read_jsonl,
    resume_chunks,
    semantic_scores,
)

RESULTS_PATH = ML_ROOT / "results.json"
REPORT_PATH = ML_ROOT / "REPORT.md"
HANDWRITTEN_PATH = ML_ROOT / "handwritten_jds.json"
METRICS = ("NDCG@10", "MAP", "Precision@5")


def ranking_metrics(
        scores: list[float],
        relevant: list[bool],
) -> dict[str, float]:
    order = np.argsort(np.asarray(scores))[::-1]
    ranked = np.asarray(relevant, dtype=bool)[order]
    total_relevant = int(sum(relevant))
    if total_relevant == 0:
        raise ValueError("A ranking query has no relevant candidates")
    cutoff = min(10, len(ranked))
    discounts = 1.0 / np.log2(np.arange(2, cutoff + 2))
    dcg = float(np.sum(ranked[:cutoff] * discounts))
    ideal_count = min(total_relevant, cutoff)
    ideal_dcg = float(np.sum(1.0 / np.log2(np.arange(2, ideal_count + 2))))
    hit_positions = np.flatnonzero(ranked)
    precisions = [
        rank / (position + 1)
        for rank, position in enumerate(hit_positions, start=1)
    ]
    average_precision = float(sum(precisions) / total_relevant)
    return {
        "NDCG@10": dcg / ideal_dcg,
        "MAP": average_precision,
        "Precision@5": float(ranked[:5].sum() / min(5, len(ranked))),
    }


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError("No ranking queries were evaluated")
    return {
        metric: float(np.mean([row[metric] for row in rows]))
        for metric in METRICS
    }


def fit_tfidf(train: list[dict[str, Any]]) -> TfidfVectorizer:
    chunks = [
        chunk for record in train for chunk in resume_chunks(record["text"])
    ]
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=50000,
        min_df=2,
        ngram_range=(1, 2),
    )
    vectorizer.fit(chunks)
    return vectorizer


def tfidf_scores(
        vectorizer: TfidfVectorizer,
        jd: str,
        chunks_by_resume: list[list[str]],
) -> list[float]:
    jd_parts = jd_chunks(jd)
    if not jd_parts:
        return [0.0] * len(chunks_by_resume)
    jd_matrix = vectorizer.transform(jd_parts)
    output: list[float] = []
    for chunks in chunks_by_resume:
        if not chunks:
            output.append(0.0)
            continue
        matrix = vectorizer.transform(chunks)
        similarities = (matrix @ jd_matrix.T).toarray()
        per_chunk = similarities.max(axis=1)
        top = np.sort(per_chunk)[-min(3, len(per_chunk)):]
        output.append(float(top.mean()))
    return output


def aggregate_queries(
        queries: list[dict[str, Any]],
        candidates: list[dict[str, Any]],
        candidate_chunks: list[list[str]],
        fine_tuned: Any,
        fine_embeddings: list[np.ndarray],
        pretrained: Any,
        pretrained_embeddings: list[np.ndarray],
        vectorizer: TfidfVectorizer,
) -> tuple[dict[str, dict[str, float]], list[float], list[bool]]:
    by_ranker: dict[str, list[dict[str, float]]] = {
        "TF-IDF": [], "Pretrained MiniLM": [], "Fine-tuned MiniLM": [],
    }
    score_pairs: list[float] = []
    pair_labels: list[bool] = []
    for query in queries:
        relevant = [
            record["category"] == query["category"] for record in candidates
        ]
        scores = {
            "TF-IDF": tfidf_scores(
                vectorizer, query["text"], candidate_chunks
            ),
            "Fine-tuned MiniLM": semantic_scores(
                fine_tuned, query["text"], fine_embeddings
            ),
            "Pretrained MiniLM": semantic_scores(
                pretrained, query["text"], pretrained_embeddings
            ),
        }
        for name, values in scores.items():
            by_ranker[name].append(ranking_metrics(values, relevant))
        score_pairs.extend(scores["Fine-tuned MiniLM"])
        pair_labels.extend(relevant)
    return (
        {name: mean_metrics(values) for name, values in by_ranker.items()},
        score_pairs,
        pair_labels,
    )


def _handwritten_queries() -> list[dict[str, Any]]:
    with HANDWRITTEN_PATH.open(encoding="utf-8") as source:
        descriptions = json.load(source)
    return [
        {
            "jd_id": f"handwritten-{index:02d}",
            "category": row["category"],
            "text": row["text"],
        }
        for index, row in enumerate(descriptions)
    ]


def _write_report(
        test_metrics: dict[str, dict[str, float]],
        hand_metrics: dict[str, dict[str, float]],
) -> None:
    lines = [
        "# Resume ranking evaluation",
        "",
        "Metrics are macro-averaged over job-description queries. "
        "Candidates are resumes from the corresponding held-out split.",
        "",
        "| Ranker | NDCG@10 | MAP | Precision@5 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, values in test_metrics.items():
        lines.append(
            f"| {name} | {values['NDCG@10']:.4f} | {values['MAP']:.4f} "
            f"| {values['Precision@5']:.4f} |"
        )
    lines.extend([
        "",
        "## Handwritten job descriptions",
        "",
        "Eight manually written descriptions across eight categories "
        "provide an out-of-template sanity check.",
        "",
        "| Ranker | NDCG@10 | MAP | Precision@5 |",
        "| --- | ---: | ---: | ---: |",
    ])
    for name, values in hand_metrics.items():
        lines.append(
            f"| {name} | {values['NDCG@10']:.4f} | {values['MAP']:.4f} "
            f"| {values['Precision@5']:.4f} |"
        )
    lines.extend([
        "",
        "## Fine-tuned score distribution",
        "",
        "Raw semantic cosine scores for relevant and irrelevant test pairs:",
        "",
        "![Relevant versus irrelevant semantic score distributions]"
        "(score_distribution.png)",
        "",
        "## Limitations",
        "",
        "- Job descriptions are synthetic proxies except for the separate "
        "eight-description sanity check.",
        "- Relevance means the resume has the same dataset category; it "
        "does not represent real hiring outcomes.",
        "- The dataset contains 24 coarse categories from a public Kaggle "
        "scrape.",
        "- These results do not prove real-world hiring accuracy.",
        "",
    ])
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def _save_distribution(scores: list[float], labels: list[bool]) -> None:
    values = np.asarray(scores)
    relevant = np.asarray(labels, dtype=bool)
    plt.figure(figsize=(8, 5))
    plt.hist(
        values[relevant],
        bins=30,
        alpha=0.65,
        density=True,
        label="Relevant",
    )
    plt.hist(
        values[~relevant],
        bins=30,
        alpha=0.65,
        density=True,
        label="Irrelevant",
    )
    plt.xlabel("Top-3 mean chunk cosine similarity")
    plt.ylabel("Density")
    plt.title("Fine-tuned model scores on test pairs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ML_ROOT / "score_distribution.png", dpi=160)
    plt.close()


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Evaluation device: {device}")
    train = read_jsonl(DATA_DIR / "train.jsonl")
    test = read_jsonl(DATA_DIR / "test.jsonl")
    test_queries = read_jsonl(DATA_DIR / "test_jds.jsonl")
    handwritten = _handwritten_queries()
    model_dir = BACKEND_ROOT / "models" / "finetuned"
    if not model_dir.is_dir() or not any(model_dir.iterdir()):
        raise FileNotFoundError(
            "Fine-tuned model is missing; run ml/train.py first"
        )
    vectorizer = fit_tfidf(train)
    model = load_model(get_settings())
    test_chunks, test_embeddings = encode_chunks(model, test)
    pretrained = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device=device,
    )
    _pretrained_chunks, pretrained_embeddings = encode_chunks(pretrained, test)
    hand_metrics, hand_scores, hand_labels = aggregate_queries(
        handwritten,
        test,
        test_chunks,
        model,
        test_embeddings,
        pretrained,
        pretrained_embeddings,
        vectorizer,
    )
    test_metrics, test_scores, test_labels = aggregate_queries(
        test_queries,
        test,
        test_chunks,
        model,
        test_embeddings,
        pretrained,
        pretrained_embeddings,
        vectorizer,
    )
    results = {
        "test": test_metrics,
        "handwritten": hand_metrics,
        "query_counts": {
            "test": len(test_queries),
            "handwritten": len(handwritten),
        },
        "score_pairs": {
            "fine_tuned_test": [
                {"score": score, "relevant": label}
                for score, label in zip(test_scores, test_labels)
            ],
            "fine_tuned_handwritten": [
                {"score": score, "relevant": label}
                for score, label in zip(hand_scores, hand_labels)
            ],
        },
    }
    RESULTS_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _save_distribution(test_scores, test_labels)
    _write_report(results["test"], results["handwritten"])
    print(json.dumps({
        "test": results["test"],
        "handwritten": results["handwritten"],
    }, indent=2))


if __name__ == "__main__":
    main()
