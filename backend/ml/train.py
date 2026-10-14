from __future__ import annotations
import argparse
import random
from collections import defaultdict
import numpy as np
import torch
from sentence_transformers import InputExample, SentenceTransformer
from sentence_transformers.sentence_transformer.losses import (
    MultipleNegativesRankingLoss,
)
from torch.utils.data import DataLoader
from ml.common import BACKEND_ROOT, DATA_DIR, read_jsonl

PRETRAINED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OUTPUT_DIR = BACKEND_ROOT / "models" / "finetuned"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune MiniLM on category-balanced hard triplets."
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-pairs", type=int, default=3000)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.max_pairs < 1:
        parser.error("epochs, batch size, and max-pairs must be positive")
    args.batch_size = min(args.batch_size, 16)
    return args


def round_robin(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_category: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)
    for category_rows in by_category.values():
        random.shuffle(category_rows)
    ordered: list[dict[str, str]] = []
    while any(by_category.values()):
        for category in sorted(by_category):
            if by_category[category]:
                ordered.append(by_category[category].pop())
    return ordered


def main() -> None:
    args = parse_args()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    triplets = read_jsonl(DATA_DIR / "triplets.jsonl")
    if not triplets:
        raise ValueError(
            "No training triplets found; run ml/prepare_data.py first"
        )
    ordered = round_robin(triplets)
    category_count = len({row["category"] for row in ordered})
    batch_size = min(args.batch_size, category_count)
    ordered = ordered[:args.max_pairs]
    examples = [
        InputExample(
            texts=[row["jd"], row["positive"], row["negative"]],
            label=1.0,
        )
        for row in ordered
    ]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training device: {device}")
    model = SentenceTransformer(PRETRAINED_MODEL, device=device)
    loader = DataLoader(
        examples,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=model.smart_batching_collate,
    )
    loss = MultipleNegativesRankingLoss(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for features, labels in loader:
            features = [
                {
                    key: value.to(model.device)
                    if isinstance(value, torch.Tensor) else value
                    for key, value in feature.items()
                }
                for feature in features
            ]
            labels = labels.to(model.device) if labels is not None else None
            optimizer.zero_grad(set_to_none=True)
            batch_loss = loss(features, labels)
            batch_loss.backward()
            optimizer.step()
            total_loss += float(batch_loss.detach().cpu())
        mean_loss = total_loss / len(loader)
        print(
            f"epoch {epoch + 1}/{args.epochs}: "
            f"loss={mean_loss:.6f} ({len(examples)} triplets)"
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(OUTPUT_DIR))
    print(f"Saved fine-tuned model to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
