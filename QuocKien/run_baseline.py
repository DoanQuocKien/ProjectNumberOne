from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from pir_baseline import (
    SplitResult,
    TruncatedSVDRecommender,
    build_seen_map,
    build_truth_map,
    evaluate_recommender,
    load_splits,
    make_submission,
    popularity_baseline,
    save_json_submission,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a collaborative-filtering baseline for PIR.")
    parser.add_argument("--data-root", type=Path, default=Path(r"d:\CS116\ProjectNumberOne"))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1_000_000)
    parser.add_argument("--components", type=int, default=64)
    parser.add_argument("--train-on-all-used-data", action="store_true")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "outputs" / "submission.json")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def run_smoke_test() -> None:
    train = pl.DataFrame(
        {
            "customer_id": [1, 1, 2, 2, 3, 3],
            "item_id": ["a", "b", "a", "c", "b", "d"],
            "weight": [1.0, 1.0, 1.0, 2.0, 1.0, 1.0],
        }
    )
    validation = pl.DataFrame({"customer_id": [1, 2], "item_id": ["c", "b"], "weight": [1.0, 1.0]})
    model = TruncatedSVDRecommender(n_components=2, random_state=0).fit(train)
    seen_map = build_seen_map(train)
    validation_truth = build_truth_map(validation)
    metrics = evaluate_recommender(model, validation_truth, seen_map, top_k=2)
    print("Smoke test metrics:", metrics)
    sample_submission = make_submission(model, [1, 2, 3, 4], seen_map, top_k=2)
    print("Smoke test submission:", sample_submission)


def main() -> None:
    args = parse_args()

    if args.smoke_test:
        run_smoke_test()
        return

    transaction_path = args.data_root / "transaction_full_2025.parquet"
    items_path = args.data_root / "items.parquet"
    splits: SplitResult = load_splits(transaction_path, items_path, batch_size=args.batch_size)

    if splits.train.empty:
        raise RuntimeError("No training interactions were found after splitting the transaction parquet.")

    print(
        "Interactions loaded:",
        {
            "train_rows": int(len(splits.train)),
            "validation_rows": int(len(splits.validation)),
            "test_rows": int(len(splits.test)),
        },
    )

    model = TruncatedSVDRecommender(n_components=args.components, random_state=42).fit(splits.train)
    train_seen_map = build_seen_map(splits.train)

    validation_truth = build_truth_map(splits.validation)
    test_truth = build_truth_map(splits.test)

    validation_metrics = evaluate_recommender(model, validation_truth, train_seen_map, top_k=args.top_k)
    test_metrics = evaluate_recommender(model, test_truth, train_seen_map, top_k=args.top_k)

    print("Validation metrics:", validation_metrics)
    print("Test metrics:", test_metrics)
    print("Popularity fallback:", popularity_baseline(splits.train, top_k=args.top_k))

    if args.train_on_all_used_data:
        full_used = pl.concat([splits.train, splits.validation, splits.test])
        print("Retraining on all used data rows:", int(len(full_used)))
        final_model = TruncatedSVDRecommender(n_components=args.components, random_state=42).fit(full_used)
        final_seen_map = build_seen_map(full_used)
    else:
        final_model = model
        final_seen_map = train_seen_map

    target_users = sorted(set(splits.train["customer_id"]).union(set(splits.validation["customer_id"])).union(set(splits.test["customer_id"])))
    submission = make_submission(final_model, target_users, final_seen_map, top_k=args.top_k)
    save_json_submission(submission, args.output)
    print(f"Saved submission to {args.output}")


if __name__ == "__main__":
    main()
