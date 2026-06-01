"""
V13 Polars-only PIR retrieval/rerank pipeline.

Goals:
- Keep evaluation honest: eval/inference never append ground-truth positives.
- Use transaction history as the main signal.
- Add event signals lightly; they are useful as intent features but not trusted as
  the primary engine.
- Export a pickle dictionary: customer_id -> [item_id, ...].
"""

from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import polars as pl


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRANSACTION_PATH = ROOT / "transaction_full_2025.parquet"
DEFAULT_EVENT_PATH = ROOT / "event_full_2025.parquet"
DEFAULT_ITEMS_PATH = ROOT / "items.parquet"
DEFAULT_OUTPUT_DIR = ROOT / "QuocKien" / "outputs"


def _dt(year: int, month: int, day: int = 1) -> datetime:
    return datetime(year, month, day)


def load_transactions(path: Path, cutoff: Optional[datetime] = None) -> pl.DataFrame:
    scan = (
        pl.scan_parquet(path)
        .select(
            [
                pl.col("customer_id").cast(pl.Int32),
                pl.col("item_id").cast(pl.Utf8),
                pl.col("quantity").cast(pl.Float32).fill_null(1.0),
                pl.col("location").cast(pl.Int32),
                pl.col("updated_date").cast(pl.Datetime).alias("event_ts"),
            ]
        )
        .filter(pl.col("item_id").is_not_null())
        .with_columns(pl.col("event_ts").dt.month().cast(pl.Int8).alias("month"))
    )
    if cutoff is not None:
        scan = scan.filter(pl.col("event_ts") < cutoff)
    return scan.collect()


def load_events(
    path: Path,
    event_end: datetime,
    lookback_days: int,
    target_users: Optional[pl.DataFrame] = None,
) -> pl.DataFrame:
    scan = (
        pl.scan_parquet(path)
        .select(
            [
                pl.col("customer_id").cast(pl.Int32),
                pl.col("item_id").cast(pl.Utf8),
                pl.col("event_type").cast(pl.Utf8),
                pl.col("event_date").cast(pl.Datetime).alias("event_ts"),
            ]
        )
        .filter(pl.col("item_id").is_not_null())
        .filter(pl.col("event_ts") < event_end)
        .filter(pl.col("event_ts") >= event_end - pl.duration(days=lookback_days))
    )
    if target_users is not None:
        scan = scan.join(target_users.lazy(), on="customer_id", how="inner")
    return scan.collect()


def load_items(path: Path) -> pl.DataFrame:
    return (
        pl.scan_parquet(path)
        .select(
            [
                pl.col("item_id").cast(pl.Utf8),
                pl.col("category_l1").cast(pl.Utf8).fill_null("Unknown"),
                pl.col("brand").cast(pl.Utf8).fill_null("Unknown"),
                pl.col("sale_status").cast(pl.Int8).fill_null(-1),
            ]
        )
        .filter(pl.col("item_id").is_not_null())
        .collect()
    )


def top_items_global(history: pl.DataFrame, top_n: int) -> pl.DataFrame:
    return (
        history.group_by("item_id")
        .agg(pl.col("quantity").sum().alias("global_qty"), pl.len().alias("global_rows"))
        .sort(["global_qty", "global_rows", "item_id"], descending=[True, True, False])
        .head(top_n)
        .with_row_index("global_rank", offset=1)
        .select(["item_id", "global_rank", "global_qty"])
    )


def make_history_candidates(history: pl.DataFrame, max_ts: datetime, target_users: pl.DataFrame) -> pl.DataFrame:
    ui = (
        history.join(target_users, on="customer_id", how="inner")
        .group_by(["customer_id", "item_id"])
        .agg(
            [
                pl.col("quantity").sum().alias("hist_qty"),
                pl.len().alias("hist_rows"),
                pl.col("event_ts").min().alias("first_ts"),
                pl.col("event_ts").max().alias("last_ts"),
            ]
        )
        .with_columns(
            [
                (pl.lit(max_ts) - pl.col("last_ts")).dt.total_days().cast(pl.Float32).alias("days_since"),
                (
                    (pl.col("last_ts") - pl.col("first_ts")).dt.total_days().cast(pl.Float32)
                    / (pl.col("hist_rows") - 1).clip(1)
                ).alias("avg_gap"),
            ]
        )
        .with_columns(
            [
                (
                    180.0
                    + pl.col("hist_qty").log1p() * 14.0
                    + pl.col("hist_rows").clip(0, 8) * 8.0
                    - pl.col("days_since").clip(0, 180) * 0.45
                ).alias("history_score"),
                (
                    pl.when(
                        (pl.col("hist_rows") >= 2)
                        & (pl.col("avg_gap") >= 3)
                        & (pl.col("avg_gap") <= 120)
                        & (pl.col("days_since") >= pl.col("avg_gap") * 0.65)
                    )
                    .then(
                        110.0
                        - (pl.col("days_since") - pl.col("avg_gap")).abs().clip(0, 60) * 1.2
                        + pl.col("hist_rows").clip(0, 8) * 5.0
                    )
                    .otherwise(0.0)
                ).alias("replenishment_score"),
            ]
        )
        .with_columns((pl.col("history_score") + pl.col("replenishment_score")).alias("score"))
        .select(["customer_id", "item_id", "score"])
    )
    return ui


def make_event_candidates(events: pl.DataFrame, max_ts: datetime) -> pl.DataFrame:
    if events.is_empty():
        return pl.DataFrame(schema={"customer_id": pl.Int32, "item_id": pl.Utf8, "score": pl.Float64})

    return (
        events.group_by(["customer_id", "item_id"])
        .agg(
            [
                (pl.col("event_type") == "add_to_cart").sum().alias("atc_cnt"),
                (pl.col("event_type") == "view_item").sum().alias("view_cnt"),
                pl.col("event_ts").max().alias("last_event_ts"),
            ]
        )
        .with_columns(
            (pl.lit(max_ts) - pl.col("last_event_ts")).dt.total_days().cast(pl.Float32).alias("event_days_since")
        )
        .with_columns(
            (
                75.0
                + pl.col("atc_cnt").clip(0, 4) * 28.0
                + pl.col("view_cnt").clip(0, 8) * 4.0
                - pl.col("event_days_since").clip(0, 45) * 1.1
            ).alias("score")
        )
        .filter((pl.col("atc_cnt") > 0) | (pl.col("view_cnt") >= 2))
        .select(["customer_id", "item_id", "score"])
    )


def make_global_candidates(target_users: pl.DataFrame, global_top: pl.DataFrame, top_k: int) -> pl.DataFrame:
    return (
        target_users.join(global_top.head(top_k), how="cross")
        .with_columns((90.0 - pl.col("global_rank") * 2.0).alias("score"))
        .select(["customer_id", "item_id", "score"])
    )


def make_local_candidates(
    history: pl.DataFrame,
    max_ts: datetime,
    target_users: pl.DataFrame,
    top_n_per_location: int,
) -> pl.DataFrame:
    recent = history.filter(pl.col("event_ts") >= max_ts - pl.duration(days=60))
    if recent.is_empty():
        return pl.DataFrame(schema={"customer_id": pl.Int32, "item_id": pl.Utf8, "score": pl.Float64})

    user_loc = (
        history.join(target_users, on="customer_id", how="inner")
        .group_by(["customer_id", "location"])
        .agg(pl.len().alias("loc_rows"), pl.col("event_ts").max().alias("loc_last_ts"))
        .sort(["customer_id", "loc_rows", "loc_last_ts"], descending=[False, True, True])
        .group_by("customer_id", maintain_order=True)
        .head(1)
        .select(["customer_id", "location"])
    )
    local_top = (
        recent.group_by(["location", "item_id"])
        .agg(pl.col("quantity").sum().alias("local_qty"), pl.len().alias("local_rows"))
        .sort(["location", "local_qty", "local_rows"], descending=[False, True, True])
        .group_by("location", maintain_order=True)
        .head(top_n_per_location)
        .with_columns(pl.int_range(1, pl.len() + 1).over("location").alias("local_rank"))
        .select(["location", "item_id", "local_rank"])
    )
    return (
        user_loc.join(local_top, on="location", how="inner")
        .with_columns((105.0 - pl.col("local_rank") * 1.5).alias("score"))
        .select(["customer_id", "item_id", "score"])
    )


def make_category_candidates(
    history: pl.DataFrame,
    items: pl.DataFrame,
    max_ts: datetime,
    target_users: pl.DataFrame,
    top_n_per_category: int,
) -> pl.DataFrame:
    recent = history.filter(pl.col("event_ts") >= max_ts - pl.duration(days=45))
    if recent.is_empty():
        return pl.DataFrame(schema={"customer_id": pl.Int32, "item_id": pl.Utf8, "score": pl.Float64})

    item_cat = items.select(["item_id", "category_l1"])
    user_cat = (
        history.join(target_users, on="customer_id", how="inner")
        .join(item_cat, on="item_id", how="left")
        .group_by(["customer_id", "category_l1"])
        .agg(pl.col("quantity").sum().alias("cat_qty"), pl.len().alias("cat_rows"))
        .sort(["customer_id", "cat_qty", "cat_rows"], descending=[False, True, True])
        .group_by("customer_id", maintain_order=True)
        .head(1)
        .select(["customer_id", "category_l1"])
    )
    cat_top = (
        recent.join(item_cat, on="item_id", how="left")
        .group_by(["category_l1", "item_id"])
        .agg(pl.col("quantity").sum().alias("cat_item_qty"), pl.len().alias("cat_item_rows"))
        .sort(["category_l1", "cat_item_qty", "cat_item_rows"], descending=[False, True, True])
        .group_by("category_l1", maintain_order=True)
        .head(top_n_per_category)
        .with_columns(pl.int_range(1, pl.len() + 1).over("category_l1").alias("cat_rank"))
        .select(["category_l1", "item_id", "cat_rank"])
    )
    return (
        user_cat.join(cat_top, on="category_l1", how="inner")
        .with_columns((100.0 - pl.col("cat_rank") * 2.0).alias("score"))
        .select(["customer_id", "item_id", "score"])
    )


def rank_candidates(candidates: list[pl.DataFrame], top_k: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    non_empty = [df for df in candidates if df is not None and not df.is_empty()]
    if not non_empty:
        empty = pl.DataFrame(schema={"customer_id": pl.Int32, "item_id": pl.Utf8, "score": pl.Float64})
        return empty, empty

    all_candidates = (
        pl.concat(non_empty, how="vertical_relaxed")
        .group_by(["customer_id", "item_id"])
        .agg(pl.col("score").max().alias("score"))
    )
    top = (
        all_candidates.sort(["customer_id", "score", "item_id"], descending=[False, True, False])
        .group_by("customer_id", maintain_order=True)
        .head(top_k)
        .with_columns(pl.int_range(1, pl.len() + 1).over("customer_id").alias("rank"))
    )
    recs = top.group_by("customer_id", maintain_order=True).agg(pl.col("item_id").alias("recommendations"))
    return top, recs


def generate_recommendations(
    transaction_path: Path,
    event_path: Path,
    items_path: Path,
    target_users: Optional[pl.DataFrame],
    cutoff: Optional[datetime],
    top_k: int,
    use_events: bool,
    rich: bool,
    event_lookback_days: int,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    history = load_transactions(transaction_path, cutoff=cutoff)
    if history.is_empty():
        raise ValueError("No transaction history is available before the requested cutoff.")

    max_ts = history["event_ts"].max()
    if target_users is None:
        target_users = history.select("customer_id").unique()

    global_top = top_items_global(history, max(top_k, 50 if rich else top_k))
    candidates = [
        make_history_candidates(history, max_ts, target_users),
        make_global_candidates(target_users, global_top, top_k),
    ]

    if use_events and event_path.exists():
        event_end = cutoff if cutoff is not None else max_ts + timedelta(days=1)
        events = load_events(event_path, event_end=event_end, lookback_days=event_lookback_days, target_users=target_users)
        candidates.append(make_event_candidates(events, max_ts))

    if rich:
        items = load_items(items_path)
        candidates.append(make_local_candidates(history, max_ts, target_users, top_n_per_location=12))
        candidates.append(make_category_candidates(history, items, max_ts, target_users, top_n_per_category=8))

    top, recs = rank_candidates(candidates, top_k)
    all_candidates = pl.concat([df for df in candidates if df is not None and not df.is_empty()], how="vertical_relaxed")
    all_candidates = all_candidates.select(["customer_id", "item_id"]).unique()
    return top, recs, all_candidates


def evaluate_month(
    transaction_path: Path,
    event_path: Path,
    items_path: Path,
    month: int,
    top_k: int,
    use_events: bool,
    rich: bool,
    event_lookback_days: int,
    sample_users: Optional[int],
) -> dict:
    cutoff = _dt(2025, month, 1)
    truth_scan = (
        pl.scan_parquet(transaction_path)
        .select([pl.col("customer_id").cast(pl.Int32), pl.col("item_id").cast(pl.Utf8), pl.col("updated_date")])
        .with_columns(pl.col("updated_date").dt.month().cast(pl.Int8).alias("month"))
        .filter(pl.col("month") == month)
        .select(["customer_id", "item_id"])
        .unique()
    )
    truth_users = truth_scan.select("customer_id").unique().collect()
    if sample_users is not None and sample_users < truth_users.height:
        truth_users = truth_users.sample(n=sample_users, seed=42)
        truth_scan = truth_scan.join(truth_users.lazy(), on="customer_id", how="inner")
    truth = truth_scan.collect()

    top, recs, candidate_pairs = generate_recommendations(
        transaction_path=transaction_path,
        event_path=event_path,
        items_path=items_path,
        target_users=truth_users,
        cutoff=cutoff,
        top_k=top_k,
        use_events=use_events,
        rich=rich,
        event_lookback_days=event_lookback_days,
    )

    hits = top.join(truth, on=["customer_id", "item_id"], how="inner")
    hit_count = hits.height
    target_user_count = truth_users.height
    precision = hit_count / max(1, target_user_count * top_k)
    mrr = (
        hits.group_by("customer_id")
        .agg(pl.col("rank").min().alias("first_rank"))
        .select((1.0 / pl.col("first_rank")).sum())
        .item()
        / max(1, target_user_count)
    )
    candidate_hits = candidate_pairs.join(truth, on=["customer_id", "item_id"], how="inner").height
    candidate_recall = candidate_hits / max(1, truth.height)

    hist_users = (
        pl.scan_parquet(transaction_path)
        .select([pl.col("customer_id").cast(pl.Int32), pl.col("updated_date")])
        .filter(pl.col("updated_date") < cutoff)
        .select("customer_id")
        .unique()
        .collect()
    )
    warm_users = truth_users.join(hist_users, on="customer_id", how="inner")
    cold_users = truth_users.join(hist_users, on="customer_id", how="anti")

    def segment_precision(users: pl.DataFrame) -> float:
        if users.is_empty():
            return 0.0
        seg_hits = hits.join(users, on="customer_id", how="inner").height
        return seg_hits / (users.height * top_k)

    metrics = {
        "month": month,
        "top_k": top_k,
        "use_events": use_events,
        "rich": rich,
        "sample_users": sample_users,
        "target_users": target_user_count,
        "warm_users": warm_users.height,
        "cold_users": cold_users.height,
        "truth_pairs": truth.height,
        "candidate_pairs": candidate_pairs.height,
        "avg_candidates_per_user": candidate_pairs.height / max(1, target_user_count),
        "candidate_recall": candidate_recall,
        "hits": hit_count,
        "precision_at_10": precision,
        "mrr": mrr,
        "precision_at_10_warm": segment_precision(warm_users),
        "precision_at_10_cold": segment_precision(cold_users),
        "recommended_users": recs.height,
    }
    return metrics


def export_pickle(recs: pl.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        f.write(b"\x80\x04}")  # PROTO 4 + EMPTY_DICT
        for customer_id, items in recs.iter_rows():
            key_bytes = pickle.dumps(int(customer_id), protocol=4)
            val_bytes = pickle.dumps(list(items), protocol=4)
            f.write(key_bytes[2:-1])  # strip PROTO + STOP so it can live inside the dict pickle
            f.write(val_bytes[2:-1])
            f.write(b"s")  # SETITEM
        f.write(b".")  # STOP


def export_all_users(
    transaction_path: Path,
    event_path: Path,
    items_path: Path,
    output_path: Path,
    top_k: int,
    use_events: bool,
    rich: bool,
    event_lookback_days: int,
    export_limit: Optional[int],
) -> dict:
    target_users = None
    if export_limit is not None:
        target_users = (
            pl.scan_parquet(transaction_path)
            .select(pl.col("customer_id").cast(pl.Int32))
            .unique()
            .sort("customer_id")
            .head(export_limit)
            .collect()
        )

    _, recs, candidate_pairs = generate_recommendations(
        transaction_path=transaction_path,
        event_path=event_path,
        items_path=items_path,
        target_users=target_users,
        cutoff=None,
        top_k=top_k,
        use_events=use_events,
        rich=rich,
        event_lookback_days=event_lookback_days,
    )
    export_pickle(recs, output_path)
    meta = {
        "output_path": str(output_path),
        "users": recs.height,
        "top_k": top_k,
        "candidate_pairs": candidate_pairs.height,
        "avg_candidates_per_user": candidate_pairs.height / max(1, recs.height),
        "use_events": use_events,
        "rich": rich,
        "export_limit": export_limit,
    }
    meta_path = output_path.with_suffix(".metadata.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V13 Polars-only PIR pipeline")
    parser.add_argument("--transaction-path", type=Path, default=DEFAULT_TRANSACTION_PATH)
    parser.add_argument("--event-path", type=Path, default=DEFAULT_EVENT_PATH)
    parser.add_argument("--items-path", type=Path, default=DEFAULT_ITEMS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--event-lookback-days", type=int, default=30)
    parser.add_argument("--use-events", action="store_true")
    parser.add_argument("--rich", action="store_true", help="Add local/category candidates. Slower for all-user export.")
    parser.add_argument("--eval-month", type=int, default=None)
    parser.add_argument("--sample-users", type=int, default=None)
    parser.add_argument("--export-all", action="store_true")
    parser.add_argument("--export-limit", type=int, default=None)
    parser.add_argument("--confirm-large-export", action="store_true")
    parser.add_argument("--output-name", default="v13_polars_all_users_recommendations.pkl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.eval_month is not None:
        metrics = evaluate_month(
            transaction_path=args.transaction_path,
            event_path=args.event_path,
            items_path=args.items_path,
            month=args.eval_month,
            top_k=args.top_k,
            use_events=args.use_events,
            rich=args.rich,
            event_lookback_days=args.event_lookback_days,
            sample_users=args.sample_users,
        )
        metrics_path = args.output_dir / f"v13_eval_month_{args.eval_month}.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(json.dumps(metrics, indent=2))

    if args.export_all:
        if args.export_limit is None and not args.confirm_large_export:
            raise SystemExit(
                "Refusing to run full all-user export without --confirm-large-export. "
                "Use --export-limit for a local smoke test, or run the full export on a high-RAM machine."
            )
        output_path = args.output_dir / args.output_name
        meta = export_all_users(
            transaction_path=args.transaction_path,
            event_path=args.event_path,
            items_path=args.items_path,
            output_path=output_path,
            top_k=args.top_k,
            use_events=args.use_events,
            rich=args.rich,
            event_lookback_days=args.event_lookback_days,
            export_limit=args.export_limit,
        )
        print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
