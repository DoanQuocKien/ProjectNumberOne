"""
V13 Direct Upgrade from V12.

This file keeps the V12 two-stage architecture:
    multi-source retrieval -> feature matrix -> LightGBM LambdaRank -> top-10.

The upgrade is focused on the actual choke point: candidate generation.
It adds EDA-backed retrieval sources and source-aware ranking features while
preserving the V12 feature family.

All dataframe processing is Polars. LightGBM is used only for the ranker.
Defaults keep the v12-scale comparison. RAM control comes from Polars-first
candidate processing, deterministic hard-negative selection, fold cleanup, and
batched export rather than smaller samples.
"""

from __future__ import annotations

import argparse
import gc
import json
import pickle
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


SEED = 42
ROOT = Path(__file__).resolve().parents[1]
KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")

DEFAULT_TRAIN_SAMPLE_USERS = 60000
DEFAULT_EVAL_SAMPLE_USERS = 40000
DEFAULT_N_NEGATIVES = 150
DEFAULT_INCLUDE_CF = False
DEFAULT_USE_EVENTS = False
DEFAULT_EVAL_APPEND_MISSING_POSITIVES = False
DEFAULT_LIGHTGBM_DEVICE = "cpu"
DEFAULT_LEARNING_RATE = 0.0129
DEFAULT_NUM_LEAVES = 494
DEFAULT_MAX_DEPTH = 13
DEFAULT_MIN_DATA_IN_LEAF = 326
DEFAULT_LAMBDA_L1 = 0.00054
DEFAULT_LAMBDA_L2 = 0.000001
DEFAULT_NUM_BOOST_ROUND = 800
DEFAULT_EARLY_STOPPING_ROUNDS = 50
DEFAULT_CF_CHUNK_SIZE = 750
DEFAULT_CF_MAX_USERS = 20000
DEFAULT_CF_MAX_ITEMS = 12000
DEFAULT_EXPORT_BATCH_USERS = 10000


def find_data_file(filename: str) -> Path:
    """Find a data file locally or in Kaggle inputs.

    Priority:
    1. Repo root / notebook working directory.
    2. /kaggle/input recursive search, for Kaggle notebooks with arbitrary dataset slugs.
    3. Repo-root fallback path, so argparse help remains stable if data is absent.
    """
    local_candidate = ROOT / filename
    if local_candidate.exists():
        return local_candidate

    if KAGGLE_INPUT.exists():
        matches = sorted(KAGGLE_INPUT.rglob(filename))
        if matches:
            return matches[0]

    return local_candidate


TRANSACTION_PATH = find_data_file("transaction_full_2025.parquet")
EVENT_PATH = find_data_file("event_full_2025.parquet")
ITEMS_PATH = find_data_file("items.parquet")
OUTPUT_DIR = (KAGGLE_WORKING / "QuocKien" / "outputs") if KAGGLE_WORKING.exists() else (ROOT / "QuocKien" / "outputs")

DISCRETIONARY_CATS = ["Thời trang", "Đồ chơi & Sách", "Phụ kiện"]
ESSENTIAL_CATS = ["Sữa", "Sữa nước", "Thực phẩm cho bé", "Tã", "Babycare", "Vệ sinh"]

SOURCE_NAMES = [
    "history",
    "replenishment",
    "global",
    "local",
    "svd",
    "i2i",
    "cat_top",
    "cat_top3",
    "local_cat",
    "brand_cat",
    "momentum_cat",
    "event_atc",
    "event_view",
]


def month_cutoff(month: int) -> datetime:
    return datetime(2025, month, 1)


def standardize_age(text: object) -> float:
    raw_text = "" if text is None else str(text).strip()
    clean_text = raw_text.lower()
    if re.search(r"(\*|x\d|cm)", clean_text):
        return 0.5
    if re.search(r"\bb\d{2}\b", clean_text):
        return 18.0
    if "s17" in clean_text:
        return 1.0
    if "110" in clean_text:
        return 5.0
    if "không xác định" in clean_text or "khÃ´ng xÃ¡c" in clean_text or not clean_text:
        return -1.0

    diaper_map = {
        r"\bnb\b": 0.0,
        r"\bss\b": 0.0,
        r"\bsơ sinh\b": 0.0,
        r"\bs\b": 0.25,
        r"\bm\b": 0.6,
        r"\bl\b": 1.2,
        r"\bxl\b": 2.0,
        r"\bxxl\b": 3.5,
    }
    for pattern, val in diaper_map.items():
        if re.search(pattern, clean_text):
            return val

    range_match = re.search(r"(\d+\.?\d*)\s*-\s*(\d+\.?\d*)", clean_text)
    if range_match:
        start, end = float(range_match.group(1)), float(range_match.group(2))
        avg = (start + end) / 2
        if any(x in clean_text for x in ["m", "tháng", "thÃ¡ng"]):
            return round(avg / 12, 3)
        return avg

    month_match = re.search(r"(\d+\.?\d*)\s*(m|tháng|thÃ¡ng)", clean_text)
    if month_match:
        return round(float(month_match.group(1)) / 12, 3)

    year_match = re.search(r"(\d+\.?\d*)\s*(y|t|tuổi|tuá»•i)", clean_text)
    if year_match:
        return float(year_match.group(1))

    pure_num = re.search(r"^(\d+)$", clean_text)
    if pure_num:
        val = float(pure_num.group(1))
        return round(val / 12, 3) if val > 6 else val
    return -1.0


def load_transactions(path: Path) -> pl.DataFrame:
    return (
        pl.scan_parquet(path)
        .select(
            [
                pl.col("customer_id").cast(pl.Int32),
                pl.col("item_id").cast(pl.Utf8),
                pl.col("quantity").cast(pl.Float32).fill_null(1.0),
                pl.col("price").cast(pl.Float32).fill_null(0.0),
                pl.col("location").cast(pl.Int32),
                pl.col("updated_date").cast(pl.Datetime).alias("event_ts"),
            ]
        )
        .filter(pl.col("item_id").is_not_null())
        .with_columns(
            [
                pl.col("event_ts").dt.month().cast(pl.Int8).alias("month"),
                pl.col("event_ts").dt.weekday().cast(pl.Int8).alias("dow"),
            ]
        )
        .collect()
    )


def load_events(path: Path, cutoff: datetime, lookback_days: int, target_users: pl.DataFrame) -> pl.DataFrame:
    if not path.exists():
        return pl.DataFrame(schema={"customer_id": pl.Int32, "item_id": pl.Utf8, "event_type": pl.Utf8, "event_ts": pl.Datetime})
    return (
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
        .filter(pl.col("event_ts") < cutoff)
        .filter(pl.col("event_ts") >= cutoff - pl.duration(days=lookback_days))
        .join(target_users.lazy(), on="customer_id", how="inner")
        .collect()
    )


def load_items(path: Path) -> pl.DataFrame:
    items = (
        pl.scan_parquet(path)
        .select(
            [
                pl.col("item_id").cast(pl.Utf8),
                pl.col("category").cast(pl.Utf8).fill_null("Unknown"),
                pl.col("category_l1").cast(pl.Utf8).fill_null("Unknown"),
                pl.col("category_l2").cast(pl.Utf8).fill_null("Unknown"),
                pl.col("category_l3").cast(pl.Utf8).fill_null("Unknown"),
                pl.col("brand").cast(pl.Utf8).fill_null("Unknown"),
                pl.col("manufacturer").cast(pl.Utf8).fill_null("Unknown"),
                pl.col("size").cast(pl.Utf8).fill_null("Unknown"),
                pl.col("sale_status").cast(pl.Int8).fill_null(-1),
            ]
        )
        .filter(pl.col("item_id").is_not_null())
        .collect()
    )

    for col in ["category", "category_l1", "category_l2", "category_l3", "brand", "manufacturer"]:
        top_vals = items[col].value_counts().sort("count", descending=True).head(254)[col].to_list()
        items = items.with_columns(
            pl.when(pl.col(col).is_in(top_vals)).then(pl.col(col)).otherwise(pl.lit("Other")).alias(col)
        )
        items = items.with_columns(pl.col(col).cast(pl.Categorical).to_physical().cast(pl.Int32).alias(f"{col}_id"))

    size_map = {item_id: standardize_age(size) for item_id, size in items.select(["item_id", "size"]).iter_rows()}
    return items.with_columns(pl.col("item_id").replace(size_map, default=-1.0).cast(pl.Float32).alias("item_age_proxy"))


def source_frame(df: pl.DataFrame, source: str, rank_col: str, score_col: str) -> pl.DataFrame:
    if df.is_empty():
        return pl.DataFrame(
            schema={
                "customer_id": pl.Int32,
                "item_id": pl.Utf8,
                "source": pl.Utf8,
                "source_rank": pl.Int32,
                "source_score": pl.Float32,
            }
        )
    return df.select(
        [
            pl.col("customer_id").cast(pl.Int32),
            pl.col("item_id").cast(pl.Utf8),
            pl.lit(source).alias("source"),
            pl.col(rank_col).cast(pl.Int32).alias("source_rank"),
            pl.col(score_col).cast(pl.Float32).alias("source_score"),
        ]
    )


class V13DirectRetriever:
    def __init__(
        self,
        history_df: pl.DataFrame,
        items_df: pl.DataFrame,
        target_users: pl.DataFrame,
        include_cf: bool = True,
        include_events: bool = False,
        events_df: Optional[pl.DataFrame] = None,
        svd_components: int = 96,
        svd_top_k: int = 70,
        i2i_top_k: int = 90,
        global_top_k: int = 160,
        local_top_k: int = 160,
        cf_chunk_size: int = DEFAULT_CF_CHUNK_SIZE,
        cf_max_users: int = DEFAULT_CF_MAX_USERS,
        cf_max_items: int = DEFAULT_CF_MAX_ITEMS,
    ) -> None:
        self.history_df = history_df
        self.items_df = items_df
        self.target_users = target_users.select("customer_id").unique()
        self.target_user_list = self.target_users["customer_id"].to_list()
        self.max_ts = history_df["event_ts"].max()
        self.include_cf = include_cf
        self.include_events = include_events
        self.events_df = events_df
        self.svd_components = svd_components
        self.svd_top_k = svd_top_k
        self.i2i_top_k = i2i_top_k
        self.global_top_k = global_top_k
        self.local_top_k = local_top_k
        self.cf_chunk_size = cf_chunk_size
        self.cf_max_users = cf_max_users
        self.cf_max_items = cf_max_items

        self.hist_s = history_df.join(self.target_users, on="customer_id", how="inner")
        self.item_l1 = items_df.select(["item_id", "category_l1", "brand"])
        self.item_cat = self.item_l1.select(["item_id", "category_l1"])
        self._user_segments: Optional[pl.DataFrame] = None
        self._hist_s_item_cat: Optional[pl.DataFrame] = None
        self._primary_location: Optional[pl.DataFrame] = None

    def hist_s_item_cat(self) -> pl.DataFrame:
        if self._hist_s_item_cat is None:
            self._hist_s_item_cat = self.hist_s.join(self.item_cat, on="item_id", how="left")
        return self._hist_s_item_cat

    def get_candidates(self) -> pl.DataFrame:
        sources = [
            self.history_candidates(),
            self.replenishment_candidates(),
            self.global_candidates(),
            self.local_candidates(),
            self.category_top_candidates(top_categories=1, source="cat_top"),
            self.category_top_candidates(top_categories=3, source="cat_top3"),
            self.local_category_candidates(),
            self.brand_category_candidates(),
            self.momentum_category_candidates(),
        ]
        if self.include_cf:
            sources.extend(self.cf_candidates())
        if self.include_events and self.events_df is not None and not self.events_df.is_empty():
            sources.extend(self.event_candidates())

        long = pl.concat([src for src in sources if src is not None and not src.is_empty()], how="vertical_relaxed")
        long = self.apply_segment_source_policy(long)
        agg_exprs = [
            pl.col("source_score").max().alias("retr_score"),
            pl.col("source_rank").min().alias("retr_best_rank"),
            pl.len().alias("retr_source_count"),
        ]
        for source in SOURCE_NAMES:
            agg_exprs.append((pl.col("source") == source).max().cast(pl.Int8).alias(f"src_{source}"))
            agg_exprs.append(
                pl.when(pl.col("source") == source)
                .then(pl.col("source_rank"))
                .otherwise(None)
                .min()
                .fill_null(9999)
                .cast(pl.Int32)
                .alias(f"rank_{source}")
            )

        return long.group_by(["customer_id", "item_id"]).agg(agg_exprs)

    def user_segments(self) -> pl.DataFrame:
        """EDA-backed user personas for candidate budget allocation.

        Segment ids are intentionally numeric because they are later consumed by
        Polars/LightGBM without categorical string overhead:
        0 = balanced, 1 = targeted habitual, 2 = active discoverer, 3 = hibernator.
        """
        if self._user_segments is not None:
            return self._user_segments

        hist_items = self.hist_s_item_cat()
        cat_counts = hist_items.group_by(["customer_id", "category_l1"]).agg(pl.len().alias("cat_count"))
        cat_hhi = (
            cat_counts.with_columns((pl.col("cat_count") / pl.col("cat_count").sum().over("customer_id")).alias("share"))
            .with_columns((pl.col("share") * pl.col("share")).alias("share_sq"))
            .group_by("customer_id")
            .agg(
                [
                    pl.col("share_sq").sum().alias("seg_cat_hhi"),
                    pl.col("cat_count").max().alias("seg_anchor_cat_count"),
                    pl.len().alias("seg_category_count"),
                ]
            )
        )
        profile = (
            self.hist_s.group_by("customer_id")
            .agg(
                [
                    pl.len().alias("seg_rows"),
                    pl.col("item_id").n_unique().alias("seg_unique_items"),
                    (pl.lit(self.max_ts) - pl.col("event_ts").max()).dt.total_days().alias("seg_recency_days"),
                    (pl.len() / ((pl.lit(self.max_ts) - pl.col("event_ts").min()).dt.total_days() / 30.0 + 1.0)).alias("seg_velocity_monthly"),
                ]
            )
            .join(cat_hhi, on="customer_id", how="left")
            .with_columns(
                [
                    pl.col("seg_cat_hhi").fill_null(0.0),
                    pl.col("seg_category_count").fill_null(0),
                    pl.col("seg_unique_items").fill_null(0),
                    pl.col("seg_recency_days").fill_null(999.0),
                    pl.col("seg_velocity_monthly").fill_null(0.0),
                ]
            )
            .with_columns(
                pl.when(pl.col("seg_recency_days") >= 120)
                .then(3)
                .when((pl.col("seg_cat_hhi") >= 0.82) & (pl.col("seg_category_count") <= 2))
                .then(1)
                .when((pl.col("seg_unique_items") >= 16) | ((pl.col("seg_velocity_monthly") >= 1.2) & (pl.col("seg_cat_hhi") <= 0.55)))
                .then(2)
                .otherwise(0)
                .cast(pl.Int8)
                .alias("retr_user_segment")
            )
            .select(["customer_id", "retr_user_segment", "seg_cat_hhi", "seg_unique_items", "seg_recency_days", "seg_velocity_monthly"])
        )
        self._user_segments = self.target_users.join(profile, on="customer_id", how="left").with_columns(
            [
                pl.col("retr_user_segment").fill_null(3),
                pl.col("seg_cat_hhi").fill_null(0.0),
                pl.col("seg_unique_items").fill_null(0),
                pl.col("seg_recency_days").fill_null(999.0),
                pl.col("seg_velocity_monthly").fill_null(0.0),
            ]
        )
        return self._user_segments

    def apply_segment_source_policy(self, long: pl.DataFrame) -> pl.DataFrame:
        if long.is_empty():
            return long
        seg = self.user_segments().select(["customer_id", "retr_user_segment"])
        long = long.join(seg, on="customer_id", how="left").with_columns(pl.col("retr_user_segment").fill_null(3))

        budget = (
            pl.when(pl.col("source") == "history")
            .then(
                pl.when(pl.col("retr_user_segment") == 1).then(160)
                .when(pl.col("retr_user_segment") == 2).then(90)
                .when(pl.col("retr_user_segment") == 3).then(80)
                .otherwise(120)
            )
            .when(pl.col("source") == "replenishment")
            .then(
                pl.when(pl.col("retr_user_segment") == 1).then(70)
                .when(pl.col("retr_user_segment") == 2).then(30)
                .when(pl.col("retr_user_segment") == 3).then(60)
                .otherwise(50)
            )
            .when(pl.col("source") == "global")
            .then(
                pl.when(pl.col("retr_user_segment") == 1).then(80)
                .when(pl.col("retr_user_segment") == 2).then(160)
                .when(pl.col("retr_user_segment") == 3).then(160)
                .otherwise(160)
            )
            .when(pl.col("source") == "local")
            .then(
                pl.when(pl.col("retr_user_segment") == 1).then(80)
                .when(pl.col("retr_user_segment") == 2).then(160)
                .when(pl.col("retr_user_segment") == 3).then(160)
                .otherwise(120)
            )
            .when(pl.col("source") == "cat_top3")
            .then(
                pl.when(pl.col("retr_user_segment") == 1).then(18)
                .when(pl.col("retr_user_segment") == 2).then(318)
                .otherwise(218)
            )
            .when(pl.col("source") == "local_cat")
            .then(
                pl.when(pl.col("retr_user_segment") == 1).then(14)
                .otherwise(114)
            )
            .when(pl.col("source") == "brand_cat")
            .then(
                pl.when(pl.col("retr_user_segment") == 1).then(60)
                .when(pl.col("retr_user_segment") == 2).then(20)
                .otherwise(35)
            )
            .when(pl.col("source") == "momentum_cat")
            .then(
                pl.when(pl.col("retr_user_segment") == 1).then(10)
                .when(pl.col("retr_user_segment") == 2).then(60)
                .when(pl.col("retr_user_segment") == 3).then(60)
                .otherwise(25)
            )
            .when(pl.col("source").is_in(["svd", "i2i"]))
            .then(
                pl.when(pl.col("retr_user_segment") == 1).then(50)
                .when(pl.col("retr_user_segment") == 2).then(120)
                .otherwise(90)
            )
            .when(pl.col("source").is_in(["event_atc", "event_view"]))
            .then(50)
            .otherwise(9999)
        )
        score_adjust = (
            pl.when((pl.col("retr_user_segment") == 1) & pl.col("source").is_in(["history", "replenishment", "brand_cat", "local_cat"]))
            .then(24.0)
            .when((pl.col("retr_user_segment") == 1) & pl.col("source").is_in(["global", "momentum_cat"]))
            .then(-14.0)
            .when((pl.col("retr_user_segment") == 2) & pl.col("source").is_in(["global", "local", "momentum_cat", "svd", "i2i"]))
            .then(16.0)
            .when((pl.col("retr_user_segment") == 2) & (pl.col("source") == "replenishment"))
            .then(-8.0)
            .when((pl.col("retr_user_segment") == 3) & pl.col("source").is_in(["global", "local", "replenishment"]))
            .then(12.0)
            .otherwise(0.0)
        )
        return (
            long.with_columns([budget.cast(pl.Int32).alias("_segment_budget"), score_adjust.alias("_segment_score_adjust")])
            .filter(pl.col("source_rank") <= pl.col("_segment_budget"))
            .with_columns((pl.col("source_score") + pl.col("_segment_score_adjust")).cast(pl.Float32).alias("source_score"))
            .drop(["retr_user_segment", "_segment_budget", "_segment_score_adjust"])
        )

    def history_candidates(self) -> pl.DataFrame:
        ui = (
            self.hist_s.group_by(["customer_id", "item_id"])
            .agg(
                [
                    pl.col("quantity").sum().alias("hist_qty"),
                    pl.len().alias("hist_rows"),
                    pl.col("event_ts").max().alias("last_ts"),
                ]
            )
            .with_columns((pl.lit(self.max_ts) - pl.col("last_ts")).dt.total_days().cast(pl.Float32).alias("days_since"))
            .with_columns(
                (
                    240.0
                    + pl.col("hist_qty").log1p() * 20.0
                    + pl.col("hist_rows").clip(0, 10) * 8.0
                    - pl.col("days_since").clip(0, 240) * 0.45
                ).alias("score")
            )
            .sort(["customer_id", "score"], descending=[False, True])
            .group_by("customer_id", maintain_order=True)
            .head(120)
            .with_columns(pl.int_range(1, pl.len() + 1).over("customer_id").alias("rank"))
        )
        return source_frame(ui, "history", "rank", "score")

    def replenishment_candidates(self) -> pl.DataFrame:
        hist_items = self.history_df.join(self.item_cat, on="item_id", how="left")
        gaps = (
            hist_items.sort(["customer_id", "item_id", "event_ts"])
            .with_columns(pl.col("event_ts").diff().over(["customer_id", "item_id"]).dt.total_days().alias("gap_days"))
            .filter((pl.col("gap_days") >= 2) & (pl.col("gap_days") <= 180))
        )
        item_gap = (
            gaps.group_by("item_id")
            .agg(
                [
                    pl.col("gap_days").median().alias("item_median_gap"),
                    pl.col("gap_days").std().fill_null(0.0).alias("item_gap_std"),
                    pl.len().alias("item_gap_obs"),
                ]
            )
        )
        cat_gap = (
            gaps.group_by("category_l1")
            .agg(
                [
                    pl.col("gap_days").median().alias("cat_median_gap"),
                    pl.col("gap_days").std().fill_null(0.0).alias("cat_gap_std"),
                    pl.len().alias("cat_gap_obs"),
                ]
            )
        )
        repl = (
            self.hist_s.group_by(["customer_id", "item_id"])
            .agg(
                [
                    pl.len().alias("buy_count"),
                    pl.col("event_ts").min().alias("first_buy"),
                    pl.col("event_ts").max().alias("last_buy"),
                ]
            )
            .join(self.item_cat, on="item_id", how="left")
            .join(item_gap, on="item_id", how="left")
            .join(cat_gap, on="category_l1", how="left")
            .with_columns(
                [
                    pl.when(pl.col("buy_count") > 1)
                    .then((pl.col("last_buy") - pl.col("first_buy")).dt.total_days() / (pl.col("buy_count") - 1))
                    .otherwise(None)
                    .alias("user_avg_gap"),
                    (pl.lit(self.max_ts) - pl.col("last_buy")).dt.total_days().alias("days_since"),
                ]
            )
            .with_columns(
                [
                    pl.coalesce(["user_avg_gap", "item_median_gap", "cat_median_gap"]).alias("expected_gap"),
                    pl.coalesce(["item_gap_obs", "cat_gap_obs"]).fill_null(0).alias("repeat_prior_obs"),
                    pl.coalesce(["item_gap_std", "cat_gap_std"]).fill_null(30.0).clip(3, 60).alias("gap_uncertainty"),
                ]
            )
            .filter((pl.col("expected_gap") >= 3) & (pl.col("expected_gap") <= 140))
            .filter((pl.col("buy_count") > 1) | (pl.col("repeat_prior_obs") >= 30))
            .filter(pl.col("days_since") >= pl.col("expected_gap") * 0.55)
            .with_columns(
                (
                    315.0
                    - (pl.col("days_since") - pl.col("expected_gap")).abs().clip(0, 90) * (2.2 - (pl.col("gap_uncertainty") / 60.0).clip(0, 1.2))
                    + pl.col("buy_count").clip(0, 10) * 9.0
                    + pl.col("repeat_prior_obs").log1p().clip(0, 8) * 4.0
                ).alias("score")
            )
            .sort(["customer_id", "score"], descending=[False, True])
            .group_by("customer_id", maintain_order=True)
            .head(70)
            .with_columns(pl.int_range(1, pl.len() + 1).over("customer_id").alias("rank"))
        )
        return source_frame(repl, "replenishment", "rank", "score")

    def global_candidates(self) -> pl.DataFrame:
        global_top = (
            self.history_df.filter(pl.col("event_ts") >= self.max_ts - pl.duration(days=14))
            .group_by("item_id")
            .agg(pl.col("quantity").sum().alias("qty"), pl.len().alias("rows"))
            .sort(["qty", "rows", "item_id"], descending=[True, True, False])
            .head(self.global_top_k)
            .with_row_index("rank", offset=1)
            .with_columns((180.0 - pl.col("rank") * 0.5).alias("score"))
            .select(["item_id", "rank", "score"])
        )
        cands = self.target_users.join(global_top, how="cross")
        return source_frame(cands, "global", "rank", "score")

    def local_candidates(self) -> pl.DataFrame:
        local_top = (
            self.history_df.filter(pl.col("event_ts") >= self.max_ts - pl.duration(days=60))
            .group_by(["location", "item_id"])
            .agg(pl.col("quantity").sum().alias("qty"), pl.len().alias("rows"))
            .sort(["location", "qty", "rows"], descending=[False, True, True])
            .group_by("location", maintain_order=True)
            .head(self.local_top_k)
            .with_columns(pl.int_range(1, pl.len() + 1).over("location").alias("rank"))
            .with_columns((210.0 - pl.col("rank") * 0.7).alias("score"))
            .select(["location", "item_id", "rank", "score"])
        )
        user_loc = self.primary_location()
        cands = user_loc.join(local_top, on="location", how="inner")
        return source_frame(cands, "local", "rank", "score")

    def category_top_candidates(self, top_categories: int, source: str) -> pl.DataFrame:
        user_cat = (
            self.hist_s_item_cat()
            .group_by(["customer_id", "category_l1"])
            .agg(pl.col("quantity").sum().alias("cat_qty"), pl.len().alias("cat_rows"))
            .sort(["customer_id", "cat_qty", "cat_rows"], descending=[False, True, True])
            .group_by("customer_id", maintain_order=True)
            .head(top_categories)
            .with_columns(pl.int_range(1, pl.len() + 1).over("customer_id").alias("user_cat_rank"))
        )
        cat_top = (
            self.history_df.filter(pl.col("event_ts") >= self.max_ts - pl.duration(days=45))
            .join(self.item_cat, on="item_id", how="left")
            .group_by(["category_l1", "item_id"])
            .agg(pl.col("quantity").sum().alias("qty"), pl.len().alias("rows"))
            .sort(["category_l1", "qty", "rows"], descending=[False, True, True])
            .group_by("category_l1", maintain_order=True)
            .head(18)
            .with_columns(pl.int_range(1, pl.len() + 1).over("category_l1").alias("cat_rank"))
        )
        cands = (
            user_cat.join(cat_top, on="category_l1", how="inner")
            .with_columns(
                (
                    205.0 - pl.col("cat_rank") * 1.8 - (pl.col("user_cat_rank") - 1) * 12.0
                ).alias("score")
            )
            .with_columns(((pl.col("user_cat_rank") - 1) * 100 + pl.col("cat_rank")).alias("rank"))
        )
        return source_frame(cands, source, "rank", "score")

    def local_category_candidates(self) -> pl.DataFrame:
        user_loc = self.primary_location()
        user_cat = (
            self.hist_s_item_cat()
            .group_by(["customer_id", "category_l1"])
            .agg(pl.col("quantity").sum().alias("cat_qty"))
            .sort(["customer_id", "cat_qty"], descending=[False, True])
            .group_by("customer_id", maintain_order=True)
            .head(2)
            .with_columns(pl.int_range(1, pl.len() + 1).over("customer_id").alias("user_cat_rank"))
        )
        local_cat_top = (
            self.history_df.filter(pl.col("event_ts") >= self.max_ts - pl.duration(days=75))
            .join(self.item_cat, on="item_id", how="left")
            .group_by(["location", "category_l1", "item_id"])
            .agg(pl.col("quantity").sum().alias("qty"), pl.len().alias("rows"))
            .sort(["location", "category_l1", "qty", "rows"], descending=[False, False, True, True])
            .group_by(["location", "category_l1"], maintain_order=True)
            .head(14)
            .with_columns(pl.int_range(1, pl.len() + 1).over(["location", "category_l1"]).alias("local_cat_rank"))
        )
        cands = (
            user_loc.join(user_cat, on="customer_id", how="inner")
            .join(local_cat_top, on=["location", "category_l1"], how="inner")
            .with_columns(
                (
                    230.0 - pl.col("local_cat_rank") * 1.6 - (pl.col("user_cat_rank") - 1) * 10.0
                ).alias("score")
            )
            .with_columns(((pl.col("user_cat_rank") - 1) * 100 + pl.col("local_cat_rank")).alias("rank"))
        )
        return source_frame(cands, "local_cat", "rank", "score")

    def brand_category_candidates(self) -> pl.DataFrame:
        user_brand = (
            self.hist_s.join(self.item_l1, on="item_id", how="left")
            .group_by(["customer_id", "category_l1", "brand"])
            .agg(pl.col("quantity").sum().alias("brand_qty"), pl.len().alias("brand_rows"))
            .sort(["customer_id", "category_l1", "brand_qty", "brand_rows"], descending=[False, False, True, True])
            .group_by(["customer_id", "category_l1"], maintain_order=True)
            .head(1)
        )
        brand_top = (
            self.history_df.filter(pl.col("event_ts") >= self.max_ts - pl.duration(days=90))
            .join(self.item_l1, on="item_id", how="left")
            .group_by(["category_l1", "brand", "item_id"])
            .agg(pl.col("quantity").sum().alias("qty"), pl.len().alias("rows"))
            .sort(["category_l1", "brand", "qty", "rows"], descending=[False, False, True, True])
            .group_by(["category_l1", "brand"], maintain_order=True)
            .head(10)
            .with_columns(pl.int_range(1, pl.len() + 1).over(["category_l1", "brand"]).alias("brand_rank"))
        )
        cands = (
            user_brand.join(brand_top, on=["category_l1", "brand"], how="inner")
            .with_columns((215.0 - pl.col("brand_rank") * 2.0).alias("score"))
            .rename({"brand_rank": "rank"})
        )
        return source_frame(cands, "brand_cat", "rank", "score")

    def momentum_category_candidates(self) -> pl.DataFrame:
        user_cat = (
            self.hist_s_item_cat()
            .group_by(["customer_id", "category_l1"])
            .agg(pl.col("quantity").sum().alias("cat_qty"))
            .sort(["customer_id", "cat_qty"], descending=[False, True])
            .group_by("customer_id", maintain_order=True)
            .head(2)
        )
        v7 = (
            self.history_df.filter(pl.col("event_ts") >= self.max_ts - pl.duration(days=7))
            .group_by("item_id")
            .agg(pl.len().alias("v7"))
        )
        v28 = (
            self.history_df.filter(pl.col("event_ts") >= self.max_ts - pl.duration(days=28))
            .group_by("item_id")
            .agg(pl.len().alias("v28"))
        )
        momentum = (
            v28.join(v7, on="item_id", how="left")
            .with_columns(pl.col("v7").fill_null(0))
            .with_columns((pl.col("v7") / (pl.col("v28") / 4.0 + 1.0)).alias("momentum"))
            .join(self.item_cat, on="item_id", how="left")
            .sort(["category_l1", "momentum", "v7"], descending=[False, True, True])
            .group_by("category_l1", maintain_order=True)
            .head(8)
            .with_columns(pl.int_range(1, pl.len() + 1).over("category_l1").alias("rank"))
            .with_columns((190.0 + pl.col("momentum").clip(0, 6) * 8.0 - pl.col("rank")).alias("score"))
        )
        cands = user_cat.join(momentum, on="category_l1", how="inner")
        return source_frame(cands, "momentum_cat", "rank", "score")

    def event_candidates(self) -> list[pl.DataFrame]:
        assert self.events_df is not None
        ev = (
            self.events_df.group_by(["customer_id", "item_id", "event_type"])
            .agg(pl.len().alias("event_cnt"), pl.col("event_ts").max().alias("last_event_ts"))
            .with_columns((pl.lit(self.max_ts) - pl.col("last_event_ts")).dt.total_days().cast(pl.Float32).alias("days_since"))
        )
        atc = (
            ev.filter(pl.col("event_type") == "add_to_cart")
            .with_columns((165.0 + pl.col("event_cnt").clip(0, 4) * 20.0 - pl.col("days_since").clip(0, 45) * 1.0).alias("score"))
            .sort(["customer_id", "score"], descending=[False, True])
            .group_by("customer_id", maintain_order=True)
            .head(25)
            .with_columns(pl.int_range(1, pl.len() + 1).over("customer_id").alias("rank"))
        )
        view = (
            ev.filter(pl.col("event_type") == "view_item")
            .filter(pl.col("event_cnt") >= 2)
            .with_columns((115.0 + pl.col("event_cnt").clip(0, 8) * 5.0 - pl.col("days_since").clip(0, 45) * 0.8).alias("score"))
            .sort(["customer_id", "score"], descending=[False, True])
            .group_by("customer_id", maintain_order=True)
            .head(25)
            .with_columns(pl.int_range(1, pl.len() + 1).over("customer_id").alias("rank"))
        )
        return [source_frame(atc, "event_atc", "rank", "score"), source_frame(view, "event_view", "rank", "score")]

    def cf_candidates(self) -> list[pl.DataFrame]:
        hist = self.history_df.filter(pl.col("event_ts") >= self.max_ts - pl.duration(days=180))
        hist = hist.join(self.target_users, on="customer_id", how="inner")
        if hist.is_empty():
            return []

        u_map = hist["customer_id"].unique()
        i_map = self.history_df.filter(pl.col("event_ts") >= self.max_ts - pl.duration(days=180))["item_id"].unique()
        if len(u_map) < 2 or len(i_map) < 3:
            return []
        if len(u_map) > self.cf_max_users or len(i_map) > self.cf_max_items:
            print(
                "Skipping CF candidates to protect RAM: "
                f"users={len(u_map)} items={len(i_map)} "
                f"limits=({self.cf_max_users}, {self.cf_max_items}). "
                "Raise --cf-max-users/--cf-max-items on a high-RAM run."
            )
            return []

        u_df = pl.DataFrame({"customer_id": u_map, "u_idx": np.arange(len(u_map), dtype=np.int32)})
        i_df = pl.DataFrame({"item_id": i_map, "i_idx": np.arange(len(i_map), dtype=np.int32)})
        hist_indexed = hist.join(u_df, on="customer_id", how="inner").join(i_df, on="item_id", how="inner")

        matrix = csr_matrix(
            (
                np.ones(hist_indexed.height, dtype=np.float32),
                (hist_indexed["u_idx"].to_numpy(), hist_indexed["i_idx"].to_numpy()),
            ),
            shape=(len(u_map), len(i_map)),
            dtype=np.float32,
        )
        idx2i = np.array(i_map.to_list())
        target_u = u_df["customer_id"].to_numpy()

        n_comp = min(self.svd_components, len(i_map) - 1)
        svd = TruncatedSVD(n_components=n_comp, random_state=SEED)
        u_emb = svd.fit_transform(matrix)
        i_emb = svd.components_.T
        norm_m = normalize(matrix, norm="l2", axis=0)
        i2i_sim = (norm_m.T.dot(norm_m)).astype(np.float32)
        i2i_sim.setdiag(0)

        svd_frames: list[pl.DataFrame] = []
        i2i_frames: list[pl.DataFrame] = []
        chunk = self.cf_chunk_size
        all_idx = np.arange(len(target_u), dtype=np.int32)
        for start in range(0, len(all_idx), chunk):
            idx = all_idx[start : start + chunk]
            users = target_u[idx]

            scores_svd = u_emb[idx] @ i_emb.T
            top_svd = np.argsort(-scores_svd, axis=1)[:, : self.svd_top_k]
            svd_frames.append(
                pl.DataFrame(
                    {
                        "customer_id": np.repeat(users, self.svd_top_k),
                        "item_id": idx2i[top_svd.flatten()],
                        "rank": np.tile(np.arange(1, self.svd_top_k + 1, dtype=np.int32), len(users)),
                    },
                    schema={"customer_id": pl.Int32, "item_id": pl.Utf8, "rank": pl.Int32},
                ).with_columns((170.0 - pl.col("rank") * 0.8).alias("score"))
            )
            del scores_svd, top_svd

            scores_i2i = matrix[idx].dot(i2i_sim).toarray()
            top_i2i = np.argsort(-scores_i2i, axis=1)[:, : self.i2i_top_k]
            top_scores = np.take_along_axis(scores_i2i, top_i2i, axis=1)
            mask = top_scores > 0
            i2i_frames.append(
                pl.DataFrame(
                    {
                        "customer_id": np.repeat(users, self.i2i_top_k)[mask.flatten()],
                        "item_id": idx2i[top_i2i.flatten()][mask.flatten()],
                        "rank": np.tile(np.arange(1, self.i2i_top_k + 1, dtype=np.int32), len(users))[mask.flatten()],
                    },
                    schema={"customer_id": pl.Int32, "item_id": pl.Utf8, "rank": pl.Int32},
                ).with_columns((175.0 - pl.col("rank") * 0.7).alias("score"))
            )
            del scores_i2i, top_i2i, top_scores, mask

        svd_df = pl.concat(svd_frames, how="vertical_relaxed") if svd_frames else pl.DataFrame()
        i2i_df = pl.concat(i2i_frames, how="vertical_relaxed") if i2i_frames else pl.DataFrame()
        return [source_frame(svd_df, "svd", "rank", "score"), source_frame(i2i_df, "i2i", "rank", "score")]

    def primary_location(self) -> pl.DataFrame:
        if self._primary_location is None:
            self._primary_location = (
                self.hist_s.group_by(["customer_id", "location"])
                .agg(pl.len().alias("loc_rows"), pl.col("event_ts").max().alias("last_ts"))
                .sort(["customer_id", "loc_rows", "last_ts"], descending=[False, True, True])
                .group_by("customer_id", maintain_order=True)
                .head(1)
                .select(["customer_id", "location"])
            )
        return self._primary_location


def select_target_users(history_df: pl.DataFrame, truth_df: Optional[pl.DataFrame], sample_users: Optional[int], all_history_users: bool) -> pl.DataFrame:
    if truth_df is not None and not all_history_users:
        users = truth_df.select("customer_id").unique()
    else:
        users = history_df.select("customer_id").unique()
    if sample_users is not None and sample_users < users.height:
        users = users.sample(n=sample_users, seed=SEED)
    return users


def build_features(history_df: pl.DataFrame, candidates: pl.DataFrame, items_df: pl.DataFrame) -> pl.DataFrame:
    max_ts = history_df["event_ts"].max()
    item_meta = items_df.select(
        [
            "item_id",
            "category_l1",
            "brand",
            "item_age_proxy",
            "sale_status",
            "category_id",
            "category_l1_id",
            "category_l2_id",
            "category_l3_id",
            "brand_id",
            "manufacturer_id",
        ]
    )

    hist_items = history_df.join(item_meta.select(["item_id", "category_l1", "brand", "item_age_proxy"]), on="item_id", how="left")

    u_brand_counts = hist_items.group_by(["customer_id", "brand"]).agg(pl.len().alias("brand_count"))
    u_brand_hhi = (
        u_brand_counts.with_columns((pl.col("brand_count") / pl.col("brand_count").sum().over("customer_id")).alias("share"))
        .with_columns((pl.col("share") * pl.col("share")).alias("share_sq"))
        .group_by("customer_id")
        .agg(pl.col("share_sq").sum().alias("u_brand_hhi"))
    )
    u_cat_counts = hist_items.group_by(["customer_id", "category_l1"]).agg(pl.len().alias("cat_count"))
    u_cat_hhi = (
        u_cat_counts.with_columns((pl.col("cat_count") / pl.col("cat_count").sum().over("customer_id")).alias("share"))
        .with_columns((pl.col("share") * pl.col("share")).alias("share_sq"))
        .group_by("customer_id")
        .agg(pl.col("share_sq").sum().alias("u_cat_hhi"))
    )
    u_avg_age = (
        hist_items.filter(pl.col("item_age_proxy") >= 0)
        .group_by("customer_id")
        .agg(pl.col("item_age_proxy").mean().alias("u_avg_age_proxy"))
    )
    global_avg_age = items_df.filter(pl.col("item_age_proxy") >= 0)["item_age_proxy"].mean()
    if global_avg_age is None:
        global_avg_age = 1.0

    u_loc_stats = history_df.group_by(["customer_id", "location"]).agg(
        [
            pl.len().alias("loc_count"),
            pl.col("event_ts").max().alias("last_ts"),
        ]
    )
    u_loc_hhi = (
        u_loc_stats.with_columns((pl.col("loc_count") / pl.col("loc_count").sum().over("customer_id")).alias("share"))
        .with_columns((pl.col("share") * pl.col("share")).alias("share_sq"))
        .group_by("customer_id")
        .agg(pl.col("share_sq").sum().alias("u_loc_hhi"))
    )

    u_prof = (
        history_df.group_by("customer_id")
        .agg(
            [
                pl.col("item_id").n_unique().alias("u_unique_items"),
                pl.col("quantity").sum().alias("u_total_qty"),
                pl.col("price").mean().alias("u_avg_price"),
                pl.col("price").std().alias("u_price_std"),
                (pl.lit(max_ts) - pl.col("event_ts").min()).dt.total_days().alias("u_tenure_days"),
                (pl.lit(max_ts) - pl.col("event_ts").max()).dt.total_days().alias("u_recency_days"),
                (pl.col("item_id").n_unique() / pl.col("quantity").sum().clip(1)).alias("u_exploration_ratio"),
                (pl.len() / ((pl.lit(max_ts) - pl.col("event_ts").min()).dt.total_days() / 30.0 + 1.0)).alias("u_velocity_monthly"),
            ]
        )
        .join(u_brand_hhi, on="customer_id", how="left")
        .join(u_cat_hhi, on="customer_id", how="left")
        .join(u_avg_age, on="customer_id", how="left")
        .join(u_loc_hhi, on="customer_id", how="left")
        .with_columns(pl.col("u_avg_age_proxy").fill_null(float(global_avg_age)))
        .with_columns(
            pl.when(pl.col("u_recency_days") >= 120)
            .then(3)
            .when((pl.col("u_cat_hhi").fill_null(0.0) >= 0.82) & (pl.col("u_unique_items") <= 10))
            .then(1)
            .when((pl.col("u_unique_items") >= 16) | ((pl.col("u_velocity_monthly") >= 1.2) & (pl.col("u_cat_hhi").fill_null(0.0) <= 0.55)))
            .then(2)
            .otherwise(0)
            .cast(pl.Int8)
            .alias("u_retrieval_segment")
        )
    )

    ui_hist = history_df.group_by(["customer_id", "item_id"]).agg(
        [
            pl.col("quantity").sum().alias("ui_total_qty"),
            (pl.lit(max_ts) - pl.col("event_ts").max()).dt.total_days().alias("ui_recency_days"),
            (pl.col("event_ts").max() - pl.col("event_ts").min()).dt.total_days().alias("ui_buy_duration_days"),
            pl.len().alias("ui_buy_count"),
        ]
    )

    i_repeats = ui_hist.filter(pl.col("ui_buy_count") > 1).group_by("item_id").agg(pl.len().alias("repeat_buyers"))
    i_prof = (
        history_df.group_by("item_id")
        .agg(
            [
                pl.col("customer_id").n_unique().alias("i_unique_users"),
                pl.col("quantity").sum().alias("i_total_qty"),
                pl.col("location").n_unique().alias("i_hubs_count"),
                pl.col("price").median().alias("i_ref_price"),
                (pl.lit(max_ts) - pl.col("event_ts").min()).dt.total_days().alias("i_launch_age_days"),
            ]
        )
        .join(i_repeats, on="item_id", how="left")
        .with_columns((pl.col("repeat_buyers").fill_null(0) / pl.col("i_unique_users").clip(1)).alias("i_repeat_rate"))
        .drop("repeat_buyers")
    )

    repeat_gaps = (
        hist_items.sort(["customer_id", "item_id", "event_ts"])
        .with_columns(pl.col("event_ts").diff().over(["customer_id", "item_id"]).dt.total_days().alias("gap_days"))
        .filter((pl.col("gap_days") >= 2) & (pl.col("gap_days") <= 180))
    )
    item_repeat_prior = (
        repeat_gaps.group_by("item_id")
        .agg(
            [
                pl.col("gap_days").median().alias("item_median_repeat_gap"),
                pl.len().alias("item_repeat_gap_obs"),
            ]
        )
    )
    cat_repeat_prior = (
        repeat_gaps.group_by("category_l1")
        .agg(
            [
                pl.col("gap_days").median().alias("cat_median_repeat_gap"),
                pl.len().alias("cat_repeat_gap_obs"),
            ]
        )
    )

    u_pref_cat = (
        hist_items.group_by(["customer_id", "category_l1"])
        .agg(pl.col("quantity").sum().alias("cat_qty"))
        .sort(["customer_id", "cat_qty"], descending=[False, True])
        .group_by("customer_id", maintain_order=True)
        .head(1)
        .select(["customer_id", "category_l1"])
        .rename({"category_l1": "pref_cat_l1"})
    )
    u_pref_brand = (
        hist_items.group_by(["customer_id", "category_l1", "brand"])
        .agg(pl.col("quantity").sum().alias("brand_qty"))
        .sort(["customer_id", "category_l1", "brand_qty"], descending=[False, False, True])
        .group_by(["customer_id", "category_l1"], maintain_order=True)
        .head(1)
        .select(["customer_id", "category_l1", "brand"])
        .rename({"brand": "pref_brand"})
    )

    v7 = history_df.filter(pl.col("event_ts") >= max_ts - pl.duration(days=7)).group_by("item_id").agg(pl.len().alias("v7"))
    v28 = history_df.filter(pl.col("event_ts") >= max_ts - pl.duration(days=28)).group_by("item_id").agg(pl.len().alias("v28"))
    momentum = (
        v28.join(v7, on="item_id", how="left")
        .with_columns(pl.col("v7").fill_null(0))
        .with_columns((pl.col("v7") / (pl.col("v28") / 4.0 + 1.0)).alias("item_momentum"))
        .select(["item_id", "item_momentum"])
    )

    u_cat = (
        u_cat_counts.with_columns((pl.col("cat_count") / pl.col("cat_count").sum().over("customer_id")).alias("u_cat_affinity"))
        .select(["customer_id", "category_l1", "u_cat_affinity"])
    )

    u_loc = (
        u_loc_stats.sort(["customer_id", "loc_count", "last_ts"], descending=[False, True, True])
        .group_by("customer_id", maintain_order=True)
        .head(1)
        .select(["customer_id", "location"])
    )
    loc_item_pop = history_df.group_by(["location", "item_id"]).agg(pl.len().alias("ui_loc_sales"))

    ds = (
        candidates.join(u_prof, on="customer_id", how="left")
        .join(i_prof, on="item_id", how="left")
        .join(ui_hist, on=["customer_id", "item_id"], how="left")
        .join(item_meta, on="item_id", how="left")
        .join(item_repeat_prior, on="item_id", how="left")
        .join(cat_repeat_prior, on="category_l1", how="left")
        .join(momentum, on="item_id", how="left")
        .join(u_cat, on=["customer_id", "category_l1"], how="left")
        .join(u_pref_cat, on="customer_id", how="left")
        .join(u_pref_brand, on=["customer_id", "category_l1"], how="left")
    )

    ds = (
        ds.with_columns(
            [
                pl.when(pl.col("category_l1") == pl.col("pref_cat_l1")).then(1).otherwise(0).alias("ui_is_primary_cat"),
                pl.when(pl.col("brand") == pl.col("pref_brand")).then(1).otherwise(0).alias("ui_is_preferred_brand"),
            ]
        )
        .drop(["pref_cat_l1", "pref_brand", "brand"])
        .with_columns(
            [
                (pl.col("i_ref_price") - pl.col("u_avg_price")).abs().alias("ui_price_diff"),
                (pl.col("i_ref_price") / (pl.col("u_avg_price") + 1e-5)).alias("ui_price_ratio"),
                (pl.col("item_age_proxy") - pl.col("u_avg_age_proxy")).abs().alias("ui_size_age_diff"),
                (pl.col("item_age_proxy") / (pl.col("u_avg_age_proxy") + 1e-5)).alias("ui_size_age_ratio"),
                pl.coalesce(["item_median_repeat_gap", "cat_median_repeat_gap"]).alias("ui_expected_replenishment_gap"),
            ]
        )
        .with_columns(
            [
                (pl.col("ui_recency_days") / (pl.col("ui_expected_replenishment_gap") + 1e-5)).alias("ui_replenishment_due_ratio"),
                (pl.col("ui_recency_days") - pl.col("ui_expected_replenishment_gap")).abs().alias("ui_replenishment_gap_error"),
            ]
        )
        .join(u_loc, on="customer_id", how="left")
        .join(loc_item_pop, on=["location", "item_id"], how="left")
        .with_columns(pl.col("ui_loc_sales").fill_null(0))
        .with_columns(
            [
                pl.when(pl.col("category_l1").is_in(DISCRETIONARY_CATS) & (pl.col("ui_total_qty").fill_null(0) > 0))
                .then(1)
                .otherwise(0)
                .alias("ui_already_bought_discretionary"),
                pl.when(pl.col("category_l1").is_in(DISCRETIONARY_CATS) & (pl.col("ui_loc_sales") == 0))
                .then(1)
                .otherwise(0)
                .alias("ui_loc_sparsity_penalty"),
                pl.when(pl.col("category_l1").is_in(ESSENTIAL_CATS)).then(1).otherwise(0).alias("is_essential_cat"),
            ]
        )
        .with_columns(
            [
                pl.col("u_retrieval_segment").fill_null(3),
                pl.col("u_recency_days").fill_null(999.0),
                pl.col("u_cat_hhi").fill_null(0.0),
                pl.col("u_brand_hhi").fill_null(0.0),
                pl.col("u_loc_hhi").fill_null(0.0),
            ]
        )
    )

    drop_cols = ["category_l1", "location"]
    numeric_cols = [c for c in ds.columns if c not in ["customer_id", "item_id", "target"] + drop_cols]
    return ds.with_columns(pl.col(numeric_cols).fill_null(0)).drop(drop_cols)


CAT_FEATURES = ["category_id", "category_l1_id", "category_l2_id", "category_l3_id", "brand_id", "manufacturer_id", "sale_status"]
SOURCE_FEATURES = ["retr_score", "retr_best_rank", "retr_source_count"]
for src in SOURCE_NAMES:
    SOURCE_FEATURES.extend([f"src_{src}", f"rank_{src}"])

BASE_FEATURES = [
    "u_unique_items",
    "u_total_qty",
    "u_avg_price",
    "u_price_std",
    "u_tenure_days",
    "u_recency_days",
    "u_exploration_ratio",
    "u_velocity_monthly",
    "u_brand_hhi",
    "u_cat_hhi",
    "u_avg_age_proxy",
    "u_loc_hhi",
    "u_retrieval_segment",
    "i_unique_users",
    "i_total_qty",
    "i_hubs_count",
    "i_ref_price",
    "i_repeat_rate",
    "i_launch_age_days",
    "ui_total_qty",
    "ui_recency_days",
    "ui_buy_duration_days",
    "ui_buy_count",
    "ui_is_primary_cat",
    "ui_is_preferred_brand",
    "ui_price_diff",
    "ui_price_ratio",
    "ui_loc_sales",
    "item_momentum",
    "item_age_proxy",
    "item_median_repeat_gap",
    "item_repeat_gap_obs",
    "cat_median_repeat_gap",
    "cat_repeat_gap_obs",
    "ui_expected_replenishment_gap",
    "ui_replenishment_due_ratio",
    "ui_replenishment_gap_error",
    "u_cat_affinity",
    "ui_size_age_diff",
    "ui_size_age_ratio",
    "ui_already_bought_discretionary",
    "ui_loc_sparsity_penalty",
    "is_essential_cat",
]

ALL_FEATURES = BASE_FEATURES + SOURCE_FEATURES + CAT_FEATURES


def create_dataset_v13(
    history_df: pl.DataFrame,
    truth_df: Optional[pl.DataFrame],
    items_df: pl.DataFrame,
    sample_users: Optional[int],
    n_negatives: Optional[int],
    include_cf: bool,
    include_events: bool,
    event_path: Path,
    mode: str,
    all_history_users: bool = False,
    append_missing_positives: bool = True,
    target_users_override: Optional[pl.DataFrame] = None,
    cf_chunk_size: int = DEFAULT_CF_CHUNK_SIZE,
    cf_max_users: int = DEFAULT_CF_MAX_USERS,
    cf_max_items: int = DEFAULT_CF_MAX_ITEMS,
) -> pl.DataFrame:
    if target_users_override is not None:
        target_users = target_users_override.select("customer_id").unique()
    else:
        target_users = select_target_users(history_df, truth_df, sample_users, all_history_users=all_history_users)
    cutoff = history_df["event_ts"].max() + timedelta(days=1)
    events_df = load_events(event_path, cutoff=cutoff, lookback_days=30, target_users=target_users) if include_events else None

    retriever = V13DirectRetriever(
        history_df=history_df,
        items_df=items_df,
        target_users=target_users,
        include_cf=include_cf,
        include_events=include_events,
        events_df=events_df,
        cf_chunk_size=cf_chunk_size,
        cf_max_users=cf_max_users,
        cf_max_items=cf_max_items,
    )
    candidates = retriever.get_candidates()

    if truth_df is not None:
        truth = truth_df.join(target_users, on="customer_id", how="inner").select(["customer_id", "item_id"]).unique()
        candidates = candidates.join(truth.with_columns(pl.lit(1).cast(pl.Int8).alias("target")), on=["customer_id", "item_id"], how="left").with_columns(pl.col("target").fill_null(0))
        if append_missing_positives and mode == "train":
            missed = truth.join(candidates, on=["customer_id", "item_id"], how="anti")
            if not missed.is_empty():
                source_defaults = {c: 0 for c in SOURCE_FEATURES}
                source_defaults.update({"retr_best_rank": 9999, "retr_score": 0.0, "retr_source_count": 0})
                missed = missed.with_columns([pl.lit(v).alias(k) for k, v in source_defaults.items()])
                missed = missed.with_columns(pl.lit(1).cast(pl.Int8).alias("target"))
                candidates = pl.concat([candidates, missed], how="vertical_relaxed")

        if n_negatives is not None:
            pos = candidates.filter(pl.col("target") == 1)
            neg = (
                candidates.filter(pl.col("target") == 0)
                .sort(["customer_id", "retr_source_count", "retr_score", "retr_best_rank"], descending=[False, True, True, False])
                .group_by("customer_id", maintain_order=True)
                .head(n_negatives)
            )
            candidates = pl.concat([pos, neg], how="vertical_relaxed")
    else:
        candidates = candidates.with_columns(pl.lit(0).cast(pl.Int8).alias("target"))

    candidates = candidates.sort(["customer_id", "target", "retr_score"], descending=[False, True, True])
    return build_features(history_df, candidates, items_df)


def prep_lgb(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = df.sort("customer_id")
    missing = [c for c in ALL_FEATURES if c not in df.columns]
    if missing:
        df = df.with_columns([pl.lit(0).alias(c) for c in missing])
    x = df.select(ALL_FEATURES).to_numpy().astype(np.float32)
    y = df["target"].to_numpy().astype(np.int8)
    group = df.group_by("customer_id", maintain_order=True).len()["len"].to_numpy()
    return x, y, group


def evaluate_scored(df: pl.DataFrame, truth_df: pl.DataFrame, pred_col: str = "pred", top_k: int = 10) -> dict:
    top = (
        df.sort(["customer_id", pred_col], descending=[False, True])
        .group_by("customer_id", maintain_order=True)
        .head(top_k)
        .with_columns(pl.int_range(1, pl.len() + 1).over("customer_id").alias("rank"))
    )
    eval_users = df.select("customer_id").unique()
    truth = truth_df.join(eval_users, on="customer_id", how="inner").select(["customer_id", "item_id"]).unique()
    hits = top.join(truth, on=["customer_id", "item_id"], how="inner")
    target_users = truth.select("customer_id").unique().height
    mrr = (
        hits.group_by("customer_id")
        .agg(pl.col("rank").min().alias("first_rank"))
        .select((1.0 / pl.col("first_rank")).sum())
        .item()
        / max(1, target_users)
    )
    return {
        "target_users": target_users,
        "recommended_users": top.select("customer_id").unique().height,
        "candidate_rows": df.height,
        "hits": hits.height,
        "Precision@10": hits.height / max(1, target_users * top_k),
        "MRR": mrr,
    }


def candidate_diagnostics(df: pl.DataFrame, truth_df: pl.DataFrame, top_k: int = 10) -> dict:
    """Report retrieval coverage and source-level hit contribution.

    This diagnostic intentionally runs before model scoring, so it tells us
    whether the candidate pool is improving. It is the main v13 tuning panel.
    """
    eval_users = df.select("customer_id").unique()
    truth = truth_df.join(eval_users, on="customer_id", how="inner").select(["customer_id", "item_id"]).unique()
    truth_users = truth.select("customer_id").unique()
    retrieved_df = df.filter(pl.col("retr_source_count") > 0) if "retr_source_count" in df.columns else df
    src_cols = [f"src_{source}" for source in SOURCE_NAMES if f"src_{source}" in retrieved_df.columns]
    retrieved_pairs = retrieved_df.select(["customer_id", "item_id"] + src_cols).unique(subset=["customer_id", "item_id"])
    pairs = retrieved_pairs.select(["customer_id", "item_id"])
    hits = pairs.join(truth, on=["customer_id", "item_id"], how="inner")
    hit_pairs = retrieved_pairs.join(
        truth.with_columns(pl.lit(1).cast(pl.Int8).alias("_truth_hit")),
        on=["customer_id", "item_id"],
        how="left",
    ).with_columns(pl.col("_truth_hit").fill_null(0))

    user_candidate_counts = pairs.group_by("customer_id").len().rename({"len": "candidate_count"})
    avg_candidates = user_candidate_counts["candidate_count"].mean() if not user_candidate_counts.is_empty() else 0.0
    p50_candidates = user_candidate_counts["candidate_count"].median() if not user_candidate_counts.is_empty() else 0.0
    p95_candidates = (
        user_candidate_counts.select(pl.col("candidate_count").quantile(0.95)).item()
        if not user_candidate_counts.is_empty()
        else 0.0
    )

    source_metrics = []
    for source in SOURCE_NAMES:
        src_col = f"src_{source}"
        if src_col not in hit_pairs.columns:
            continue
        src_summary = hit_pairs.filter(pl.col(src_col) == 1).select(
            [
                pl.len().alias("pairs"),
                pl.col("_truth_hit").sum().alias("hit_pairs"),
            ]
        )
        src_pair_count = int(src_summary["pairs"][0])
        src_hit_count = int(src_summary["hit_pairs"][0])
        if src_pair_count == 0:
            source_metrics.append(
                {
                    "source": source,
                    "pairs": 0,
                    "avg_pairs_per_target_user": 0.0,
                    "hit_pairs": 0,
                    "pair_recall": 0.0,
                }
            )
            continue
        source_metrics.append(
            {
                "source": source,
                "pairs": src_pair_count,
                "avg_pairs_per_target_user": src_pair_count / max(1, truth_users.height),
                "hit_pairs": src_hit_count,
                "pair_recall": src_hit_count / max(1, truth.height),
            }
        )

    return {
        "target_users": truth_users.height,
        "truth_pairs": truth.height,
        "candidate_pairs": pairs.height,
        "avg_candidates_per_user": float(avg_candidates or 0.0),
        "p50_candidates_per_user": float(p50_candidates or 0.0),
        "p95_candidates_per_user": float(p95_candidates or 0.0),
        "candidate_hit_pairs": hits.height,
        "candidate_recall": hits.height / max(1, truth.height),
        "theoretical_precision_ceiling_at_10": min(hits.height, truth_users.height * top_k) / max(1, truth_users.height * top_k),
        "source_metrics": source_metrics,
    }


def train_eval(args: argparse.Namespace) -> None:
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise SystemExit("lightgbm is required for --mode train-eval. Install it in the high-RAM environment.") from exc

    df_raw = load_transactions(args.transaction_path)
    items_df = load_items(args.items_path)

    folds = [(8, 9), (9, 10), (10, 11)]
    fold_arrays = []
    fold_candidate_reports = []
    for train_end, val_month in folds:
        print(f"Building fold train<=M{train_end} -> M{val_month}")
        val_truth = df_raw.filter(pl.col("month") == val_month)
        fold_set = create_dataset_v13(
            history_df=df_raw.filter(pl.col("month") <= train_end),
            truth_df=val_truth,
            items_df=items_df,
            sample_users=args.train_sample_users,
            n_negatives=args.n_negatives,
            include_cf=args.enable_cf and not args.no_cf,
            include_events=args.use_events,
            event_path=args.event_path,
            mode="train",
            append_missing_positives=True,
            cf_chunk_size=args.cf_chunk_size,
            cf_max_users=args.cf_max_users,
            cf_max_items=args.cf_max_items,
        )
        fold_report = candidate_diagnostics(fold_set, val_truth)
        fold_report.update({"train_end_month": train_end, "validation_month": val_month})
        fold_candidate_reports.append(fold_report)
        print("Candidate diagnostics:")
        print(json.dumps(fold_report, indent=2))
        fold_arrays.append(prep_lgb(fold_set))
        del fold_set, val_truth
        gc.collect()

    (x1, y1, g1), (x2, y2, g2), (x3, y3, g3) = fold_arrays
    x_train = np.vstack([x1, x2])
    y_train = np.concatenate([y1, y2])
    g_train = np.concatenate([g1, g2])

    cat_idx = [ALL_FEATURES.index(c) for c in CAT_FEATURES if c in ALL_FEATURES]
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [10],
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "max_depth": args.max_depth,
        "min_data_in_leaf": args.min_data_in_leaf,
        "lambda_l1": args.lambda_l1,
        "lambda_l2": args.lambda_l2,
        "verbosity": -1,
        "random_state": SEED,
    }
    if args.lightgbm_device != "cpu":
        params.update({"device": args.lightgbm_device, "max_bin": 255})

    dtrain = lgb.Dataset(x_train, y_train, group=g_train, categorical_feature=cat_idx)
    dval = lgb.Dataset(x3, y3, group=g3, categorical_feature=cat_idx, reference=dtrain)
    model = lgb.train(
        params,
        dtrain,
        valid_sets=[dval],
        num_boost_round=args.num_boost_round,
        callbacks=[lgb.early_stopping(args.early_stopping_rounds), lgb.log_evaluation(25)],
    )

    x_final = np.vstack([x1, x2, x3])
    y_final = np.concatenate([y1, y2, y3])
    g_final = np.concatenate([g1, g2, g3])
    final_rounds = model.best_iteration or args.num_boost_round
    del dtrain, dval, x_train, y_train, g_train
    gc.collect()

    final_data = lgb.Dataset(x_final, y_final, group=g_final, categorical_feature=cat_idx)
    final_model = lgb.train(params, final_data, num_boost_round=final_rounds)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / args.model_name
    final_model.save_model(str(model_path))
    del fold_arrays, final_data, x1, x2, x3, y1, y2, y3, g1, g2, g3, x_final, y_final, g_final
    gc.collect()

    print(f"Building M12 evaluation dataset, append_missing_positives={args.eval_append_missing_positives}")
    test_set = create_dataset_v13(
        history_df=df_raw.filter(pl.col("month") <= 11),
        truth_df=df_raw.filter(pl.col("month") == 12),
        items_df=items_df,
        sample_users=args.eval_sample_users,
        n_negatives=None,
        include_cf=args.enable_cf and not args.no_cf,
        include_events=args.use_events,
        event_path=args.event_path,
        mode="train" if args.eval_append_missing_positives else "eval",
        append_missing_positives=args.eval_append_missing_positives,
        cf_chunk_size=args.cf_chunk_size,
        cf_max_users=args.cf_max_users,
        cf_max_items=args.cf_max_items,
    )
    m12_candidate_report = candidate_diagnostics(test_set, df_raw.filter(pl.col("month") == 12))
    print("M12 candidate diagnostics:")
    print(json.dumps(m12_candidate_report, indent=2))
    x_test, _, _ = prep_lgb(test_set)
    test_set = test_set.with_columns(pl.Series("pred", final_model.predict(x_test)))
    metrics = evaluate_scored(test_set, df_raw.filter(pl.col("month") == 12))
    metrics.update(
        {
            "model_path": str(model_path),
            "eval_append_missing_positives": args.eval_append_missing_positives,
            "include_cf": args.enable_cf and not args.no_cf,
            "use_events": args.use_events,
            "eval_sample_users": args.eval_sample_users,
            "fold_candidate_reports": fold_candidate_reports,
            "m12_candidate_report": m12_candidate_report,
        }
    )
    metrics_path = args.output_dir / "v13_direct_eval_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


def append_pickle_records(f, recs: pl.DataFrame) -> None:
    for customer_id, items in recs.iter_rows():
        f.write(pickle.dumps(int(customer_id), protocol=4)[2:-1])
        f.write(pickle.dumps(list(items), protocol=4)[2:-1])
        f.write(b"s")


def export_pickle_stream(recs: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(b"\x80\x04}")  # PROTO 4 + EMPTY_DICT
        append_pickle_records(f, recs)
        f.write(b".")


def export_all(args: argparse.Namespace) -> None:
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise SystemExit("lightgbm is required for --mode export-all. Install it in the high-RAM environment.") from exc
    if not args.confirm_large_export and args.export_limit is None:
        raise SystemExit("Refusing full export without --confirm-large-export. Use --export-limit for smoke tests.")

    df_raw = load_transactions(args.transaction_path)
    items_df = load_items(args.items_path)
    model = lgb.Booster(model_file=str(args.model_path))

    target_users = df_raw.select("customer_id").unique().sort("customer_id")
    if args.export_limit is not None:
        target_users = target_users.head(args.export_limit)

    output_path = args.output_dir / args.output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_users = 0
    total_candidate_rows = 0
    batch_size = args.export_batch_users
    with output_path.open("wb") as f:
        f.write(b"\x80\x04}")  # PROTO 4 + EMPTY_DICT
        for start in range(0, target_users.height, batch_size):
            batch_users = target_users.slice(start, batch_size)
            print(f"Export batch users {start + 1}-{start + batch_users.height} / {target_users.height}")
            export_set = create_dataset_v13(
                history_df=df_raw,
                truth_df=None,
                items_df=items_df,
                sample_users=None,
                n_negatives=None,
                include_cf=args.enable_cf and not args.no_cf,
                include_events=args.use_events,
                event_path=args.event_path,
                mode="inference",
                all_history_users=True,
                target_users_override=batch_users,
                cf_chunk_size=args.cf_chunk_size,
                cf_max_users=args.cf_max_users,
                cf_max_items=args.cf_max_items,
            )
            x, _, _ = prep_lgb(export_set)
            export_set = export_set.with_columns(pl.Series("pred", model.predict(x)))
            top = (
                export_set.sort(["customer_id", "pred"], descending=[False, True])
                .group_by("customer_id", maintain_order=True)
                .head(10)
                .group_by("customer_id", maintain_order=True)
                .agg(pl.col("item_id").alias("recommendations"))
            )
            append_pickle_records(f, top)
            total_users += top.height
            total_candidate_rows += export_set.height
            del export_set, x, top
        f.write(b".")

    meta = {
        "output_path": str(output_path),
        "users": total_users,
        "candidate_rows": total_candidate_rows,
        "avg_candidates_per_user": total_candidate_rows / max(1, total_users),
        "model_path": str(args.model_path),
        "use_events": args.use_events,
        "include_cf": args.enable_cf and not args.no_cf,
        "export_limit": args.export_limit,
        "export_batch_users": args.export_batch_users,
    }
    output_path.with_suffix(".metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="V13 direct upgrade from V12",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=["train-eval", "export-all"], required=True)
    parser.add_argument("--transaction-path", type=Path, default=TRANSACTION_PATH)
    parser.add_argument("--event-path", type=Path, default=EVENT_PATH)
    parser.add_argument("--items-path", type=Path, default=ITEMS_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--model-name", default="v13_direct_lgbm.txt")
    parser.add_argument("--model-path", type=Path, default=OUTPUT_DIR / "v13_direct_lgbm.txt")
    parser.add_argument("--output-name", default="v13_direct_all_users_recommendations.pkl")
    parser.add_argument("--use-events", action="store_true")
    parser.add_argument("--enable-cf", action="store_true", help="Enable SVD/I2I candidates. Off by default because it is the largest RAM risk.")
    parser.add_argument("--no-cf", action="store_true", help="Deprecated safety switch; keeps SVD/I2I disabled.")
    parser.add_argument("--cf-chunk-size", type=int, default=DEFAULT_CF_CHUNK_SIZE, help="CF scoring chunk size.")
    parser.add_argument("--cf-max-users", type=int, default=DEFAULT_CF_MAX_USERS, help="Skip CF above this target-user count.")
    parser.add_argument("--cf-max-items", type=int, default=DEFAULT_CF_MAX_ITEMS, help="Skip CF above this item count.")
    parser.add_argument("--train-sample-users", type=int, default=DEFAULT_TRAIN_SAMPLE_USERS, help="Users sampled for each temporal training fold.")
    parser.add_argument("--eval-sample-users", type=int, default=DEFAULT_EVAL_SAMPLE_USERS, help="Users sampled for Month 12 evaluation.")
    parser.add_argument("--n-negatives", type=int, default=DEFAULT_N_NEGATIVES, help="Hard negatives kept per user after Polars candidate selection.")
    parser.add_argument("--eval-append-missing-positives", action="store_true", help="V12-comparable evaluation candidate set.")
    parser.add_argument("--lightgbm-device", choices=["cpu", "gpu"], default=DEFAULT_LIGHTGBM_DEVICE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--num-leaves", type=int, default=DEFAULT_NUM_LEAVES)
    parser.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH)
    parser.add_argument("--min-data-in-leaf", type=int, default=DEFAULT_MIN_DATA_IN_LEAF)
    parser.add_argument("--lambda-l1", type=float, default=DEFAULT_LAMBDA_L1)
    parser.add_argument("--lambda-l2", type=float, default=DEFAULT_LAMBDA_L2)
    parser.add_argument("--num-boost-round", type=int, default=DEFAULT_NUM_BOOST_ROUND, help="Maximum LightGBM boosting rounds.")
    parser.add_argument("--early-stopping-rounds", type=int, default=DEFAULT_EARLY_STOPPING_ROUNDS, help="LightGBM early stopping rounds.")
    parser.add_argument("--export-limit", type=int, default=None)
    parser.add_argument("--export-batch-users", type=int, default=DEFAULT_EXPORT_BATCH_USERS, help="Users per export batch.")
    parser.add_argument("--confirm-large-export", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "train-eval":
        train_eval(args)
    elif args.mode == "export-all":
        export_all(args)


if __name__ == "__main__":
    main()
