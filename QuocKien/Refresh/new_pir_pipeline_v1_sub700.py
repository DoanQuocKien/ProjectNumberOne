import os
import gc
import json
import numpy as np
import polars as pl
from datetime import datetime
from pathlib import Path
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

# Define base paths
ROOT = Path("d:/CS116/ProjectNumberOne")
TRANSACTION_PATH = ROOT / "transaction_full_2025.parquet"
EVENT_PATH = ROOT / "event_full_2025.parquet"
ITEMS_PATH = ROOT / "items.parquet"

print("Polars & Scikit-Learn libraries imported successfully.")

def load_items() -> pl.DataFrame:
    print("Loading items catalog...")
    return pl.read_parquet(ITEMS_PATH).select([
        pl.col("item_id").cast(pl.Utf8),
        pl.col("category_l1").cast(pl.Utf8).fill_null("Unknown"),
        pl.col("category_l2").cast(pl.Utf8).fill_null("Unknown"),
        pl.col("category_l3").cast(pl.Utf8).fill_null("Unknown"),
        pl.col("brand").cast(pl.Utf8).fill_null("Unknown"),
        pl.col("sale_status").cast(pl.Int8).fill_null(-1),
    ])

def sample_active_december_users(sample_n: int, seed: int = 42) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    print("Sampling active December users for evaluation...")
    dec_df = (
        pl.scan_parquet(TRANSACTION_PATH)
        .select([
            pl.col("customer_id").cast(pl.Int32),
            pl.col("item_id").cast(pl.Utf8),
            pl.col("updated_date").cast(pl.Datetime).alias("event_ts")
        ])
        .filter((pl.col("event_ts") >= datetime(2025, 12, 1)) & (pl.col("event_ts") < datetime(2026, 1, 1)))
        .collect()
    )
    
    unique_users = dec_df.select("customer_id").unique()
    sampled_users = unique_users.sample(n=min(sample_n, unique_users.height), seed=seed)
    
    truth = (
        dec_df.join(sampled_users, on="customer_id", how="inner")
        .select(["customer_id", "item_id"])
        .unique()
    )
    return unique_users, sampled_users, truth

def load_history_data() -> tuple[pl.DataFrame, pl.DataFrame]:
    print("Loading transaction and event history (Months 1-11)...")
    cutoff = datetime(2025, 12, 1)
    
    hist_tx = (
        pl.scan_parquet(TRANSACTION_PATH)
        .select([
            pl.col("customer_id").cast(pl.Int32),
            pl.col("item_id").cast(pl.Utf8),
            pl.col("quantity").cast(pl.Float32).fill_null(1.0),
            pl.col("location").cast(pl.Int32),
            pl.col("updated_date").cast(pl.Datetime).alias("event_ts"),
        ])
        .filter(pl.col("event_ts") < cutoff)
        .collect()
    )
    
    hist_ev = (
        pl.scan_parquet(EVENT_PATH)
        .select([
            pl.col("customer_id").cast(pl.Int32),
            pl.col("item_id").cast(pl.Utf8),
            pl.col("event_type").cast(pl.Utf8),
            pl.col("event_date").cast(pl.Datetime).alias("event_ts"),
        ])
        .filter(pl.col("event_ts") < cutoff)
        .collect()
    )
    
    return hist_tx, hist_ev

# For evaluation stability and speed, sample 5,000 active users
unique_users, target_users, truth = sample_active_december_users(5000, seed=42)
hist_tx, hist_ev = load_history_data()
items = load_items()

def compute_user_segments(hist_tx: pl.DataFrame, items: pl.DataFrame, target_users: pl.DataFrame) -> pl.DataFrame:
    max_ts = hist_tx["event_ts"].max()
    hist_items = hist_tx.join(items.select(["item_id", "category_l1"]), on="item_id", how="left")
    
    cat_counts = hist_items.group_by(["customer_id", "category_l1"]).agg(pl.len().alias("cat_count"))
    cat_hhi = (
        cat_counts.with_columns((pl.col("cat_count") / pl.col("cat_count").sum().over("customer_id")).alias("share"))
        .with_columns((pl.col("share") * pl.col("share")).alias("share_sq"))
        .group_by("customer_id")
        .agg([
            pl.col("share_sq").sum().alias("seg_cat_hhi"),
            pl.col("cat_count").max().alias("seg_anchor_cat_count"),
            pl.len().alias("seg_category_count"),
        ])
    )
    
    profile = (
        hist_tx.group_by("customer_id")
        .agg([
            pl.len().alias("seg_rows"),
            pl.col("item_id").n_unique().alias("seg_unique_items"),
            (pl.lit(max_ts) - pl.col("event_ts").max()).dt.total_days().alias("seg_recency_days"),
            (pl.len() / ((pl.lit(max_ts) - pl.col("event_ts").min()).dt.total_days() / 30.0 + 1.0)).alias("seg_velocity_monthly"),
        ])
        .join(cat_hhi, on="customer_id", how="left")
        .with_columns([
            pl.col("seg_cat_hhi").fill_null(0.0),
            pl.col("seg_category_count").fill_null(0),
            pl.col("seg_unique_items").fill_null(0),
            pl.col("seg_recency_days").fill_null(999.0),
            pl.col("seg_velocity_monthly").fill_null(0.0),
        ])
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
        .select(["customer_id", "retr_user_segment"])
    )
    
    return target_users.join(profile, on="customer_id", how="left").with_columns(
        pl.col("retr_user_segment").fill_null(3)
    )

user_segs = compute_user_segments(hist_tx, items, target_users)

# Get primary location for target users
user_loc = (
    hist_tx.join(target_users, on="customer_id", how="inner")
    .group_by(["customer_id", "location"])
    .agg(pl.len().alias("loc_qty"), pl.col("event_ts").max().alias("last_ts"))
    .sort(["customer_id", "loc_qty", "last_ts"], descending=[False, True, True])
    .group_by("customer_id")
    .head(1)
    .select(["customer_id", "location"])
)

# Map location assortment to determine if item is stocked
location_assortment = (
    hist_tx.select(["location", "item_id"])
    .unique()
    .with_columns(pl.lit(1).alias("is_stocked"))
)

def channel_history(hist_tx: pl.DataFrame, items: pl.DataFrame, target_users: pl.DataFrame) -> pl.DataFrame:
    tx_items = hist_tx.join(items.select(["item_id", "category_l1"]), on="item_id", how="left")
    non_repeatable = ["Thời trang", "Đồ chơi & Sách", "Phụ kiện", "Gói Hội Viên"]
    
    return (
        tx_items.join(target_users, on="customer_id", how="inner")
        .group_by(["customer_id", "item_id", "category_l1"])
        .agg([
            pl.len().alias("purchase_count"),
            pl.col("event_ts").max().alias("last_purchase_ts")
        ])
        .filter(
            ~pl.col("category_l1").is_in(non_repeatable) |
            (pl.col("last_purchase_ts") >= (hist_tx["event_ts"].max() - pl.duration(days=15)))
        )
        .sort(["customer_id", "last_purchase_ts", "purchase_count"], descending=[False, True, True])
        .with_columns(
            pl.int_range(1, pl.len() + 1).over("customer_id").cast(pl.Int64).alias("rank")
        )
        .select(["customer_id", "item_id", "rank"])
    )

cand_history = channel_history(hist_tx, items, target_users)

# SHORT EVALUATION CELL FOR MODEL
hits = cand_history.join(truth, on=["customer_id", "item_id"], how="inner").height
recall = hits / truth.height
avg_size = cand_history.height / target_users.height
print(f"--- Model History Evaluation ---")
print(f"Average Candidates: {avg_size:.2f}")
print(f"Recall Metric      : {recall:.4%}")

def channel_events(hist_ev: pl.DataFrame, target_users: pl.DataFrame, lookback_days: int = 60) -> tuple[pl.DataFrame, pl.DataFrame]:
    if hist_ev.is_empty():
        empty = pl.DataFrame(schema={"customer_id": pl.Int32, "item_id": pl.Utf8, "rank": pl.Int64})
        return empty, empty
    
    max_ts = hist_ev["event_ts"].max()
    cutoff_ts = max_ts - pl.duration(days=lookback_days)
    target_ev = hist_ev.join(target_users, on="customer_id", how="inner")
    
    recent_ev = (
        target_ev.filter(pl.col("event_ts") >= cutoff_ts)
        .group_by(["customer_id", "item_id"])
        .agg([
            pl.len().alias("event_count"),
            pl.col("event_ts").max().alias("last_event_ts")
        ])
        .sort(["customer_id", "last_event_ts", "event_count"], descending=[False, True, True])
        .with_columns(
            pl.int_range(1, pl.len() + 1).over("customer_id").cast(pl.Int64).alias("rank")
        )
        .select(["customer_id", "item_id", "rank"])
    )
    
    all_time_ev = (
        target_ev.group_by(["customer_id", "item_id"])
        .agg([
            pl.len().alias("event_count"),
            pl.col("event_ts").max().alias("last_event_ts")
        ])
        .sort(["customer_id", "last_event_ts", "event_count"], descending=[False, True, True])
        .with_columns(
            pl.int_range(1, pl.len() + 1).over("customer_id").cast(pl.Int64).alias("rank")
        )
        .select(["customer_id", "item_id", "rank"])
    )
    
    return recent_ev, all_time_ev

cand_ev_recent, cand_ev_all = channel_events(hist_ev, target_users, lookback_days=60)

# SHORT EVALUATION CELL FOR MODEL
hits_rec = cand_ev_recent.join(truth, on=["customer_id", "item_id"], how="inner").height
rec_recall = hits_rec / truth.height
print(f"--- Model Clickstream Evaluation (Recent Events) ---")
print(f"Average Candidates: {cand_ev_recent.height / target_users.height:.2f}")
print(f"Recall Metric      : {rec_recall:.4%}")

hits_all = cand_ev_all.join(truth, on=["customer_id", "item_id"], how="inner").height
all_recall = hits_all / truth.height
print(f"\n--- Model Clickstream Evaluation (All-Time Events) ---")
print(f"Average Candidates: {cand_ev_all.height / target_users.height:.2f}")
print(f"Recall Metric      : {all_recall:.4%}")

def channel_category_popular(
    hist_tx: pl.DataFrame, 
    items: pl.DataFrame, 
    target_users: pl.DataFrame,
    top_categories_per_user: int = 5,
    bestsellers_per_category: int = 180,
    lookback_days: int = 45,
    expand_connections: bool = True,
    cat_connections_limit: int = 3
) -> pl.DataFrame:
    item_cat = items.select(["item_id", "category_l1"])
    user_cats = (
        hist_tx.join(target_users, on="customer_id", how="inner")
        .join(item_cat, on="item_id", how="left")
        .filter(pl.col("category_l1") != "Unknown")
        .group_by(["customer_id", "category_l1"])
        .agg(pl.col("quantity").sum().alias("cat_qty"))
        .sort(["customer_id", "cat_qty"], descending=[False, True])
        .group_by("customer_id")
        .head(top_categories_per_user)
        .with_columns(
            pl.int_range(1, pl.len() + 1).over("customer_id").cast(pl.Int64).alias("cat_rank")
        )
        .select(["customer_id", "category_l1", "cat_rank"])
    )
    
    if expand_connections:
        user_cat_pairs = (
            hist_tx.join(item_cat, on="item_id", how="left")
            .filter(pl.col("category_l1") != "Unknown")
            .select(["customer_id", "category_l1"])
            .unique()
        )
        co_occur = (
            user_cat_pairs.join(user_cat_pairs, on="customer_id", suffix="_other")
            .filter(pl.col("category_l1") != pl.col("category_l1_other"))
            .group_by(["category_l1", "category_l1_other"])
            .len()
        )
        cat_links = (
            co_occur.sort(["category_l1", "len"], descending=[False, True])
            .group_by("category_l1")
            .head(cat_connections_limit)
            .select(["category_l1", "category_l1_other"])
        )
        
        connected_user_cats = (
            user_cats.join(cat_links, on="category_l1", how="inner")
            .select([
                "customer_id", 
                pl.col("category_l1_other").alias("category_l1"), 
                (pl.col("cat_rank") + top_categories_per_user).alias("cat_rank")
            ])
        )
        user_cats = (
            pl.concat([user_cats, connected_user_cats])
            .group_by(["customer_id", "category_l1"])
            .agg(pl.col("cat_rank").min())
        )
    
    max_ts = hist_tx["event_ts"].max()
    recent_tx = hist_tx.filter(pl.col("event_ts") >= max_ts - pl.duration(days=lookback_days))
    
    cat_bestsellers = (
        recent_tx.join(item_cat, on="item_id", how="left")
        .filter(pl.col("category_l1") != "Unknown")
        .group_by(["category_l1", "item_id"])
        .agg(pl.col("quantity").sum().alias("qty"))
        .sort(["category_l1", "qty"], descending=[False, True])
        .group_by("category_l1")
        .head(bestsellers_per_category)
        .with_columns(
            pl.int_range(1, pl.len() + 1).over("category_l1").cast(pl.Int64).alias("item_rank")
        )
        .select(["category_l1", "item_id", "item_rank"])
    )
    
    return (
        user_cats.join(cat_bestsellers, on="category_l1", how="inner")
        .sort(["customer_id", "cat_rank", "item_rank"])
        .group_by("customer_id")
        .head(bestsellers_per_category * top_categories_per_user)
        .with_columns(
            pl.int_range(1, pl.len() + 1).over("customer_id").cast(pl.Int64).alias("rank")
        )
        .select(["customer_id", "item_id", "rank"])
    )

cand_category = channel_category_popular(hist_tx, items, target_users)

# SHORT EVALUATION CELL FOR MODEL
hits = cand_category.join(truth, on=["customer_id", "item_id"], how="inner").height
recall = hits / truth.height
print(f"--- Model Category Bestsellers Evaluation ---")
print(f"Average Candidates: {cand_category.height / target_users.height:.2f}")
print(f"Recall Metric      : {recall:.4%}")

def channel_category_trending(
    hist_tx: pl.DataFrame,
    items: pl.DataFrame,
    target_users: pl.DataFrame,
    top_categories_per_user: int = 5,
    trending_per_category: int = 50,
    lookback_days: int = 30
) -> pl.DataFrame:
    item_cat = items.select(["item_id", "category_l1"])
    user_cats = (
        hist_tx.join(target_users, on="customer_id", how="inner")
        .join(item_cat, on="item_id", how="left")
        .filter(pl.col("category_l1") != "Unknown")
        .group_by(["customer_id", "category_l1"])
        .agg(pl.col("quantity").sum().alias("cat_qty"))
        .sort(["customer_id", "cat_qty"], descending=[False, True])
        .group_by("customer_id")
        .head(top_categories_per_user)
        .with_columns(
            pl.int_range(1, pl.len() + 1).over("customer_id").cast(pl.Int64).alias("cat_rank")
        )
        .select(["customer_id", "category_l1", "cat_rank"])
    )
    
    max_ts = hist_tx["event_ts"].max()
    t_recent = max_ts - pl.duration(days=lookback_days)
    t_prior = max_ts - pl.duration(days=lookback_days * 2)
    
    recent_sales = (
        hist_tx.filter(pl.col("event_ts") >= t_recent)
        .group_by("item_id")
        .agg(pl.col("quantity").sum().alias("qty_recent"))
    )
    prior_sales = (
        hist_tx.filter((pl.col("event_ts") >= t_prior) & (pl.col("event_ts") < t_recent))
        .group_by("item_id")
        .agg(pl.col("quantity").sum().alias("qty_prior"))
    )
    
    momentum = (
        recent_sales.join(prior_sales, on="item_id", how="full")
        .fill_null(0.0)
        .with_columns((pl.col("qty_recent") - pl.col("qty_prior")).alias("momentum"))
        .join(item_cat, on="item_id", how="left")
        .filter(pl.col("category_l1") != "Unknown")
    )
    
    cat_trending = (
        momentum.sort(["category_l1", "momentum"], descending=[False, True])
        .group_by("category_l1")
        .head(trending_per_category)
        .with_columns(
            pl.int_range(1, pl.len() + 1).over("category_l1").cast(pl.Int64).alias("item_rank")
        )
        .select(["category_l1", "item_id", "item_rank"])
    )
    
    return (
        user_cats.join(cat_trending, on="category_l1", how="inner")
        .sort(["customer_id", "cat_rank", "item_rank"])
        .group_by("customer_id")
        .head(trending_per_category * top_categories_per_user)
        .with_columns(
            pl.int_range(1, pl.len() + 1).over("customer_id").cast(pl.Int64).alias("rank")
        )
        .select(["customer_id", "item_id", "rank"])
    )

cand_trending = channel_category_trending(hist_tx, items, target_users)

# SHORT EVALUATION CELL FOR MODEL
hits = cand_trending.join(truth, on=["customer_id", "item_id"], how="inner").height
recall = hits / truth.height
print(f"--- Model Category Trending Evaluation ---")
print(f"Average Candidates: {cand_trending.height / target_users.height:.2f}")
print(f"Recall Metric      : {recall:.4%}")

def channel_local_category_popular(
    hist_tx: pl.DataFrame,
    items: pl.DataFrame,
    target_users: pl.DataFrame,
    user_loc: pl.DataFrame,
    top_categories_per_user: int = 5,
    bestsellers_per_loc_cat: int = 40,
    lookback_days: int = 60
) -> pl.DataFrame:
    item_cat = items.select(["item_id", "category_l1"])
    user_cats = (
        hist_tx.join(target_users, on="customer_id", how="inner")
        .join(item_cat, on="item_id", how="left")
        .filter(pl.col("category_l1") != "Unknown")
        .group_by(["customer_id", "category_l1"])
        .agg(pl.col("quantity").sum().alias("cat_qty"))
        .sort(["customer_id", "cat_qty"], descending=[False, True])
        .group_by("customer_id")
        .head(top_categories_per_user)
        .with_columns(
            pl.int_range(1, pl.len() + 1).over("customer_id").cast(pl.Int64).alias("cat_rank")
        )
        .select(["customer_id", "category_l1", "cat_rank"])
    )
    
    user_loc_cat = user_loc.join(user_cats, on="customer_id", how="inner")
    max_ts = hist_tx["event_ts"].max()
    recent_tx = hist_tx.filter(pl.col("event_ts") >= max_ts - pl.duration(days=lookback_days))
    
    loc_cat_bestsellers = (
        recent_tx.join(item_cat, on="item_id", how="left")
        .filter(pl.col("category_l1") != "Unknown")
        .group_by(["location", "category_l1", "item_id"])
        .agg(pl.col("quantity").sum().alias("qty"))
        .sort(["location", "category_l1", "qty"], descending=[False, False, True])
        .group_by(["location", "category_l1"])
        .head(bestsellers_per_loc_cat)
        .with_columns(
            pl.int_range(1, pl.len() + 1).over(["location", "category_l1"]).cast(pl.Int64).alias("item_rank")
        )
        .select(["location", "category_l1", "item_id", "item_rank"])
    )
    
    return (
        user_loc_cat.join(loc_cat_bestsellers, on=["location", "category_l1"], how="inner")
        .sort(["customer_id", "cat_rank", "item_rank"])
        .group_by("customer_id")
        .head(bestsellers_per_loc_cat * top_categories_per_user)
        .with_columns(
            pl.int_range(1, pl.len() + 1).over("customer_id").cast(pl.Int64).alias("rank")
        )
        .select(["customer_id", "item_id", "rank"])
    )

cand_loc_category = channel_local_category_popular(hist_tx, items, target_users, user_loc)

# SHORT EVALUATION CELL FOR MODEL
hits = cand_loc_category.join(truth, on=["customer_id", "item_id"], how="inner").height
recall = hits / truth.height
print(f"--- Model Local Category Evaluation ---")
print(f"Average Candidates: {cand_loc_category.height / target_users.height:.2f}")
print(f"Recall Metric      : {recall:.4%}")

def channel_brand_popular(
    hist_tx: pl.DataFrame,
    items: pl.DataFrame,
    target_users: pl.DataFrame,
    top_brands_per_user: int = 4,
    bestsellers_per_brand: int = 45,
    lookback_days: int = 45
) -> pl.DataFrame:
    item_brand = items.select(["item_id", "brand"])
    user_brands = (
        hist_tx.join(target_users, on="customer_id", how="inner")
        .join(item_brand, on="item_id", how="left")
        .filter((pl.col("brand") != "Unknown") & (pl.col("brand") != "Không xác định"))
        .group_by(["customer_id", "brand"])
        .agg(pl.col("quantity").sum().alias("brand_qty"))
        .sort(["customer_id", "brand_qty"], descending=[False, True])
        .group_by("customer_id")
        .head(top_brands_per_user)
        .with_columns(
            pl.int_range(1, pl.len() + 1).over("customer_id").cast(pl.Int64).alias("brand_rank")
        )
        .select(["customer_id", "brand", "brand_rank"])
    )
    
    max_ts = hist_tx["event_ts"].max()
    recent_tx = hist_tx.filter(pl.col("event_ts") >= max_ts - pl.duration(days=lookback_days))
    
    brand_bestsellers = (
        recent_tx.join(item_brand, on="item_id", how="left")
        .filter((pl.col("brand") != "Unknown") & (pl.col("brand") != "Không xác định"))
        .group_by(["brand", "item_id"])
        .agg(pl.col("quantity").sum().alias("qty"))
        .sort(["brand", "qty"], descending=[False, True])
        .group_by("brand")
        .head(bestsellers_per_brand)
        .with_columns(
            pl.int_range(1, pl.len() + 1).over("brand").cast(pl.Int64).alias("item_rank")
        )
        .select(["brand", "item_id", "item_rank"])
    )
    
    return (
        user_brands.join(brand_bestsellers, on="brand", how="inner")
        .sort(["customer_id", "brand_rank", "item_rank"])
        .group_by("customer_id")
        .head(bestsellers_per_brand * top_brands_per_user)
        .with_columns(
            pl.int_range(1, pl.len() + 1).over("customer_id").cast(pl.Int64).alias("rank")
        )
        .select(["customer_id", "item_id", "rank"])
    )

cand_brand = channel_brand_popular(hist_tx, items, target_users)

# SHORT EVALUATION CELL FOR MODEL
hits = cand_brand.join(truth, on=["customer_id", "item_id"], how="inner").height
recall = hits / truth.height
print(f"--- Model Brand Evaluation ---")
print(f"Average Candidates: {cand_brand.height / target_users.height:.2f}")
print(f"Recall Metric      : {recall:.4%}")

def channel_local_popular(
    hist_tx: pl.DataFrame,
    target_users: pl.DataFrame,
    user_loc: pl.DataFrame,
    bestsellers_per_location: int = 100,
    lookback_days: int = 60
) -> pl.DataFrame:
    max_ts = hist_tx["event_ts"].max()
    recent_tx = hist_tx.filter(pl.col("event_ts") >= max_ts - pl.duration(days=lookback_days))
    
    local_bestsellers = (
        recent_tx.group_by(["location", "item_id"])
        .agg(pl.col("quantity").sum().alias("qty"))
        .sort(["location", "qty"], descending=[False, True])
        .group_by("location")
        .head(bestsellers_per_location)
        .with_columns(
            pl.int_range(1, pl.len() + 1).over("location").cast(pl.Int64).alias("rank")
        )
        .select(["location", "item_id", "rank"])
    )
    
    return (
        user_loc.join(local_bestsellers, on="location", how="inner")
        .select(["customer_id", "item_id", "rank"])
    )

cand_local = channel_local_popular(hist_tx, target_users, user_loc)

# SHORT EVALUATION CELL FOR MODEL
hits = cand_local.join(truth, on=["customer_id", "item_id"], how="inner").height
recall = hits / truth.height
print(f"--- Model Local Warehouse Evaluation ---")
print(f"Average Candidates: {cand_local.height / target_users.height:.2f}")
print(f"Recall Metric      : {recall:.4%}")

def channel_cf_latent(
    hist_tx: pl.DataFrame,
    hist_ev: pl.DataFrame,
    training_users: pl.DataFrame,
    target_users: pl.DataFrame,
    svd_k: int = 180,
    i2i_k: int = 200,
    svd_components: int = 180,
    seed: int = 42
) -> tuple[pl.DataFrame, pl.DataFrame]:
    train_user_ids = training_users["customer_id"].unique().to_list()
    hist_train = hist_tx.filter(pl.col("customer_id").is_in(train_user_ids))
    ev_train = hist_ev.filter(pl.col("customer_id").is_in(train_user_ids))
    
    tx_agg = (
        hist_train.group_by(["customer_id", "item_id"])
        .agg(pl.col("quantity").sum().alias("qty"))
        .with_columns((pl.col("qty") * 10.0).cast(pl.Float32).alias("weight"))
        .select(["customer_id", "item_id", "weight"])
    )
    
    ev_agg = (
        ev_train.group_by(["customer_id", "item_id"])
        .agg(
            (pl.col("event_type") == "add_to_cart").sum().alias("atc"),
            (pl.col("event_type") == "view_item").sum().alias("view")
        )
        .with_columns((pl.col("atc") * 5.0 + pl.col("view") * 1.0).cast(pl.Float32).alias("weight"))
        .select(["customer_id", "item_id", "weight"])
    )
    
    hybrid = (
        pl.concat([tx_agg, ev_agg])
        .group_by(["customer_id", "item_id"])
        .agg(pl.col("weight").sum().alias("weight"))
    )
    
    u_map = hybrid["customer_id"].unique()
    i_map = hist_tx["item_id"].unique()
    
    u_df = pl.DataFrame({"customer_id": u_map, "u_idx": np.arange(len(u_map), dtype=np.int32)})
    i_df = pl.DataFrame({"item_id": i_map, "i_idx": np.arange(len(i_map), dtype=np.int32)})
    
    hybrid_indexed = (
        hybrid.join(u_df, on="customer_id", how="inner")
        .join(i_df, on="item_id", how="inner")
    )
    
    rows = hybrid_indexed["u_idx"].to_numpy()
    cols = hybrid_indexed["i_idx"].to_numpy()
    data = hybrid_indexed["weight"].to_numpy()
    
    mtx = csr_matrix((data, (rows, cols)), shape=(len(u_map), len(i_map)))
    u2idx = dict(zip(u_df["customer_id"], u_df["u_idx"]))
    idx2i = i_map.to_list()
    i_arr = np.array(idx2i)
    
    svd = TruncatedSVD(n_components=svd_components, random_state=seed)
    u_emb = svd.fit_transform(mtx)
    i_emb = svd.components_.T
    
    norm_m = normalize(mtx, norm='l2', axis=0)
    i2i_sim = (norm_m.T.dot(norm_m)).astype(np.float32)
    i2i_sim.setdiag(0)
    
    c_svd, c_i2i = [], []
    chunk_size = 5000
    target_user_ids = target_users["customer_id"].unique().to_list()
    u_indices = [u2idx[u] for u in target_user_ids if u in u2idx]
    users_matched = [u for u in target_user_ids if u in u2idx]
    
    for idx in range(0, len(u_indices), chunk_size):
        chunk_idx = u_indices[idx:idx + chunk_size]
        chunk_users = np.array(users_matched[idx:idx + chunk_size])
        
        scores_svd = u_emb[chunk_idx] @ i_emb.T
        top_svd = np.argsort(-scores_svd, axis=1)[:, :svd_k]
        svd_ranks = np.tile(np.arange(1, svd_k + 1), len(chunk_users))
        c_svd.append(pl.DataFrame({
            "customer_id": pl.Series(np.repeat(chunk_users, svd_k), dtype=pl.Int32),
            "item_id": i_arr[top_svd.flatten()],
            "rank": pl.Series(svd_ranks, dtype=pl.Int64)
        }))
        
        scores_i2i = mtx[chunk_idx].dot(i2i_sim).toarray()
        top_i2i = np.argsort(-scores_i2i, axis=1)[:, :i2i_k]
        mask = np.take_along_axis(scores_i2i, top_i2i, axis=1) > 0.0
        i2i_ranks = np.tile(np.arange(1, i2i_k + 1), len(chunk_users))
        c_svd_len = len(chunk_users)
        c_i2i.append(pl.DataFrame({
            "customer_id": pl.Series(np.repeat(chunk_users, i2i_k)[mask.flatten()], dtype=pl.Int32),
            "item_id": i_arr[top_i2i.flatten()][mask.flatten()],
            "rank": pl.Series(i2i_ranks[mask.flatten()], dtype=pl.Int64)
        }))
        
    df_svd = pl.concat(c_svd).unique() if c_svd else pl.DataFrame(schema={"customer_id": pl.Int32, "item_id": pl.Utf8, "rank": pl.Int64})
    df_i2i = pl.concat(c_i2i).unique() if c_i2i else pl.DataFrame(schema={"customer_id": pl.Int32, "item_id": pl.Utf8, "rank": pl.Int64})
    
    return df_svd, df_i2i

cand_svd, cand_i2i = channel_cf_latent(hist_tx, hist_ev, unique_users, target_users)

# SHORT EVALUATION CELL FOR MODEL
hits_svd = cand_svd.join(truth, on=["customer_id", "item_id"], how="inner").height
recall_svd = hits_svd / truth.height
print(f"--- Model Latent SVD Evaluation ---")
print(f"Average Candidates: {cand_svd.height / target_users.height:.2f}")
print(f"Recall Metric      : {recall_svd:.4%}")

hits_i2i = cand_i2i.join(truth, on=["customer_id", "item_id"], how="inner").height
recall_i2i = hits_i2i / truth.height
print(f"\n--- Model Latent I2I Evaluation ---")
print(f"Average Candidates: {cand_i2i.height / target_users.height:.2f}")
print(f"Recall Metric      : {recall_i2i:.4%}")

def channel_global_popular(hist_tx: pl.DataFrame, target_users: pl.DataFrame, global_k: int = 200) -> pl.DataFrame:
    max_ts = hist_tx["event_ts"].max()
    recent = hist_tx.filter(pl.col("event_ts") >= max_ts - pl.duration(days=30))
    
    global_top = (
        recent.group_by("item_id")
        .agg(pl.col("quantity").sum().alias("qty"))
        .sort("qty", descending=True)
        .head(global_k)
        .with_columns(
            pl.int_range(1, pl.len() + 1).cast(pl.Int64).alias("rank")
        )
        .select(["item_id", "rank"])
    )
    return target_users.join(global_top.with_columns(pl.lit(1).alias("_k")), how="cross").drop("_k")

cand_global = channel_global_popular(hist_tx, target_users)

# SHORT EVALUATION CELL FOR MODEL
hits = cand_global.join(truth, on=["customer_id", "item_id"], how="inner").height
recall = hits / truth.height
print(f"--- Model Global Fallback Evaluation ---")
print(f"Average Candidates: {cand_global.height / target_users.height:.2f}")
print(f"Recall Metric      : {recall:.4%}")

# Optimal candidate channel weights
weights = {
    "A_history": 60.0,
    "B_events_recent": 20.0,
    "B_events_alltime": 10.0,
    "C_category": 50.0,
    "D_brand": 15.0,
    "E_local": 8.0,
    "F_svd": 25.0,
    "G_i2i": 20.0,
    "H_global": 4.0,
    "I_category_trending": 15.0,
    "J_local_category": 20.0
}
rrf_const = 150.0

# Segment budgets mapping
segment_budgets = {0: 500, 1: 200, 2: 1050, 3: 300}
print(f"Applying segment budgets: {segment_budgets}")

# Construct lists
cands = {
    "A_history": cand_history,
    "B_events_recent": cand_ev_recent,
    "B_events_alltime": cand_ev_all,
    "C_category": cand_category,
    "D_brand": cand_brand,
    "E_local": cand_local,
    "F_svd": cand_svd,
    "G_i2i": cand_i2i,
    "H_global": cand_global,
    "I_category_trending": cand_trending,
    "J_local_category": cand_loc_category
}

scored_list = []
for name, df in cands.items():
    if df is not None and df.height > 0:
        w = weights.get(name, 1.0)
        scored_df = df.with_columns(
            (w / (rrf_const + pl.col("rank").cast(pl.Float64))).alias("score")
        ).select(["customer_id", "item_id", "score"])
        scored_list.append(scored_df)

combined_scored = (
    pl.concat(scored_list)
    .group_by(["customer_id", "item_id"])
    .agg(pl.col("score").sum())
)

# Whitelist location assortment filter (stock check penalty)
combined_scored = (
    combined_scored.join(user_loc, on="customer_id", how="left")
    .join(location_assortment, on=["location", "item_id"], how="left")
    .with_columns(
        pl.when(pl.col("location").is_not_null() & pl.col("is_stocked").is_null())
        .then(pl.col("score") * 0.01)
        .otherwise(pl.col("score"))
        .alias("score")
    )
    .drop(["location", "is_stocked"])
)

# Map segments to user budgets
budget_df = pl.DataFrame({
    "retr_user_segment": list(segment_budgets.keys()),
    "segment_budget": list(segment_budgets.values())
}).with_columns([
    pl.col("retr_user_segment").cast(pl.Int8),
    pl.col("segment_budget").cast(pl.Int32)
])
user_budgets = user_segs.join(budget_df, on="retr_user_segment", how="left").select(["customer_id", "segment_budget"])

# Final Rank Selection
combined_cands = (
    combined_scored.sort(["customer_id", "score"], descending=[False, True])
    .join(user_budgets, on="customer_id", how="left")
    .with_columns(
        pl.int_range(1, pl.len() + 1).over("customer_id").alias("combined_rank")
    )
    .filter(pl.col("combined_rank") <= pl.col("segment_budget"))
    .select(["customer_id", "item_id"])
)

# Combined Recall & Footprint
c_hits = combined_cands.join(truth, on=["customer_id", "item_id"], how="inner").height
c_recall = c_hits / truth.height
c_size = combined_cands.height / target_users.height

print("\n=======================================================")
print(f"  FINAL COMBINED RETRIEVER RECALL : {c_recall:.4%} ({c_hits}/{truth.height} hits)")
print(f"  Average Candidates per User     : {c_size:.2f}")
print("=======================================================")

