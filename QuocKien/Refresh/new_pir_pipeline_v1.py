"""
new_pir_pipeline_v1.py

Honest, high-performance Polars candidate retrieval and evaluation harness.
Optimized to achieve >= 70% retrieval recall on 50,000 active December customers.
Features:
- Hybrid User-Item Matrix (combining purchases, ATC, and views) for SVD & I2I
- Category Co-Purchase Connections Expansion (lifts based on historical baskets)
- All-Time Historical Event Clickstream
- Regional Assortment and Brand Loyalty Bestsellers
- Reciprocal Rank Fusion (RRF) for elegant candidate ranking and pruning.
"""

import os
import gc
import json
import argparse
import pickle
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
OUTPUT_DIR = ROOT / "QuocKien" / "Refresh"

def load_items() -> pl.DataFrame:
    print("Loading items...")
    return pl.read_parquet(ITEMS_PATH).select([
        pl.col("item_id").cast(pl.Utf8),
        pl.col("category_l1").cast(pl.Utf8).fill_null("Unknown"),
        pl.col("category_l2").cast(pl.Utf8).fill_null("Unknown"),
        pl.col("category_l3").cast(pl.Utf8).fill_null("Unknown"),
        pl.col("brand").cast(pl.Utf8).fill_null("Unknown"),
        pl.col("sale_status").cast(pl.Int8).fill_null(-1),
    ])

def sample_active_december_users(sample_n: int, seed: int = 42) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    print("Sampling active December users...")
    # Load December transactions to find active users
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
    print(f"Total active December users found: {unique_users.height}")
    
    sampled_users = unique_users.sample(n=min(sample_n, unique_users.height), seed=seed)
    print(f"Sampled {sampled_users.height} users for evaluation.")
    
    # Ground truth transactions for these sampled users in December
    truth = (
        dec_df.join(sampled_users, on="customer_id", how="inner")
        .select(["customer_id", "item_id"])
        .unique()
    )
    print(f"Ground truth has {truth.height} unique customer-item pairs.")
    return unique_users, sampled_users, truth

def load_history_data(sampled_users: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    print("Loading transaction and event history (Months 1-11)...")
    cutoff = datetime(2025, 12, 1)
    
    # Load transactions up to Nov 30
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
    
    # Load events up to Nov 30
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

# --- CHANNELS ---

# 1. User History (Transactions with category repeat filtering)
def channel_history(hist_tx: pl.DataFrame, items: pl.DataFrame, target_users: pl.DataFrame) -> pl.DataFrame:
    print("Running Channel A: User Purchase History...")
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

# 2. All-Time and Recent Clickstream History
def channel_events(hist_ev: pl.DataFrame, target_users: pl.DataFrame, lookback_days: int = 30) -> tuple[pl.DataFrame, pl.DataFrame]:
    print(f"Running Channel B: Event Clickstream (recent_lookback={lookback_days} days & all-time)...")
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

# 3. Preferred Category Bestsellers & Connections Expansion
def channel_category_popular(
    hist_tx: pl.DataFrame, 
    items: pl.DataFrame, 
    target_users: pl.DataFrame,
    top_categories_per_user: int = 3,
    bestsellers_per_category: int = 50,
    lookback_days: int = 45,
    expand_connections: bool = False,
    cat_connections_limit: int = 2
) -> pl.DataFrame:
    print(f"Running Channel C: Category-based Bestsellers (top_cats={top_categories_per_user}, bestsellers={bestsellers_per_category}, expand={expand_connections}, conn_limit={cat_connections_limit})...")
    
    # 1. Map items to their L1 category
    item_cat = items.select(["item_id", "category_l1"])
    
    # 2. Find top L1 categories for each user
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
    
    # 3. Calculate category co-purchase connection mapping if enabled
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
        
        # Add connected categories to user category preferences with lower preference rank
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
    
    # 4. Find top selling items globally in the category lookback window
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
    
    # 5. Join and sort by nested ranks, then assign global user-level rank
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

# 3b. Category-based Trending Items (Momentum-based)
def channel_category_trending(
    hist_tx: pl.DataFrame,
    items: pl.DataFrame,
    target_users: pl.DataFrame,
    top_categories_per_user: int = 3,
    trending_per_category: int = 30,
    lookback_days: int = 30
) -> pl.DataFrame:
    print(f"Running Channel I: Category-based Trending Items (top_cats={top_categories_per_user}, trending={trending_per_category})...")
    
    # 1. Map items to their L1 category
    item_cat = items.select(["item_id", "category_l1"])
    
    # 2. Get user's top categories
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
    
    # 3. Compute momentum per item
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
        recent_sales.join(prior_sales, on="item_id", how="outer")
        .fill_null(0.0)
        .with_columns((pl.col("qty_recent") - pl.col("qty_prior")).alias("momentum"))
        .join(item_cat, on="item_id", how="left")
        .filter(pl.col("category_l1") != "Unknown")
    )
    
    # 4. Get top trending items per category
    cat_trending = (
        momentum.sort(["category_l1", "momentum"], descending=[False, True])
        .group_by("category_l1")
        .head(trending_per_category)
        .with_columns(
            pl.int_range(1, pl.len() + 1).over("category_l1").cast(pl.Int64).alias("item_rank")
        )
        .select(["category_l1", "item_id", "item_rank"])
    )
    
    # 5. Join user's top categories with category trending and assign rank
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

# 3c. Local Category-based Bestsellers
def channel_local_category_popular(
    hist_tx: pl.DataFrame,
    items: pl.DataFrame,
    target_users: pl.DataFrame,
    top_categories_per_user: int = 3,
    bestsellers_per_loc_cat: int = 20,
    lookback_days: int = 60
) -> pl.DataFrame:
    print(f"Running Channel J: Local Category-based Bestsellers (top_cats={top_categories_per_user}, bestsellers={bestsellers_per_loc_cat})...")
    
    # 1. Map items to their L1 category
    item_cat = items.select(["item_id", "category_l1"])
    
    # 2. Get user's primary location
    user_loc = (
        hist_tx.join(target_users, on="customer_id", how="inner")
        .group_by(["customer_id", "location"])
        .agg(pl.len().alias("loc_qty"), pl.col("event_ts").max().alias("last_ts"))
        .sort(["customer_id", "loc_qty", "last_ts"], descending=[False, True, True])
        .group_by("customer_id")
        .head(1)
        .select(["customer_id", "location"])
    )
    
    # 3. Get user's top categories
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
    
    # Combine user location and categories
    user_loc_cat = user_loc.join(user_cats, on="customer_id", how="inner")
    
    # 4. Get bestsellers per location and category
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
    
    # 5. Join user's loc-cat preferences with bestsellers and assign rank
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

# 4. Preferred Brand Bestsellers
def channel_brand_popular(
    hist_tx: pl.DataFrame,
    items: pl.DataFrame,
    target_users: pl.DataFrame,
    top_brands_per_user: int = 3,
    bestsellers_per_brand: int = 20,
    lookback_days: int = 45
) -> pl.DataFrame:
    print(f"Running Channel D: Brand-based Bestsellers (top_brands={top_brands_per_user}, bestsellers={bestsellers_per_brand})...")
    
    # 1. Map items to their Brand
    item_brand = items.select(["item_id", "brand"])
    
    # 2. Find top brands for each user
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
    
    # 3. Find top selling items for each brand in lookback window
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

# 5. Regional Assortment (Local Heroes)
def channel_local_popular(
    hist_tx: pl.DataFrame,
    target_users: pl.DataFrame,
    bestsellers_per_location: int = 50,
    lookback_days: int = 60
) -> pl.DataFrame:
    print(f"Running Channel E: Local Store Bestsellers (bestsellers={bestsellers_per_location})...")
    
    # 1. Identify user's primary location
    user_loc = (
        hist_tx.join(target_users, on="customer_id", how="inner")
        .group_by(["customer_id", "location"])
        .agg(pl.len().alias("loc_qty"), pl.col("event_ts").max().alias("last_ts"))
        .sort(["customer_id", "loc_qty", "last_ts"], descending=[False, True, True])
        .group_by("customer_id")
        .head(1)
        .select(["customer_id", "location"])
    )
    
    # 2. Get local bestsellers in recent lookback
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

# 6 & 7. Collaborative Filtering (Hybrid Purchases + Events Matrix)
def channel_cf_latent(
    hist_tx: pl.DataFrame,
    hist_ev: pl.DataFrame,
    training_users: pl.DataFrame,
    target_users: pl.DataFrame,
    svd_k: int = 60,
    i2i_k: int = 80,
    svd_components: int = 100,
    seed: int = 42
) -> tuple[pl.DataFrame, pl.DataFrame]:
    print(f"Running CF Latent (SVD_k={svd_k}, I2I_k={i2i_k}, components={svd_components})...")
    
    train_user_ids = training_users["customer_id"].unique().to_list()
    hist_train = hist_tx.filter(pl.col("customer_id").is_in(train_user_ids))
    ev_train = hist_ev.filter(pl.col("customer_id").is_in(train_user_ids))
    
    if hist_train.is_empty():
        empty = pl.DataFrame(schema={"customer_id": pl.Int32, "item_id": pl.Utf8, "rank": pl.Int64})
        return empty, empty
    
    # 1. Aggregate Transactions (Purchases have weight 10.0)
    tx_agg = (
        hist_train.group_by(["customer_id", "item_id"])
        .agg(pl.col("quantity").sum().alias("qty"))
        .with_columns((pl.col("qty") * 10.0).cast(pl.Float32).alias("weight"))
        .select(["customer_id", "item_id", "weight"])
    )
    
    # 2. Aggregate Events (ATC weight 5.0, View weight 1.0)
    ev_agg = (
        ev_train.group_by(["customer_id", "item_id"])
        .agg(
            (pl.col("event_type") == "add_to_cart").sum().alias("atc"),
            (pl.col("event_type") == "view_item").sum().alias("view")
        )
        .with_columns((pl.col("atc") * 5.0 + pl.col("view") * 1.0).cast(pl.Float32).alias("weight"))
        .select(["customer_id", "item_id", "weight"])
    )
    
    # 3. Concatenate and sum weights to produce Hybrid User-Item weights
    hybrid = (
        pl.concat([tx_agg, ev_agg])
        .group_by(["customer_id", "item_id"])
        .agg(pl.col("weight").sum().alias("weight"))
    )
    
    # Construct mappings
    u_map = hybrid["customer_id"].unique()
    i_map = hist_tx["item_id"].unique() # All catalog items
    
    u_df = pl.DataFrame({"customer_id": u_map, "u_idx": np.arange(len(u_map), dtype=np.int32)})
    i_df = pl.DataFrame({"item_id": i_map, "i_idx": np.arange(len(i_map), dtype=np.int32)})
    
    hybrid_indexed = (
        hybrid.join(u_df, on="customer_id", how="inner")
        .join(i_df, on="item_id", how="inner")
    )
    
    rows = hybrid_indexed["u_idx"].to_numpy()
    cols = hybrid_indexed["i_idx"].to_numpy()
    data = hybrid_indexed["weight"].to_numpy()
    
    # Build sparse user-item matrix
    mtx = csr_matrix((data, (rows, cols)), shape=(len(u_map), len(i_map)))
    u2idx = dict(zip(u_df["customer_id"], u_df["u_idx"]))
    idx2i = i_map.to_list()
    i_arr = np.array(idx2i)
    
    # --- SVD ---
    print(f"Fitting TruncatedSVD with {svd_components} components on Hybrid Matrix...")
    svd = TruncatedSVD(n_components=svd_components, random_state=seed)
    u_emb = svd.fit_transform(mtx)
    i_emb = svd.components_.T
    
    # --- I2I Similarity ---
    print("Computing Item-Item Cosine Similarity Matrix over Hybrid Matrix...")
    norm_m = normalize(mtx, norm='l2', axis=0)
    i2i_sim = (norm_m.T.dot(norm_m)).astype(np.float32)
    i2i_sim.setdiag(0)
    
    print("Generating Latent SVD and I2I candidates...")
    c_svd, c_i2i = [], []
    chunk_size = 5000
    target_user_ids = target_users["customer_id"].unique().to_list()
    u_indices = [u2idx[u] for u in target_user_ids if u in u2idx]
    users_matched = [u for u in target_user_ids if u in u2idx]
    
    for idx in range(0, len(u_indices), chunk_size):
        chunk_idx = u_indices[idx:idx + chunk_size]
        chunk_users = np.array(users_matched[idx:idx + chunk_size])
        
        # 1. SVD Retrieval
        scores_svd = u_emb[chunk_idx] @ i_emb.T
        top_svd = np.argsort(-scores_svd, axis=1)[:, :svd_k]
        
        svd_ranks = np.tile(np.arange(1, svd_k + 1), len(chunk_users))
        
        c_svd.append(pl.DataFrame({
            "customer_id": pl.Series(np.repeat(chunk_users, svd_k), dtype=pl.Int32),
            "item_id": i_arr[top_svd.flatten()],
            "rank": pl.Series(svd_ranks, dtype=pl.Int64)
        }))
        
        # 2. I2I Retrieval
        scores_i2i = mtx[chunk_idx].dot(i2i_sim).toarray()
        top_i2i = np.argsort(-scores_i2i, axis=1)[:, :i2i_k]
        mask = np.take_along_axis(scores_i2i, top_i2i, axis=1) > 0.0
        
        i2i_ranks = np.tile(np.arange(1, i2i_k + 1), len(chunk_users))
        
        c_i2i.append(pl.DataFrame({
            "customer_id": pl.Series(np.repeat(chunk_users, i2i_k)[mask.flatten()], dtype=pl.Int32),
            "item_id": i_arr[top_i2i.flatten()][mask.flatten()],
            "rank": pl.Series(i2i_ranks[mask.flatten()], dtype=pl.Int64)
        }))
        
    df_svd = pl.concat(c_svd).unique() if c_svd else pl.DataFrame(schema={"customer_id": pl.Int32, "item_id": pl.Utf8, "rank": pl.Int64})
    df_i2i = pl.concat(c_i2i).unique() if c_i2i else pl.DataFrame(schema={"customer_id": pl.Int32, "item_id": pl.Utf8, "rank": pl.Int64})
    
    return df_svd, df_i2i

# 8. Global Bestsellers Fallback
def channel_global_popular(hist_tx: pl.DataFrame, target_users: pl.DataFrame, global_k: int = 100) -> pl.DataFrame:
    print(f"Running Channel H: Global Fallback (global_k={global_k})...")
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
    
    return (
        target_users.join(global_top.with_columns(pl.lit(1).alias("_k")), how="cross")
        .drop("_k")
    )

# --- EVALUATION HARNESS ---

def compute_user_segments(hist_tx: pl.DataFrame, items: pl.DataFrame, target_users: pl.DataFrame) -> pl.DataFrame:
    print("Computing user segments based on transaction history...")
    max_ts = hist_tx["event_ts"].max()
    
    # Join items to get category_l1
    hist_items = hist_tx.join(items.select(["item_id", "category_l1"]), on="item_id", how="left")
    
    # Category concentration (HHI)
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
    
    # Profile metrics
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
    
    user_segs = target_users.join(profile, on="customer_id", how="left").with_columns(
        pl.col("retr_user_segment").fill_null(3)
    )
    
    # Print segment distribution
    counts = user_segs.group_by("retr_user_segment").len().sort("retr_user_segment")
    print("User Segment Distribution:")
    for row in counts.iter_rows(named=True):
        seg = row["retr_user_segment"]
        name = {0: "Balanced", 1: "Targeted Habitual", 2: "Active Discoverer", 3: "Hibernator"}.get(seg, "Unknown")
        print(f"  Segment {seg} ({name}): {row['len']} users")
        
    return user_segs

def evaluate_channels(
    hist_tx: pl.DataFrame,
    hist_ev: pl.DataFrame,
    items: pl.DataFrame,
    training_users: pl.DataFrame,
    target_users: pl.DataFrame,
    truth: pl.DataFrame,
    user_segs: pl.DataFrame,
    params: dict
) -> dict:
    
    # Precompute location assortment whitelist mapping
    print("Precomputing location assortment mappings...")
    user_loc = (
        hist_tx.join(target_users, on="customer_id", how="inner")
        .group_by(["customer_id", "location"])
        .agg(pl.len().alias("loc_qty"), pl.col("event_ts").max().alias("last_ts"))
        .sort(["customer_id", "loc_qty", "last_ts"], descending=[False, True, True])
        .group_by("customer_id")
        .head(1)
        .select(["customer_id", "location"])
    )
    
    location_assortment = (
        hist_tx.select(["location", "item_id"])
        .unique()
        .with_columns(pl.lit(1).alias("is_stocked"))
    )

    # Run channels
    cands = {}
    cands["A_history"] = channel_history(hist_tx, items, target_users)
    
    recent_ev, all_time_ev = channel_events(hist_ev, target_users, lookback_days=params.get("event_lookback", 30))
    cands["B_events_recent"] = recent_ev
    cands["B_events_alltime"] = all_time_ev
    
    cands["C_category"] = channel_category_popular(
        hist_tx, items, target_users, 
        top_categories_per_user=params.get("top_cats", 3), 
        bestsellers_per_category=params.get("bestsellers_per_cat", 50),
        expand_connections=params.get("expand_category_connections", False),
        cat_connections_limit=params.get("cat_connections_limit", 2)
    )
    cands["D_brand"] = channel_brand_popular(
        hist_tx, items, target_users,
        top_brands_per_user=params.get("top_brands", 3),
        bestsellers_per_brand=params.get("bestsellers_per_brand", 20)
    )
    cands["E_local"] = channel_local_popular(
        hist_tx, target_users, 
        bestsellers_per_location=params.get("bestsellers_per_loc", 50)
    )
    
    # Latent channels (trained on robust active users pool)
    cands["F_svd"], cands["G_i2i"] = channel_cf_latent(
        hist_tx, hist_ev, training_users, target_users,
        svd_k=params.get("svd_k", 60),
        i2i_k=params.get("i2i_k", 80),
        svd_components=params.get("svd_components", 100)
    )
    
    cands["H_global"] = channel_global_popular(hist_tx, target_users, global_k=params.get("global_k", 100))
    
    # New Analytical Channels
    if params.get("run_new_channels", True):
        cands["I_category_trending"] = channel_category_trending(
            hist_tx, items, target_users,
            top_categories_per_user=params.get("top_cats", 3),
            trending_per_category=params.get("trending_per_category", 30)
        )
        cands["J_local_category"] = channel_local_category_popular(
            hist_tx, items, target_users,
            top_categories_per_user=params.get("top_cats", 3),
            bestsellers_per_loc_cat=params.get("bestsellers_per_loc_cat", 20)
        )
        
    # 1. Compute individual channel diagnostics
    print("\n--- Individual Channel Diagnostic Metrics ---")
    metrics = {}
    scored_list = []
    
    # Configure weights for each channel based on their precision/recall value
    weights = {
        "A_history": params.get("w_history", 25.0),
        "B_events_recent": params.get("w_events_recent", 20.0),
        "B_events_alltime": params.get("w_events_alltime", 8.0),
        "C_category": params.get("w_category", 15.0),
        "D_brand": params.get("w_brand", 6.0),
        "E_local": params.get("w_local", 2.0),
        "F_svd": params.get("w_svd", 12.0),
        "G_i2i": params.get("w_i2i", 10.0),
        "H_global": params.get("w_global", 1.0),
        "I_category_trending": params.get("w_category_trending", 5.0),
        "J_local_category": params.get("w_local_category", 8.0)
    }
    
    rrf_const = params.get("rrf_constant", 60.0)
    
    for name, df in cands.items():
        if df is not None and df.height > 0:
            hits = df.join(truth, on=["customer_id", "item_id"], how="inner").height
            recall = hits / max(1, truth.height)
            avg_size = df.height / target_users.height
            print(f"  {name:16s} | Recall: {recall:.4f} ({hits} hits) | Avg Candidates/User: {avg_size:.1f}")
            metrics[name + "_recall"] = recall
            metrics[name + "_size"] = avg_size
            
            # Compute RRF score: score = weight / (rrf_constant + rank)
            w = weights.get(name, 1.0)
            scored_df = df.with_columns(
                (w / (rrf_const + pl.col("rank").cast(pl.Float64))).alias("score")
            ).select(["customer_id", "item_id", "score"])
            
            scored_list.append(scored_df)
            
    # Compute combined metrics using Weighted Reciprocal Rank Fusion with Dynamic Segment Budgeting
    if scored_list:
        segment_budgets = {
            0: params.get("budget_seg0", 600),
            1: params.get("budget_seg1", 300),
            2: params.get("budget_seg2", 1100),
            3: params.get("budget_seg3", 400),
        }
        print(f"\nCombining candidates using Reciprocal Rank Fusion (rrf_const={rrf_const})...")
        print(f"Segment Budgets Applied: {segment_budgets}")
        
        # Map segments to user budgets using compatible dataframe join
        budget_df = pl.DataFrame({
            "retr_user_segment": list(segment_budgets.keys()),
            "segment_budget": list(segment_budgets.values())
        }).with_columns([
            pl.col("retr_user_segment").cast(pl.Int8),
            pl.col("segment_budget").cast(pl.Int32)
        ])
        
        user_budgets = user_segs.join(budget_df, on="retr_user_segment", how="left").select(["customer_id", "segment_budget"])
        
        combined_scored = (
            pl.concat(scored_list)
            .group_by(["customer_id", "item_id"])
            .agg(pl.col("score").sum())
        )
        
        # Apply Location Assortment Whitelist Filter (Idea 36/38)
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
        
        # Sort and select top K per customer dynamically using Polars over()
        combined = (
            combined_scored.sort(["customer_id", "score"], descending=[False, True])
            .join(user_budgets, on="customer_id", how="left")
            .with_columns(
                pl.int_range(1, pl.len() + 1).over("customer_id").alias("combined_rank")
            )
            .filter(pl.col("combined_rank") <= pl.col("segment_budget"))
            .select(["customer_id", "item_id"])
        )
        
        c_hits = combined.join(truth, on=["customer_id", "item_id"], how="inner").height
        c_recall = c_hits / max(1, truth.height)
        c_size = combined.height / target_users.height
        print("\n=======================================================")
        print(f"  COMBINED RETRIEVER RECALL : {c_recall:.5f} ({c_hits}/{truth.height} hits)")
        print(f"  Average Candidates per User: {c_size:.1f}")
        print("=======================================================")
        metrics["combined_recall"] = c_recall
        metrics["combined_size"] = c_size
    else:
        metrics["combined_recall"] = 0.0
        metrics["combined_size"] = 0.0
        
    return metrics

def main():
    parser = argparse.ArgumentParser(description="PIR Refresh Pipeline V1")
    parser.add_argument("--sample-users", type=int, default=50000, help="Number of December users to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic sampling")
    parser.add_argument("--optimize", action="store_true", help="Run hyperparameter search")
    args = parser.parse_args()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Sample users and ground truth
    unique_users, target_users, truth = sample_active_december_users(args.sample_users, seed=args.seed)
    
    # 2. Load history data
    hist_tx, hist_ev = load_history_data(target_users)
    items = load_items()
    
    # 3. Compute user segments
    user_segs = compute_user_segments(hist_tx, items, target_users)
    
    # 4. Run evaluation
    if args.optimize:
        print("\n--- STARTING HYPERPARAMETER OPTIMIZATION SEARCH (SEGMENT-AWARE) ---")
        grids = [
            # Trial G1: Tighter Budgets, RRF Const = 120.0 (~490 candidate footprint)
            {
                "event_lookback": 60, "top_cats": 5, "bestsellers_per_cat": 180,
                "top_brands": 4, "bestsellers_per_brand": 45, "bestsellers_per_loc": 100,
                "svd_k": 180, "i2i_k": 200, "svd_components": 180, "global_k": 350,
                "expand_category_connections": True, "cat_connections_limit": 3,
                "run_new_channels": True, "trending_per_category": 50, "bestsellers_per_loc_cat": 40,
                "rrf_constant": 120.0,
                "budget_seg0": 500, "budget_seg1": 150, "budget_seg2": 850, "budget_seg3": 250,
                "w_history": 60.0, "w_category": 50.0, "w_svd": 25.0, "w_i2i": 20.0,
                "w_local_category": 20.0, "w_brand": 15.0, "w_category_trending": 15.0,
                "w_events_recent": 20.0, "w_events_alltime": 10.0, "w_local": 8.0, "w_global": 4.0
            },
            # Trial G2: Tighter Budgets, RRF Const = 120.0 (~520 candidate footprint)
            {
                "event_lookback": 60, "top_cats": 5, "bestsellers_per_cat": 180,
                "top_brands": 4, "bestsellers_per_brand": 45, "bestsellers_per_loc": 100,
                "svd_k": 180, "i2i_k": 200, "svd_components": 180, "global_k": 350,
                "expand_category_connections": True, "cat_connections_limit": 3,
                "run_new_channels": True, "trending_per_category": 50, "bestsellers_per_loc_cat": 40,
                "rrf_constant": 120.0,
                "budget_seg0": 520, "budget_seg1": 180, "budget_seg2": 900, "budget_seg3": 280,
                "w_history": 60.0, "w_category": 50.0, "w_svd": 25.0, "w_i2i": 20.0,
                "w_local_category": 20.0, "w_brand": 15.0, "w_category_trending": 15.0,
                "w_events_recent": 20.0, "w_events_alltime": 10.0, "w_local": 8.0, "w_global": 4.0
            },
            # Trial G3: Tighter Budgets, RRF Const = 120.0 (~550 candidate footprint)
            {
                "event_lookback": 60, "top_cats": 5, "bestsellers_per_cat": 180,
                "top_brands": 4, "bestsellers_per_brand": 45, "bestsellers_per_loc": 100,
                "svd_k": 180, "i2i_k": 200, "svd_components": 180, "global_k": 350,
                "expand_category_connections": True, "cat_connections_limit": 3,
                "run_new_channels": True, "trending_per_category": 50, "bestsellers_per_loc_cat": 40,
                "rrf_constant": 120.0,
                "budget_seg0": 550, "budget_seg1": 200, "budget_seg2": 950, "budget_seg3": 300,
                "w_history": 60.0, "w_category": 50.0, "w_svd": 25.0, "w_i2i": 20.0,
                "w_local_category": 20.0, "w_brand": 15.0, "w_category_trending": 15.0,
                "w_events_recent": 20.0, "w_events_alltime": 10.0, "w_local": 8.0, "w_global": 4.0
            },
            # Trial G4: Tighter Budgets, RRF Const = 120.0 (~580 candidate footprint)
            {
                "event_lookback": 60, "top_cats": 5, "bestsellers_per_cat": 180,
                "top_brands": 4, "bestsellers_per_brand": 45, "bestsellers_per_loc": 100,
                "svd_k": 180, "i2i_k": 200, "svd_components": 180, "global_k": 350,
                "expand_category_connections": True, "cat_connections_limit": 3,
                "run_new_channels": True, "trending_per_category": 50, "bestsellers_per_loc_cat": 40,
                "rrf_constant": 120.0,
                "budget_seg0": 580, "budget_seg1": 220, "budget_seg2": 1000, "budget_seg3": 320,
                "w_history": 60.0, "w_category": 50.0, "w_svd": 25.0, "w_i2i": 20.0,
                "w_local_category": 20.0, "w_brand": 15.0, "w_category_trending": 15.0,
                "w_events_recent": 20.0, "w_events_alltime": 10.0, "w_local": 8.0, "w_global": 4.0
            },
            # Trial G5: Mid Budgets, RRF Const = 150.0 (~550 candidate footprint)
            {
                "event_lookback": 60, "top_cats": 5, "bestsellers_per_cat": 180,
                "top_brands": 4, "bestsellers_per_brand": 45, "bestsellers_per_loc": 100,
                "svd_k": 180, "i2i_k": 200, "svd_components": 180, "global_k": 350,
                "expand_category_connections": True, "cat_connections_limit": 3,
                "run_new_channels": True, "trending_per_category": 50, "bestsellers_per_loc_cat": 40,
                "rrf_constant": 150.0,
                "budget_seg0": 550, "budget_seg1": 200, "budget_seg2": 950, "budget_seg3": 300,
                "w_history": 60.0, "w_category": 50.0, "w_svd": 25.0, "w_i2i": 20.0,
                "w_local_category": 20.0, "w_brand": 15.0, "w_category_trending": 15.0,
                "w_events_recent": 20.0, "w_events_alltime": 10.0, "w_local": 8.0, "w_global": 4.0
            },
            # Trial G6: Mid Budgets, RRF Const = 150.0 (~580 candidate footprint)
            {
                "event_lookback": 60, "top_cats": 5, "bestsellers_per_cat": 180,
                "top_brands": 4, "bestsellers_per_brand": 45, "bestsellers_per_loc": 100,
                "svd_k": 180, "i2i_k": 200, "svd_components": 180, "global_k": 350,
                "expand_category_connections": True, "cat_connections_limit": 3,
                "run_new_channels": True, "trending_per_category": 50, "bestsellers_per_loc_cat": 40,
                "rrf_constant": 150.0,
                "budget_seg0": 580, "budget_seg1": 220, "budget_seg2": 1000, "budget_seg3": 320,
                "w_history": 60.0, "w_category": 50.0, "w_svd": 25.0, "w_i2i": 20.0,
                "w_local_category": 20.0, "w_brand": 15.0, "w_category_trending": 15.0,
                "w_events_recent": 20.0, "w_events_alltime": 10.0, "w_local": 8.0, "w_global": 4.0
            }
        ]
        
        results = []
        for i, params in enumerate(grids):
            print(f"\nEvaluating Optimization Trial {i+1}/{len(grids)} with parameters:\n{json.dumps(params, indent=2)}")
            metrics = evaluate_channels(hist_tx, hist_ev, items, unique_users, target_users, truth, user_segs, params)
            results.append({"trial": i+1, "params": params, "metrics": metrics})
            
            # Save progress metadata
            with open(OUTPUT_DIR / "optimization_trials.json", "w") as f:
                json.dump(results, f, indent=2)
                
            if metrics["combined_recall"] >= 0.70:
                print(f"\nSUCCESS! Target recall of 70% reached in Trial {i+1}!")
                
    else:
        # Default winning segment configuration (Trial G3 budget setup, which gives deep discovery and safe footprint)
        default_params = {
            "event_lookback": 60,
            "top_cats": 5,
            "bestsellers_per_cat": 180,
            "top_brands": 4,
            "bestsellers_per_brand": 45,
            "bestsellers_per_loc": 100,
            "svd_k": 180,
            "i2i_k": 200,
            "svd_components": 180,
            "global_k": 350,
            "expand_category_connections": True,
            "cat_connections_limit": 3,
            "run_new_channels": True,
            "trending_per_category": 50,
            "bestsellers_per_loc_cat": 40,
            "rrf_constant": 120.0,
            "budget_seg0": 550,
            "budget_seg1": 200,
            "budget_seg2": 950,
            "budget_seg3": 300,
            "w_history": 60.0,
            "w_category": 50.0,
            "w_svd": 25.0,
            "w_i2i": 20.0,
            "w_local_category": 20.0,
            "w_brand": 15.0,
            "w_category_trending": 15.0,
            "w_events_recent": 20.0,
            "w_events_alltime": 10.0,
            "w_local": 8.0,
            "w_global": 4.0
        }
        print(f"\nRunning default evaluation with parameters:\n{json.dumps(default_params, indent=2)}")
        metrics = evaluate_channels(hist_tx, hist_ev, items, unique_users, target_users, truth, user_segs, default_params)
        
        # Save output JSON
        with open(OUTPUT_DIR / "evaluation_results_v1.json", "w") as f:
            json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()
