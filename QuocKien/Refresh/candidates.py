import os
import gc
import json
import argparse
import pickle
import time
import psutil
import numpy as np
import polars as pl
import scipy.sparse as sp
from datetime import datetime
from pathlib import Path
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.linalg import svds
import implicit  # Thư viện cho luồng Collaborative Filtering ALS chuyên sâu
import pyarrow as pa
import pyarrow.parquet as pq

# ==============================================================================
# CONFIGURATION & BASE PATHS
# ==============================================================================
ROOT = Path(r"d:\CS116\ProjectNumberOne")
TRANSACTION_PATH = ROOT / "transaction_full_2025.parquet"
EVENT_PATH = ROOT / "event_full_2025.parquet"
ITEMS_PATH = ROOT / "items.parquet"

CACHE_DIR = Path("./recs_cache")
CACHE_DIR.mkdir(exist_ok=True)

# HẰNG SỐ ĐIỀU TỐC KIỂM THỬ TOÀN CỤC (-1 để chạy FILE SCALE 2.8 triệu user)
USER_LIMIT = 5000  

# CÔNG TẮC NGẮT MẠCH CHỐNG OOM TUYỆT ĐỐI
ENABLE_COBUY = True  

def print_status(msg: str):
    """Hàm ghi nhận mốc thời gian và lượng RAM tiêu thụ thực tế tại thời điểm gọi"""
    process = psutil.Process(os.getpid())
    ram_gb = process.memory_info().rss / (1024 ** 3)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [RAM: {ram_gb:.2f} GB] {msg}")

def load_items() -> pl.DataFrame:
    print_status("Loading items metadata...")
    return pl.read_parquet(ITEMS_PATH).select([
        pl.col("item_id").cast(pl.Utf8),
        pl.col("category_l1").cast(pl.Utf8).fill_null("Unknown"),
        pl.col("category_l2").cast(pl.Utf8).fill_null("Unknown"),
        pl.col("category_l3").cast(pl.Utf8).fill_null("Unknown"),
        pl.col("brand").cast(pl.Utf8).fill_null("Unknown"),
        pl.col("price").cast(pl.Float32).fill_null(0.0),
    ])

def load_history_data() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    print_status("Loading full transaction history and extracting target users...")
    tx = (
        pl.scan_parquet(TRANSACTION_PATH)
        .select([
            pl.col("customer_id").cast(pl.Int32),
            pl.col("item_id").cast(pl.Utf8),
            pl.col("quantity").cast(pl.Float32).fill_null(1.0),
            pl.col("location").cast(pl.Int32),
            pl.col("updated_date").cast(pl.Datetime).alias("event_ts"),
            pl.col("bill_id"),
        ])
        .collect()
    )
    max_ts = tx["event_ts"].max()
    cutoff = max_ts - pl.duration(days=7)
    
    hist_tx = tx.filter(pl.col("event_ts") < cutoff)
    valid_tx = tx.filter(pl.col("event_ts") >= cutoff)
    
    target_users = hist_tx.select("customer_id").unique()
    print_status(f"Extracted {target_users.height:,} unique target users from history.")
    return hist_tx, valid_tx, target_users

def load_event_history(target_users: pl.DataFrame, cutoff: datetime) -> pl.DataFrame:
    print_status("Loading event interaction history...")
    return (
        pl.scan_parquet(EVENT_PATH)
        .filter(
            pl.col("event_type").is_in(["view_item", "add_to_cart"]) & 
            pl.col("item_id").is_not_null() & 
            (pl.col("event_date").cast(pl.Datetime) < cutoff)
        )
        .select([
            pl.col("customer_id").cast(pl.Int32),
            pl.col("item_id").cast(pl.Utf8),
            pl.col("event_date").cast(pl.Datetime).alias("event_ts")
        ])
        .join(target_users.lazy(), on="customer_id", how="inner")
        .collect()
    )

# ==============================================================================
# ADVANCED FEATURE EXTRACTION (100% POLARS NATIVE - ZERO PANDAS)
# ==============================================================================
def precompute_advanced_features(hist_tx: pl.DataFrame, items: pl.DataFrame) -> tuple[Path, Path, int]:
    print_status("Precomputing advanced filtering features (User-Item Price Metrics & Category SVD)...")
    
    hist_joined = hist_tx.join(items.select(["item_id", "category_l2"]), on="item_id", how="left")
    user_cat_matrix = (
        hist_joined.group_by(["customer_id", "category_l2"])
        .agg(pl.len().alias("purchases"))
    )
    
    unique_users_svd = user_cat_matrix.select("customer_id").unique()["customer_id"].to_numpy()
    unique_cats_svd = user_cat_matrix.select("category_l2").drop_nulls().unique()["category_l2"].to_numpy()
    
    uc_valid = user_cat_matrix.filter(pl.col("category_l2").is_not_null())
    
    k_svd = 10
    if len(unique_users_svd) > k_svd and len(unique_cats_svd) > k_svd:
        user_map_df = pl.DataFrame({"customer_id": unique_users_svd, "user_idx": np.arange(len(unique_users_svd), dtype=np.int32)})
        cat_map_df = pl.DataFrame({"category_l2": unique_cats_svd, "cat_idx": np.arange(len(unique_cats_svd), dtype=np.int32)})
        
        uc_indexed = uc_valid.join(user_map_df, on="customer_id").join(cat_map_df, on="category_l2")
        row_svd = uc_indexed["user_idx"].to_numpy()
        col_svd = uc_indexed["cat_idx"].to_numpy()
        data_svd = uc_indexed["purchases"].to_numpy()
        
        mat_svd = coo_matrix((data_svd, (row_svd, col_svd)), shape=(len(unique_users_svd), len(unique_cats_svd))).astype(np.float32)
        
        print_status("Executing svds decomposition...")
        U, S, Vt = svds(mat_svd, k=k_svd)
        sort_idx = np.argsort(S)[::-1]
        U = U[:, sort_idx]
        Vt = Vt[sort_idx, :]
        
        U_df = pl.DataFrame(U, schema=[f"u_svd_{i}" for i in range(k_svd)]).with_columns(pl.Series("customer_id", unique_users_svd, dtype=pl.Int32))
        V_df = pl.DataFrame(Vt.T, schema=[f"c_svd_{i}" for i in range(k_svd)]).with_columns(pl.Series("category_l2", unique_cats_svd))
    else:
        k_svd = 0
        U_df = pl.DataFrame({"customer_id": pl.Series(unique_users_svd, dtype=pl.Int32)})
        V_df = pl.DataFrame({"category_l2": pl.Series(unique_cats_svd, dtype=pl.Utf8)})

    hist_with_price = hist_tx.join(items.select(["item_id", "price"]), on="item_id", how="left")

    print_status("Building User Feature Base...")
    user_features = (
        hist_with_price.group_by("customer_id").agg([pl.col("price").mean().alias("u_avg_price")])
        .join(U_df, on="customer_id", how="left")
    )
    user_features = user_features.with_columns([
        pl.col(c).fill_null(0.0) for c in user_features.columns if c != "customer_id"
    ])
    
    print_status("Building Item Feature Base...")
    item_features = (
        hist_with_price.group_by("item_id").agg([
            pl.col("price").mean().alias("item_avg_price"),
            pl.len().alias("i_total_sales")
        ])
        .join(items.select(["item_id", "category_l2"]), on="item_id", how="left")
        .join(V_df, on="category_l2", how="left").drop("category_l2")
    )
    item_features = item_features.with_columns([
        pl.col(c).fill_null(0.0) for c in item_features.columns if c != "item_id"
    ])
    
    user_features_path = CACHE_DIR / "user_features.parquet"
    item_features_path = CACHE_DIR / "item_features.parquet"
    user_features.write_parquet(user_features_path)
    item_features.write_parquet(item_features_path)
    
    del hist_joined, user_cat_matrix, uc_valid, hist_with_price, U_df, V_df, user_features, item_features, user_map_df, cat_map_df, uc_indexed; gc.collect()
    print_status("Advanced features computed and offloaded to Disk.")
    return user_features_path, item_features_path, k_svd

# ==============================================================================
# PROFILE EXTRACTION CHANNELS (TRÍCH XUẤT HỒ SƠ THƯA - KHÔNG EXPLODE)
# ==============================================================================
def compute_user_archetypes_and_profiles(hist_tx: pl.DataFrame, items: pl.DataFrame, target_users: pl.DataFrame) -> tuple[Path, Path, Path, Path]:
    print_status("Computing deep behavioral archetypes and building lightweight user target maps...")
    max_ts = datetime(2025, 12, 1)
    
    item_sub = items.select(["item_id", "category_l1", "brand"])
    hist_items = hist_tx.join(item_sub, on="item_id", how="left")
    
    user_loc = (
        hist_tx.group_by(["customer_id", "location"]).agg(pl.len().alias("loc_qty"))
        .sort(["customer_id", "loc_qty"], descending=[False, True])
        .group_by("customer_id").head(1).select(["customer_id", "location"])
    )
    user_loc_path = CACHE_DIR / "profile_user_loc.parquet"
    user_loc.write_parquet(user_loc_path)
    del user_loc; gc.collect()
    
    user_cats = (
        hist_items.filter(pl.col("category_l1") != "Unknown")
        .group_by(["customer_id", "category_l1"]).agg(pl.col("quantity").sum().alias("cat_qty"))
        .sort(["customer_id", "cat_qty"], descending=[False, True]).group_by("customer_id").head(5)
        .with_columns(pl.int_range(1, pl.len() + 1).over("customer_id").cast(pl.Int64).alias("cat_rank"))
        .select(["customer_id", "category_l1", "cat_rank"])
    )
    user_cats_path = CACHE_DIR / "profile_user_cats.parquet"
    user_cats.write_parquet(user_cats_path)
    del user_cats; gc.collect()
    
    user_brands = (
        hist_items.filter((pl.col("brand") != "Unknown") & (pl.col("brand") != "Không xác định"))
        .group_by(["customer_id", "brand"]).agg(pl.col("quantity").sum().alias("brand_qty"))
        .sort(["customer_id", "brand_qty"], descending=[False, True]).group_by("customer_id").head(5)
        .with_columns(pl.int_range(1, pl.len() + 1).over("customer_id").cast(pl.Int64).alias("brand_rank"))
        .select(["customer_id", "brand", "brand_rank"])
    )
    user_brands_path = CACHE_DIR / "profile_user_brands.parquet"
    user_brands.write_parquet(user_brands_path)
    del user_brands; gc.collect()

    loc_hhi = (
        hist_tx.group_by(["customer_id", "location"]).agg(pl.len().alias("visits"))
        .with_columns((pl.col("visits") / pl.col("visits").sum().over("customer_id")).alias("share"))
        .group_by("customer_id").agg([(pl.col("share") * pl.col("share")).sum().alias("loc_hhi")])
    )
    cat_hhi = (
        hist_items.group_by(["customer_id", "category_l1"]).agg(pl.len().alias("cat_visits"))
        .with_columns((pl.col("cat_visits") / pl.col("cat_visits").sum().over("customer_id")).alias("cat_share"))
        .group_by("customer_id").agg([
            (pl.col("cat_share") * pl.col("cat_share")).sum().alias("cat_hhi"),
            pl.col("category_l1").n_unique().alias("unique_cats")
        ])
    )
    profile = (
        hist_tx.group_by("customer_id")
        .agg([
            pl.len().alias("total_tx"),
            pl.col("item_id").n_unique().alias("unique_items"),
            (pl.lit(max_ts) - pl.col("event_ts").min()).dt.total_days().alias("tenure_days"),
            (pl.lit(max_ts) - pl.col("event_ts").max()).dt.total_days().alias("recency_days"),
        ])
        .join(loc_hhi, on="customer_id", how="left")
        .join(cat_hhi, on="customer_id", how="left")
        .with_columns([
            pl.col("loc_hhi").fill_null(1.0),
            pl.col("cat_hhi").fill_null(1.0),
            pl.col("unique_cats").fill_null(1),
            pl.col("recency_days").fill_null(999.0)
        ])
        .with_columns(
            pl.when(pl.col("recency_days") >= 90).then(pl.lit("Dormant"))
            .when(pl.col("tenure_days") <= 60).then(pl.lit("New"))
            .when((pl.col("cat_hhi") >= 0.7) & (pl.col("total_tx") >= 3)).then(pl.lit("Habitual"))
            .when(pl.col("unique_cats") >= 4).then(pl.lit("Explorer"))
            .otherwise(pl.lit("Standard"))
            .alias("archetype")
        )
    )
    archetypes_df = target_users.join(profile, on="customer_id", how="left").with_columns(pl.col("archetype").fill_null("Dormant"))
    archetypes_path = CACHE_DIR / "user_archetypes.parquet"
    archetypes_df.write_parquet(archetypes_path)
    
    del hist_items, loc_hhi, cat_hhi, profile, archetypes_df, item_sub; gc.collect()
    print_status("Thin profiles built and stored to Disk.")
    return archetypes_path, user_loc_path, user_cats_path, user_brands_path

# ==============================================================================
# COMPACT REFERENCE MAP CHANNELS (KHÔNG TẠO EXPLODE TRÊN TOÀN CỤC)
# ==============================================================================
def channel_history(hist_tx: pl.DataFrame) -> pl.DataFrame:
    print_status("Running Channel A: Purchase History...")
    return (
        hist_tx.group_by(["customer_id", "item_id"])
        .agg([pl.len().alias("purchase_count"), pl.col("event_ts").max().alias("last_ts")])
        .sort(["customer_id", "last_ts", "purchase_count"], descending=[False, True, True])
        .with_columns(pl.int_range(1, pl.len() + 1).over("customer_id").cast(pl.Int64).alias("rank"))
        .select(["customer_id", "item_id", "rank"])
    )

def channel_local_popular_map(hist_tx: pl.DataFrame, top_k: int = 500) -> pl.DataFrame:
    print_status("Building Reference Map B: Local Store Bestsellers...")
    recent_tx = hist_tx.filter(pl.col("event_ts") >= hist_tx["event_ts"].max() - pl.duration(days=60))
    return (
        recent_tx.group_by(["location", "item_id"]).agg(pl.len().alias("qty"))
        .sort(["location", "qty"], descending=[False, True]).group_by("location").head(top_k)
        .with_columns(pl.int_range(1, pl.len() + 1).over("location").cast(pl.Int64).alias("rank"))
        .select(["location", "item_id", "rank"])
    )

def train_cf_latent(hist_tx: pl.DataFrame, svd_components: int = 100) -> tuple:
    print_status("Training Global Latent CF Matrices Natively via SciPy (Zero-Sklearn/Zero-Pandas)...")
    max_ts = hist_tx["event_ts"].max()
    tx_weighted = (
        hist_tx
        .with_columns(((pl.lit(max_ts) - pl.col("event_ts")).dt.total_days() / 30.0).alias("months_ago"))
        .with_columns((pl.col("quantity") * pl.lit(0.70).pow(pl.col("months_ago"))).cast(pl.Float32).alias("weight"))
        .group_by(["customer_id", "item_id"]).agg(pl.col("weight").sum().alias("weight"))
    )
    
    u_map, i_map = tx_weighted["customer_id"].unique(), hist_tx["item_id"].unique()
    u_df = pl.DataFrame({"customer_id": u_map, "u_idx": np.arange(len(u_map), dtype=np.int32)})
    i_df = pl.DataFrame({"item_id": i_map, "i_idx": np.arange(len(i_map), dtype=np.int32)})
    hybrid_indexed = tx_weighted.join(u_df, on="customer_id", how="inner").join(i_df, on="item_id", how="inner")
    
    rows, cols, data = hybrid_indexed["u_idx"].to_numpy(), hybrid_indexed["i_idx"].to_numpy(), hybrid_indexed["weight"].to_numpy()
    mtx = csr_matrix((data, (rows, cols)), shape=(len(u_map), len(i_map))).astype(np.float32)
    u2idx = dict(zip(u_df["customer_id"], u_df["u_idx"]))
    idx2i = i_map.to_list()
    i_arr = np.array(idx2i)
    
    print_status("Executing SciPy svds low-rank decomposition...")
    U, S, Vt = svds(mtx, k=svd_components)
    sort_idx = np.argsort(S)[::-1]
    U = U[:, sort_idx]
    S = S[sort_idx]
    Vt = Vt[sort_idx, :]
    
    u_emb = (U * S).astype(np.float32)
    i_emb = Vt.T.astype(np.float32)
    
    print_status("Executing Vectorized Sparse Matrix L2-Column Normalization...")
    col_norms = np.sqrt(np.array(mtx.power(2).sum(axis=0))).flatten()
    col_norms[col_norms == 0] = 1.0
    norm_m = mtx @ sp.diags(1.0 / col_norms)
    
    i2i_sim = (norm_m.T.dot(norm_m)).astype(np.float32)
    i2i_sim.setdiag(0)
    
    i2i_sim = i2i_sim.tocsr()
    i2i_sim.data[i2i_sim.data < 0.05] = 0.0
    i2i_sim.eliminate_zeros()
    
    del tx_weighted, u_df, i_df, hybrid_indexed, norm_m, col_norms; gc.collect()
    return u_emb, i_emb, i2i_sim, mtx, u2idx, i_arr

def channel_category_popular_map(hist_tx: pl.DataFrame, items: pl.DataFrame, bestsellers_per_category: int = 80) -> pl.DataFrame:
    print_status("Building Reference Map E: Category-based Bestsellers...")
    item_cat = items.select(["item_id", "category_l1"])
    recent_tx = hist_tx.filter(pl.col("event_ts") >= hist_tx["event_ts"].max() - pl.duration(days=45))
    return (
        recent_tx.join(item_cat, on="item_id", how="left").filter(pl.col("category_l1") != "Unknown")
        .group_by(["category_l1", "item_id"]).agg(pl.len().alias("qty"))
        .sort(["category_l1", "qty"], descending=[False, True]).group_by("category_l1").head(bestsellers_per_category)
        .with_columns(pl.int_range(1, pl.len() + 1).over("category_l1").cast(pl.Int64).alias("item_rank"))
        .select(["category_l1", "item_id", "item_rank"])
    )

def channel_brand_popular_map(hist_tx: pl.DataFrame, items: pl.DataFrame, bestsellers_per_brand: int = 50) -> pl.DataFrame:
    print_status("Building Reference Map F: Brand-based Bestsellers...")
    item_brand = items.select(["item_id", "brand"])
    recent_tx = hist_tx.filter(pl.col("event_ts") >= hist_tx["event_ts"].max() - pl.duration(days=60))
    return (
        recent_tx.join(item_brand, on="item_id", how="left").filter((pl.col("brand") != "Unknown") & (pl.col("brand") != "Không xác định"))
        .group_by(["brand", "item_id"]).agg(pl.len().alias("qty"))
        .sort(["brand", "qty"], descending=[False, True]).group_by("brand").head(bestsellers_per_brand)
        .with_columns(pl.int_range(1, pl.len() + 1).over("brand").cast(pl.Int64).alias("item_rank"))
        .select(["brand", "item_id", "item_rank"])
    )

def channel_global_popular_map(hist_tx: pl.DataFrame, global_k: int = 300) -> pl.DataFrame:
    print_status("Building Reference Map G: Global Fallback...")
    recent = hist_tx.filter(pl.col("event_ts") >= hist_tx["event_ts"].max() - pl.duration(days=30))
    return (
        recent.group_by("item_id").agg(pl.len().alias("qty"))
        .sort("qty", descending=True).head(global_k)
        .with_columns(pl.int_range(1, pl.len() + 1).cast(pl.Int64).alias("rank"))
        .select(["item_id", "rank"])
    )

def channel_category_trending_map(hist_tx: pl.DataFrame, items: pl.DataFrame, trending: int = 50) -> pl.DataFrame:
    print_status("Building Reference Map H: Category Trending...")
    item_cat = items.select(["item_id", "category_l1"])
    max_ts = hist_tx["event_ts"].max()
    t_recent, t_prior = max_ts - pl.duration(days=30), max_ts - pl.duration(days=60)
    
    recent_sales = hist_tx.filter(pl.col("event_ts") >= t_recent).group_by("item_id").agg(pl.len().alias("qty_recent"))
    prior_sales = hist_tx.filter((pl.col("event_ts") >= t_prior) & (pl.col("event_ts") < t_recent)).group_by("item_id").agg(pl.len().alias("qty_prior"))
    
    momentum = (
        recent_sales.join(prior_sales, on="item_id", how="full").fill_null(0.0)
        .with_columns((pl.col("qty_recent") - pl.col("qty_prior")).alias("momentum"))
        .join(item_cat, on="item_id", how="left").filter(pl.col("category_l1") != "Unknown")
    )
    return (
        momentum.sort(["category_l1", "momentum"], descending=[False, True]).group_by("category_l1").head(trending)
        .with_columns(pl.int_range(1, pl.len() + 1).over("category_l1").cast(pl.Int64).alias("item_rank"))
        .select(["category_l1", "item_id", "item_rank"])
    )

def channel_s1_history_recent(hist_tx: pl.DataFrame, hist_events: pl.DataFrame, n_history: int = 200) -> pl.DataFrame:
    print_status("Running Channel S1: Recent Interaction History (Tx + Events)...")
    cutoff_2m = datetime(2025, 10, 1)
    tx_sub = hist_tx.filter(pl.col("event_ts") >= cutoff_2m).select(["customer_id", "item_id", "event_ts"])
    ev_sub = hist_events.filter(pl.col("event_ts") >= cutoff_2m).select(["customer_id", "item_id", "event_ts"])
    
    combined = pl.concat([tx_sub, ev_sub])
    return (
        combined.sort(["customer_id", "event_ts"], descending=[False, True])
        .unique(subset=["customer_id", "item_id"], keep="first")
        .group_by("customer_id").head(n_history)
        .with_columns(pl.int_range(1, pl.len() + 1).over("customer_id").cast(pl.Int64).alias("rank"))
        .select(["customer_id", "item_id", "rank"])
    )

def train_s2_als(hist_tx: pl.DataFrame) -> tuple:
    print_status("Training Global Advanced ALS CF Model...")
    unique_users = hist_tx.select("customer_id").unique().with_row_index("u_idx")
    unique_items = hist_tx.select("item_id").unique().with_row_index("i_idx")
    
    interactions = (
        hist_tx.group_by(["customer_id", "item_id"]).agg(pl.len().alias("weight"))
        .join(unique_users, on="customer_id").join(unique_items, on="item_id")
    )
    user_item_matrix = sp.csr_matrix(
        (interactions["weight"].to_numpy(), (interactions["u_idx"].to_numpy(), interactions["i_idx"].to_numpy())),
        shape=(unique_users.height, unique_items.height)
    ).astype(np.float32)
    
    model = implicit.als.AlternatingLeastSquares(factors=64, iterations=15, regularization=0.01, random_state=42)
    model.fit(user_item_matrix)
    
    als_u2idx = dict(zip(unique_users["customer_id"], unique_users["u_idx"]))
    als_i_arr = unique_items.sort("i_idx")["item_id"].to_numpy()
    
    del unique_users, unique_items, interactions; gc.collect()
    return model, user_item_matrix, als_u2idx, als_i_arr

def channel_s3_cobuy(hist_tx: pl.DataFrame, s1_path: Path) -> pl.DataFrame:
    print_status("Running Channel S3: Basket Co-buy Graph via Loss-less Bill Pruning...")
    
    s1_df = pl.read_parquet(s1_path)
    user_seed_items = s1_df.filter(pl.col("rank") <= 5).select(["customer_id", "item_id"])
    unique_seeds = user_seed_items.select("item_id").unique()
    
    cutoff_6m = datetime(2025, 6, 1)
    hist_6m = hist_tx.filter(pl.col("event_ts") >= cutoff_6m)
    
    bill_sizes = hist_6m.group_by("bill_id").agg(pl.len().alias("size"))
    hist_filtered = hist_6m.join(bill_sizes, on="bill_id").filter(pl.col("size") <= 50)
    
    valid_bills = hist_filtered.filter(pl.col("item_id").is_in(unique_seeds["item_id"].to_list())).select("bill_id").unique()
    hist_filtered = hist_filtered.join(valid_bills, on="bill_id")
    
    unique_bills = hist_filtered.select("bill_id").unique().with_row_index("bill_idx")
    unique_items = hist_filtered.select("item_id").unique().with_row_index("item_idx")
    
    indexed_tx = (
        hist_filtered.select(["bill_id", "item_id"])
        .join(unique_bills, on="bill_id")
        .join(unique_items, on="item_id")
    )
    
    bill_indices = indexed_tx["bill_idx"].to_numpy()
    item_indices = indexed_tx["item_idx"].to_numpy()
    data = np.ones(len(indexed_tx), dtype=np.float32)
    
    bill_item_matrix = sp.csc_matrix((data, (bill_indices, item_indices)), shape=(unique_bills.height, unique_items.height))
    A_T = bill_item_matrix.T.tocsr()
    item_counts_v = np.array(bill_item_matrix.sum(axis=0)).flatten().astype(np.float32)
    
    del indexed_tx, bill_indices, item_indices, data; gc.collect()
    
    seeds_indexed = unique_seeds.join(unique_items, on="item_id")
    seed_indices = seeds_indexed["item_idx"].to_numpy()
    
    seed_chunk_size = 500
    num_seed_chunks = int(np.ceil(len(seed_indices) / seed_chunk_size))
    print_status(f"Seed optimization mapped. Computing Co-buy only for {len(seed_indices)} active seed items over {num_seed_chunks} chunks...")
    
    append_seed_idx = []
    append_neighbor_idx = []
    append_score = []
    
    for start_idx in range(0, len(seed_indices), seed_chunk_size):
        end_idx = min(start_idx + seed_chunk_size, len(seed_indices))
        curr_chunk_idxs = seed_indices[start_idx:end_idx]
        
        A_slice = bill_item_matrix[:, curr_chunk_idxs]
        co_occur_chunk = A_T.dot(A_slice).toarray()
        
        chunk_item_counts = item_counts_v[curr_chunk_idxs]
        denom = np.sqrt(item_counts_v[:, None] * chunk_item_counts[None, :])
        denom[denom == 0] = 1.0
        scores_chunk = co_occur_chunk / denom
        
        top_k = min(50, unique_items.height - 1)
        for j, seed_idx in enumerate(curr_chunk_idxs):
            scores = scores_chunk[:, j]
            scores[seed_idx] = 0.0
            if np.max(scores) == 0:
                continue
                
            partitioned_indices = np.argpartition(-scores, top_k)[:top_k]
            sorted_top_indices = partitioned_indices[np.argsort(-scores[partitioned_indices])]
            
            for neighbor_idx in sorted_top_indices:
                scr = scores[neighbor_idx]
                if scr > 0.0:
                    append_seed_idx.append(seed_idx)
                    append_neighbor_idx.append(neighbor_idx)
                    append_score.append(scr)
                    
        del A_slice, co_occur_chunk, scores_chunk, denom, chunk_item_counts; gc.collect()
        
    cobuy_graph_idx = pl.DataFrame({
        "item_idx": pl.Series(append_seed_idx, dtype=pl.Int32),
        "neighbor_idx": pl.Series(append_neighbor_idx, dtype=pl.Int32),
        "score": pl.Series(append_score, dtype=pl.Float32)
    })
    del append_seed_idx, append_neighbor_idx, append_score; gc.collect()
    
    unique_items_renamed = unique_items.select([pl.col("item_idx"), pl.col("item_id").alias("item_id_str")])
    cobuy_graph = (
        cobuy_graph_idx
        .join(unique_items_renamed.rename({"item_idx": "item_idx", "item_id_str": "item_A"}), on="item_idx")
        .join(unique_items_renamed.rename({"item_idx": "neighbor_idx", "item_id_str": "item_B"}), on="neighbor_idx")
        .select(["item_A", "item_B", "score"])
    )
    del cobuy_graph_idx, unique_items_renamed; gc.collect()
    
    res = (
        user_seed_items.join(cobuy_graph, left_on="item_id", right_on="item_A", how="inner")
        .select([pl.col("customer_id"), pl.col("item_B").alias("item_id")])
        .unique(subset=["customer_id", "item_id"])
        .with_columns(pl.int_range(1, pl.len() + 1).over("customer_id").cast(pl.Int64).alias("rank"))
    )
    
    del hist_6m, bill_sizes, hist_filtered, cobuy_graph, s1_df, user_seed_items, unique_bills, unique_items, bill_item_matrix, A_T, item_counts_v, seeds_indexed, seed_indices, unique_seeds, valid_bills; gc.collect()
    return res

def channel_s4_monthly_pop_map(hist_tx: pl.DataFrame, n_pop: int = 100) -> pl.DataFrame:
    return (
        hist_tx.filter((pl.col("event_ts") >= datetime(2025, 11, 1)) & (pl.col("event_ts") < datetime(2025, 12, 1)))
        .group_by("item_id").agg(pl.len().alias("sales"))
        .sort("sales", descending=True).head(n_pop)
        .with_row_index("rank", offset=1).with_columns(pl.col("rank").cast(pl.Int64))
        .select(["item_id", "rank"])
    )

def channel_s5_monthly_trend_map(hist_tx: pl.DataFrame, n_trend: int = 20) -> pl.DataFrame:
    sales_m1 = hist_tx.filter((pl.col("event_ts") >= datetime(2025, 11, 1)) & (pl.col("event_ts") < datetime(2025, 12, 1))).group_by("item_id").agg(pl.len().alias("sales_m1"))
    sales_m2 = hist_tx.filter((pl.col("event_ts") >= datetime(2025, 10, 1)) & (pl.col("event_ts") < datetime(2025, 11, 1))).group_by("item_id").agg(pl.len().alias("sales_m2"))
    
    return (
        sales_m1.join(sales_m2, on="item_id", how="left").fill_null(0)
        .filter(pl.col("sales_m1") >= 10)
        .with_columns((pl.col("sales_m1") / (pl.col("sales_m2") + 1.0)).alias("trend_ratio"))
        .sort("trend_ratio", descending=True).head(n_trend)
        .with_row_index("rank", offset=1).with_columns(pl.col("rank").cast(pl.Int64))
        .select(["item_id", "rank"])
    )

def channel_s6_full_history(hist_tx: pl.DataFrame, n_full_history: int = 300) -> pl.DataFrame:
    print_status("Running Channel S6: Full History Freq Count...")
    return (
        hist_tx.group_by(["customer_id", "item_id"]).agg(pl.len().alias("ui_purchases"))
        .sort(["customer_id", "ui_purchases"], descending=[False, True])
        .group_by("customer_id").head(n_full_history)
        .with_columns(pl.int_range(1, pl.len() + 1).over("customer_id").cast(pl.Int64).alias("rank"))
        .select(["customer_id", "item_id", "rank"])
    )

# ==============================================================================
# DISK-BASED STREAMING CHUNK FUSION ENGINE (STREAMING MAPPING)
# ==============================================================================
def evaluate_and_fuse(channel_files: dict, ref_maps: dict, profile_paths: dict, latent_models: dict, max_final_candidates: int = 200) -> pl.DataFrame:
    print_status("Fusing candidates dynamically using Disk-Based streaming RRF & user chunking...")
    
    item_features = pl.read_parquet(profile_paths["item_features"])
    unique_users = pl.read_parquet(profile_paths["archetypes"]).select("customer_id").unique()["customer_id"].to_list()
    
    u_emb, i_emb, i2i_sim, mtx, u2idx, i_arr = latent_models["cf_latent"]
    als_model, als_matrix, als_u2idx, als_i_arr = latent_models["als"]
    k_svd = latent_models["k_svd"]
    
    # TỐI ƯU I/O CHURN: Caching toàn bộ profile thưa lên RAM nền trước vòng lặp
    print_status("Pre-loading lightweight sparse profile bases into memory to eliminate repetitive disk I/O...")
    full_archetypes = pl.read_parquet(profile_paths["archetypes"])
    full_user_features = pl.read_parquet(profile_paths["user_features"])
    full_user_loc = pl.read_parquet(profile_paths["user_loc"])
    full_user_cats = pl.read_parquet(profile_paths["user_cats"])
    full_user_brands = pl.read_parquet(profile_paths["user_brands"])
    
    chunk_size = 2000
    num_chunks = int(np.ceil(len(unique_users) / chunk_size))
    
    weight_map = {
        "Habitual": {"A_history": 5.0, "B_local": 1.0, "C_svd": 0.5, "D_i2i": 0.5, "E_cat": 2.0, "F_brand": 3.0, "G_global": 0.1, "H_trend": 0.5, "S1_hist_recent": 4.0, "S2_als": 1.0, "S3_cobuy": 1.0, "S4_month_pop": 0.2, "S5_month_trend": 0.5, "S6_full_hist": 4.5},
        "Explorer": {"A_history": 1.0, "B_local": 1.0, "C_svd": 4.0, "D_i2i": 4.0, "E_cat": 3.0, "F_brand": 1.0, "G_global": 1.0, "H_trend": 2.0, "S1_hist_recent": 1.5, "S2_als": 4.5, "S3_cobuy": 3.5, "S4_month_pop": 1.0, "S5_month_trend": 2.5, "S6_full_hist": 0.5},
        "Dormant":  {"A_history": 3.0, "B_local": 4.0, "C_svd": 0.5, "D_i2i": 0.5, "E_cat": 1.0, "F_brand": 2.0, "G_global": 5.0, "H_trend": 1.0, "S1_hist_recent": 2.0, "S2_als": 0.5, "S3_cobuy": 0.5, "S4_month_pop": 4.5, "S5_month_trend": 2.0, "S6_full_hist": 3.5},
        "New":      {"A_history": 0.5, "B_local": 3.0, "C_svd": 2.5, "D_i2i": 2.5, "E_cat": 3.0, "F_brand": 1.0, "G_global": 4.0, "H_trend": 4.0, "S1_hist_recent": 0.5, "S2_als": 3.0, "S3_cobuy": 2.0, "S4_month_pop": 4.5, "S5_month_trend": 4.5, "S6_full_hist": 0.1},
        "Standard": {"A_history": 2.0, "B_local": 2.0, "C_svd": 1.5, "D_i2i": 1.5, "E_cat": 2.0, "F_brand": 2.0, "G_global": 1.0, "H_trend": 1.5, "S1_hist_recent": 2.0, "S2_als": 2.0, "S3_cobuy": 2.0, "S4_month_pop": 1.5, "S5_month_trend": 2.0, "S6_full_hist": 2.0},
    }
    w_df = pl.DataFrame([{"archetype": a, "channel": c, "ch_weight": w} for a, cw in weight_map.items() for c, w in cw.items()])
    
    # KHỞI TẠO DISK APPEND STREAMING WRITER CHỐNG TÍCH LŨY RAM
    fusion_output_path = CACHE_DIR / "fused_candidates_stream.parquet"
    arrow_schema = pa.schema([
        ('customer_id', pa.int32()),
        ('item_id', pa.large_string()),
        ('final_rank', pa.int64()),
        ('archetype', pa.large_string()),
        ('rrf_score', pa.float32()),
        ('svd_similarity', pa.float32()),
        ('price_ratio', pa.float32())
    ])
    writer = pq.ParquetWriter(str(fusion_output_path), arrow_schema, compression='SNAPPY')
    
    for i in range(0, len(unique_users), chunk_size):
        curr_chunk_idx = (i // chunk_size) + 1
        if curr_chunk_idx % 50 == 0 or curr_chunk_idx == 1 or curr_chunk_idx == num_chunks:
            print_status(f"Processing Fusion Chunk {curr_chunk_idx}/{num_chunks}...")
        
        chunk_u = unique_users[i : i + chunk_size]
        
        arch_chunk = full_archetypes.filter(pl.col("customer_id").is_in(chunk_u))
        user_feat_chunk = full_user_features.filter(pl.col("customer_id").is_in(chunk_u))
        
        chunk_cands_list = []
        
        for c_name, f_path in channel_files.items():
            df_c = pl.scan_parquet(f_path).filter(pl.col("customer_id").is_in(chunk_u)).collect()
            if not df_c.is_empty():
                df_c = df_c.with_columns(pl.lit(c_name).alias("channel")).select(["customer_id", "item_id", "rank", "channel"])
                chunk_cands_list.append(df_c)
                
        chunk_u_mapped = [u2idx[u] for u in chunk_u if u in u2idx]
        chunk_users_matched = [u for u in chunk_u if u in u2idx]
        
        if chunk_u_mapped:
            # Channel C
            scores_svd = u_emb[chunk_u_mapped] @ i_emb.T
            top_svd = np.argsort(-scores_svd, axis=1)[:, :100]
            df_c_svd = pl.DataFrame({
                "customer_id": pl.Series(np.repeat(chunk_users_matched, 100), dtype=pl.Int32),
                "item_id": i_arr[top_svd.flatten()],
                "rank": pl.Series(np.tile(np.arange(1, 101, dtype=np.int64), len(chunk_users_matched)), dtype=pl.Int64),
                "channel": pl.Series(["C_svd"] * (len(chunk_users_matched) * 100), dtype=pl.Utf8)
            }).select(["customer_id", "item_id", "rank", "channel"])
            chunk_cands_list.append(df_c_svd)
            
            # Channel D
            scores_i2i = mtx[chunk_u_mapped].dot(i2i_sim).toarray()
            top_i2i = np.argsort(-scores_i2i, axis=1)[:, :100]
            mask = np.take_along_axis(scores_i2i, top_i2i, axis=1) > 0.0
            df_d_i2i = pl.DataFrame({
                "customer_id": pl.Series(np.repeat(chunk_users_matched, 100)[mask.flatten()], dtype=pl.Int32),
                "item_id": i_arr[top_i2i.flatten()][mask.flatten()],
                "rank": pl.Series(np.tile(np.arange(1, 101, dtype=np.int64), len(chunk_users_matched))[mask.flatten()], dtype=pl.Int64),
                "channel": pl.Series(["D_i2i"] * np.sum(mask), dtype=pl.Utf8)
            }).select(["customer_id", "item_id", "rank", "channel"])
            chunk_cands_list.append(df_d_i2i)
            del scores_svd, top_svd, scores_i2i, top_i2i, mask, df_c_svd, df_d_i2i; gc.collect()

        # Channel S2
        chunk_als_u = [als_u2idx[u] for u in chunk_u if u in als_u2idx]
        chunk_users_als = [u for u in chunk_u if u in als_u2idx]
        if chunk_als_u:
            ids, _ = als_model.recommend(chunk_als_u, als_matrix[chunk_als_u], N=100, filter_already_liked_items=False)
            df_s2_als = pl.DataFrame({
                "customer_id": np.repeat(chunk_users_als, 100).astype(np.int32),
                "item_id": als_i_arr[ids.flatten()],
                "rank": np.tile(np.arange(1, 101, dtype=np.int64), len(chunk_users_als)),
                "channel": pl.Series(["S2_als"] * (len(chunk_users_als) * 100), dtype=pl.Utf8)
            }).select(["customer_id", "item_id", "rank", "channel"])
            chunk_cands_list.append(df_s2_als)
            del ids, df_s2_als; gc.collect()

        u_loc_c = full_user_loc.filter(pl.col("customer_id").is_in(chunk_u))
        df_b = u_loc_c.join(ref_maps["B_local"], on="location", how="inner").drop("location")
        chunk_cands_list.append(df_b.with_columns(pl.lit("B_local").alias("channel")).select(["customer_id", "item_id", "rank", "channel"]))
        
        u_cat_c = full_user_cats.filter(pl.col("customer_id").is_in(chunk_u))
        df_e = u_cat_c.join(ref_maps["E_cat"], on="category_l1", how="inner").drop("category_l1")
        df_e = df_e.with_columns(pl.int_range(1, pl.len() + 1).over("customer_id").cast(pl.Int64).alias("rank"))
        chunk_cands_list.append(df_e.with_columns(pl.lit("E_cat").alias("channel")).select(["customer_id", "item_id", "rank", "channel"]))
        
        u_brd_c = full_user_brands.filter(pl.col("customer_id").is_in(chunk_u))
        df_f = u_brd_c.join(ref_maps["F_brand"], on="brand", how="inner").drop("brand")
        df_f = df_f.with_columns(pl.int_range(1, pl.len() + 1).over("customer_id").cast(pl.Int64).alias("rank"))
        chunk_cands_list.append(df_f.with_columns(pl.lit("F_brand").alias("channel")).select(["customer_id", "item_id", "rank", "channel"]))
        
        df_h = u_cat_c.join(ref_maps["H_trend"], on="category_l1", how="inner").drop("category_l1")
        df_h = df_h.with_columns(pl.int_range(1, pl.len() + 1).over("customer_id").cast(pl.Int64).alias("rank"))
        chunk_cands_list.append(df_h.with_columns(pl.lit("H_trend").alias("channel")).select(["customer_id", "item_id", "rank", "channel"]))
        
        num_chunk_users = len(chunk_u)
        
        g_arr_items = ref_maps["G_global"]["item_id"].to_numpy()
        g_arr_ranks = ref_maps["G_global"]["rank"].to_numpy()
        k_g = len(g_arr_items)
        df_g = pl.DataFrame({
            "customer_id": pl.Series(np.repeat(chunk_u, k_g), dtype=pl.Int32),
            "item_id": np.tile(g_arr_items, num_chunk_users),
            "rank": pl.Series(np.tile(g_arr_ranks, num_chunk_users), dtype=pl.Int64),
            "channel": pl.Series(["G_global"] * (num_chunk_users * k_g), dtype=pl.Utf8)
        }).select(["customer_id", "item_id", "rank", "channel"])
        chunk_cands_list.append(df_g)
        
        s4_arr_items = ref_maps["S4_month_pop"]["item_id"].to_numpy()
        s4_arr_ranks = ref_maps["S4_month_pop"]["rank"].to_numpy()
        k_s4 = len(s4_arr_items)
        df_s4 = pl.DataFrame({
            "customer_id": pl.Series(np.repeat(chunk_u, k_s4), dtype=pl.Int32),
            "item_id": np.tile(s4_arr_items, num_chunk_users),
            "rank": pl.Series(np.tile(s4_arr_ranks, num_chunk_users), dtype=pl.Int64),
            "channel": pl.Series(["S4_month_pop"] * (num_chunk_users * k_s4), dtype=pl.Utf8)
        }).select(["customer_id", "item_id", "rank", "channel"])
        chunk_cands_list.append(df_s4)
        
        s5_arr_items = ref_maps["S5_month_trend"]["item_id"].to_numpy()
        s5_arr_ranks = ref_maps["S5_month_trend"]["rank"].to_numpy()
        k_s5 = len(s5_arr_items)
        df_s5 = pl.DataFrame({
            "customer_id": pl.Series(np.repeat(chunk_u, k_s5), dtype=pl.Int32),
            "item_id": np.tile(s5_arr_items, num_chunk_users),
            "rank": pl.Series(np.tile(s5_arr_ranks, num_chunk_users), dtype=pl.Int64),
            "channel": pl.Series(["S5_month_trend"] * (num_chunk_users * k_s5), dtype=pl.Utf8)
        }).select(["customer_id", "item_id", "rank", "channel"])
        chunk_cands_list.append(df_s5)

        stacked_chunk = pl.concat(chunk_cands_list)
        merged = stacked_chunk.join(arch_chunk.select(["customer_id", "archetype"]), on="customer_id", how="inner")
        
        scored = (
            merged.join(w_df, on=["archetype", "channel"], how="left")
            .with_columns(pl.col("ch_weight").fill_null(1.0))
            .with_columns((pl.col("ch_weight") / (pl.col("rank") + 60.0)).alias("score"))
            .group_by(["customer_id", "item_id", "archetype"])
            .agg(pl.col("score").sum().alias("rrf_score"))
        )
        scored = scored.join(user_feat_chunk, on="customer_id", how="left").join(item_features, on="item_id", how="left")
        
        svd_dot_expr = pl.lit(0.0)
        for idx in range(k_svd):
            svd_dot_expr = svd_dot_expr + (pl.col(f"u_svd_{idx}").fill_null(0.0) * pl.col(f"c_svd_{idx}").fill_null(0.0))
            
        scored = scored.with_columns([
            svd_dot_expr.alias("svd_similarity"),
            (pl.col("item_avg_price") / (pl.col("u_avg_price") + 1e-6)).alias("price_ratio")
        ])
        
        final_chunk = (
            scored.sort(["customer_id", "rrf_score", "svd_similarity"], descending=[False, True, True])
            .with_columns(pl.int_range(1, pl.len() + 1).over("customer_id").cast(pl.Int64).alias("final_rank"))
            .filter(pl.col("final_rank") <= max_final_candidates)
            .select(["customer_id", "item_id", "final_rank", "archetype", "rrf_score", "svd_similarity", "price_ratio"])
            .with_columns([
                pl.col("customer_id").cast(pl.Int32),
                pl.col("item_id").cast(pl.Utf8),
                pl.col("final_rank").cast(pl.Int64),
                pl.col("archetype").cast(pl.Utf8),
                pl.col("rrf_score").cast(pl.Float32),
                pl.col("svd_similarity").cast(pl.Float32),
                pl.col("price_ratio").cast(pl.Float32),
            ])
        )
        
        writer.write_table(final_chunk.to_arrow())
        del stacked_chunk, merged, scored, final_chunk, arch_chunk, user_feat_chunk, df_g, df_s4, df_s5; gc.collect()

    writer.close()
    
    print_status("Streaming loop finished successfully. Kept lean fused collection on disk!")
    
    del full_archetypes, full_user_features, full_user_loc, full_user_cats, full_user_brands; gc.collect()
    print_status("Successfully generated final combined candidate collection.")
    return fusion_output_path

# ==============================================================================
# PIPELINE EXECUTION ENGINE
# ==============================================================================
def run_pipeline(max_cands: int):
    print_status("--- Starting Full Feature-Integrated PIR Pipeline v2 (Safe Full Scale) ---")
    
    items = load_items()
    hist_tx, valid_tx, target_users = load_history_data()
    
    if USER_LIMIT > 0:
        print_status(f"[SMOKE TEST ACTIVATED] Restricting entire pipeline scale down to exactly {USER_LIMIT} users.")
        # Make sure we sample users who are actually in the validation set so we have ground truth
        valid_users = valid_tx.select("customer_id").unique()
        target_users = valid_users.sample(n=USER_LIMIT, seed=42)
        hist_tx = hist_tx.filter(pl.col("customer_id").is_in(target_users["customer_id"]))
        
    cutoff_dt = hist_tx["event_ts"].max()
    hist_events = load_event_history(target_users, cutoff_dt)
    
    profile_paths = {}
    p_uf, p_if, k_svd = precompute_advanced_features(hist_tx, items)
    profile_paths["user_features"] = p_uf
    profile_paths["item_features"] = p_if
    
    p_arch, p_uloc, p_ucats, p_ubrds = compute_user_archetypes_and_profiles(hist_tx, items, target_users)
    profile_paths["archetypes"] = p_arch
    profile_paths["user_loc"] = p_uloc
    profile_paths["user_cats"] = p_ucats
    profile_paths["user_brands"] = p_ubrds
    
    ref_maps = {}
    ref_maps["B_local"] = channel_local_popular_map(hist_tx, top_k=500)
    ref_maps["E_cat"] = channel_category_popular_map(hist_tx, items, bestsellers_per_category=80)
    ref_maps["F_brand"] = channel_brand_popular_map(hist_tx, items, bestsellers_per_brand=50)
    ref_maps["G_global"] = channel_global_popular_map(hist_tx, global_k=300)
    ref_maps["H_trend"] = channel_category_trending_map(hist_tx, items, trending=50)
    ref_maps["S4_month_pop"] = channel_s4_monthly_pop_map(hist_tx, n_pop=100)
    ref_maps["S5_month_trend"] = channel_s5_monthly_trend_map(hist_tx, n_trend=20)
    
    latent_models = {}
    latent_models["cf_latent"] = train_cf_latent(hist_tx)
    latent_models["als"] = train_s2_als(hist_tx)
    latent_models["k_svd"] = k_svd
    
    channel_files = {}
    
    # Channel A
    df_a = channel_history(hist_tx)
    p_a = CACHE_DIR / "ch_A_history.parquet"
    df_a.write_parquet(p_a)
    channel_files["A_history"] = p_a
    del df_a; gc.collect()
    
    # Channel S1
    df_s1 = channel_s1_history_recent(hist_tx, hist_events, n_history=200)
    p_s1 = CACHE_DIR / "ch_S1_hist_recent.parquet"
    df_s1.write_parquet(p_s1)
    channel_files["S1_hist_recent"] = p_s1
    del df_s1; gc.collect()
    
    # Channel S3
    if ENABLE_COBUY:
        df_s3 = channel_s3_cobuy(hist_tx, channel_files["S1_hist_recent"])
        p_s3 = CACHE_DIR / "ch_S3_cobuy.parquet"
        df_s3.write_parquet(p_s3)
        channel_files["S3_cobuy"] = p_s3
        del df_s3; gc.collect()
    else:
        print_status("[FLAG BYPASS] ENABLE_COBUY is False. Skipping Basket Co-buy Graph module to secure system RAM.")
        
    # Channel S6
    df_s6 = channel_s6_full_history(hist_tx, n_full_history=300)
    p_s6 = CACHE_DIR / "ch_S6_full_hist.parquet"
    df_s6.write_parquet(p_s6)
    channel_files["S6_full_hist"] = p_s6
    
    # TRÍCH XUẤT GROUND TRUTH VALIDATION TRƯỚC KHI XÓA HIST_TX KHỎI RAM
    print_status("Extracting Validation ground truth for Recall validation...")
    valid_gt_users = (
        valid_tx
        .group_by("customer_id")
        .agg(pl.col("item_id").unique().alias("true_items"))
    )
    
    # BƯỚC GIẢI PHÓNG RAM QUYẾT ĐỊNH
    del df_s6, hist_tx, target_users, hist_events; gc.collect() 
    print_status("Main transaction records cleared from RAM. Commencing Streaming Fusion Engine...")
    
    fusion_output_path = evaluate_and_fuse(
        channel_files, ref_maps, profile_paths, latent_models, max_final_candidates=max_cands
    )
    
    output_filename = "candidates_pir_integrated_v2.parquet"
    if Path(output_filename).exists():
        Path(output_filename).unlink()
    fusion_output_path.rename(output_filename)
    
    # TÍNH TOÁN RECALL@K TRÊN VALIDATION (7 DAYS)
    print_status("Evaluating Recall@K against Validation ground truth...")
    
    # Only load validation users to save RAM
    valid_cands = (
        pl.scan_parquet(output_filename)
        .filter(pl.col("customer_id").is_in(valid_gt_users["customer_id"]))
        .group_by("customer_id", maintain_order=True)
        .agg(pl.col("item_id").alias("recommended_items"))
        .collect(engine="cpu")
    )
    
    recall_df = valid_cands.join(valid_gt_users, on="customer_id", how="inner")
    if not recall_df.is_empty():
        recall_metrics = (
            recall_df
            .with_columns([
                pl.col("recommended_items").list.set_intersection(pl.col("true_items")).list.len().alias("hits"),
                pl.col("true_items").list.len().alias("true_count")
            ])
            .select([
                (pl.col("hits") > 0).cast(pl.Float32).mean().alias("user_recall"),
                (pl.col("hits") / pl.col("true_count")).mean().alias("item_recall")
            ])
        )
        user_recall = recall_metrics["user_recall"][0]
        item_recall = recall_metrics["item_recall"][0]
        print(f"\n==================================================")
        print(f"»»» Validation User-level Recall@{max_cands}: {user_recall:.4f}")
        print(f"»»» Validation Item-level Recall@{max_cands}: {item_recall:.4f}")
        print(f"==================================================\n")
        del recall_metrics
    else:
        print_status("Warning: No matching users between candidates and Validation transactions.")
        
    del recall_df, valid_gt_users, valid_cands; gc.collect()
        
    print_status("Cleaning temporary disk cache files...")
    all_temp_paths = list(channel_files.values()) + list(profile_paths.values())
    for f_path in all_temp_paths:
        if f_path.exists():
            f_path.unlink()
    CACHE_DIR.rmdir()
    
    print_status(f"✓ Pipeline completed successfully! Clean submission file saved at '{output_filename}'")

if __name__ == "__main__":
    MAX_CANDIDATES = 500
    run_pipeline(max_cands=MAX_CANDIDATES)