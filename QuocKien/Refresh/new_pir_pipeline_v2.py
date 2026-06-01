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
ITEMS_PATH = ROOT / "items.parquet"

def load_items() -> pl.DataFrame:
    print("Loading items...")
    return pl.read_parquet(ITEMS_PATH).select([
        pl.col("item_id").cast(pl.Utf8),
        pl.col("category_l1").cast(pl.Utf8).fill_null("Unknown"),
        pl.col("category_l2").cast(pl.Utf8).fill_null("Unknown"),
        pl.col("brand").cast(pl.Utf8).fill_null("Unknown"),
        pl.col("price").cast(pl.Float32).fill_null(0.0),
    ])

def sample_active_december_users(sample_n: int, seed: int = 42) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    print(f"Sampling {sample_n} active December users...")
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

def load_history_data(sampled_users: pl.DataFrame) -> pl.DataFrame:
    print("Loading transaction history (Months 1-11)...")
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
    return hist_tx

def compute_user_archetypes(hist_tx: pl.DataFrame, items: pl.DataFrame, target_users: pl.DataFrame) -> pl.DataFrame:
    print("Computing deep behavioral user archetypes...")
    max_ts = datetime(2025, 12, 1)
    hist_items = hist_tx.join(items, on="item_id", how="left")
    
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
    
    res = target_users.join(profile, on="customer_id", how="left").with_columns(
        pl.col("archetype").fill_null("Dormant")
    )
    return res

def channel_history(hist_tx: pl.DataFrame, target_users: pl.DataFrame) -> pl.DataFrame:
    print("Running Channel A: Purchase History...")
    return (
        hist_tx.join(target_users, on="customer_id", how="inner")
        .group_by(["customer_id", "item_id"])
        .agg([pl.len().alias("purchase_count"), pl.col("event_ts").max().alias("last_ts")])
        .sort(["customer_id", "last_ts", "purchase_count"], descending=[False, True, True])
        .with_columns(pl.int_range(1, pl.len() + 1).over("customer_id").cast(pl.Int64).alias("rank"))
        .select(["customer_id", "item_id", "rank"])
    )

def channel_local_popular(hist_tx: pl.DataFrame, target_users: pl.DataFrame, top_k: int = 400) -> pl.DataFrame:
    print("Running Channel B: Local Store Bestsellers...")
    user_loc = (
        hist_tx.join(target_users, on="customer_id", how="inner")
        .group_by(["customer_id", "location"]).agg(pl.len().alias("loc_qty"))
        .sort(["customer_id", "loc_qty"], descending=[False, True])
        .group_by("customer_id").head(1).select(["customer_id", "location"])
    )
    recent_tx = hist_tx.filter(pl.col("event_ts") >= hist_tx["event_ts"].max() - pl.duration(days=60))
    local_bestsellers = (
        recent_tx.group_by(["location", "item_id"]).agg(pl.col("quantity").sum().alias("qty"))
        .sort(["location", "qty"], descending=[False, True]).group_by("location").head(top_k)
        .with_columns(pl.int_range(1, pl.len() + 1).over("location").cast(pl.Int64).alias("rank"))
    )
    return user_loc.join(local_bestsellers, on="location", how="inner").select(["customer_id", "item_id", "rank"])

def channel_category_popular(hist_tx: pl.DataFrame, items: pl.DataFrame, target_users: pl.DataFrame, top_categories_per_user: int = 3, bestsellers_per_category: int = 50, lookback_days: int = 45) -> pl.DataFrame:
    print("Running Channel E: Category-based Bestsellers...")
    item_cat = items.select(["item_id", "category_l1"])
    user_cats = (
        hist_tx.join(target_users, on="customer_id", how="inner").join(item_cat, on="item_id", how="left")
        .filter(pl.col("category_l1") != "Unknown").group_by(["customer_id", "category_l1"]).agg(pl.col("quantity").sum().alias("cat_qty"))
        .sort(["customer_id", "cat_qty"], descending=[False, True]).group_by("customer_id").head(top_categories_per_user)
        .with_columns(pl.int_range(1, pl.len() + 1).over("customer_id").cast(pl.Int64).alias("cat_rank"))
        .select(["customer_id", "category_l1", "cat_rank"])
    )
    max_ts = hist_tx["event_ts"].max()
    recent_tx = hist_tx.filter(pl.col("event_ts") >= max_ts - pl.duration(days=lookback_days))
    cat_bestsellers = (
        recent_tx.join(item_cat, on="item_id", how="left").filter(pl.col("category_l1") != "Unknown")
        .group_by(["category_l1", "item_id"]).agg(pl.col("quantity").sum().alias("qty"))
        .sort(["category_l1", "qty"], descending=[False, True]).group_by("category_l1").head(bestsellers_per_category)
        .with_columns(pl.int_range(1, pl.len() + 1).over("category_l1").cast(pl.Int64).alias("item_rank"))
        .select(["category_l1", "item_id", "item_rank"])
    )
    return (
        user_cats.join(cat_bestsellers, on="category_l1", how="inner").sort(["customer_id", "cat_rank", "item_rank"])
        .group_by("customer_id").head(bestsellers_per_category * top_categories_per_user)
        .with_columns(pl.int_range(1, pl.len() + 1).over("customer_id").cast(pl.Int64).alias("rank"))
        .select(["customer_id", "item_id", "rank"])
    )

def channel_brand_popular(hist_tx: pl.DataFrame, items: pl.DataFrame, target_users: pl.DataFrame, top_brands_per_user: int = 3, bestsellers_per_brand: int = 30, lookback_days: int = 60) -> pl.DataFrame:
    print("Running Channel F: Brand-based Bestsellers...")
    item_brand = items.select(["item_id", "brand"])
    user_brands = (
        hist_tx.join(target_users, on="customer_id", how="inner").join(item_brand, on="item_id", how="left")
        .filter((pl.col("brand") != "Unknown") & (pl.col("brand") != "Không xác định")).group_by(["customer_id", "brand"]).agg(pl.col("quantity").sum().alias("brand_qty"))
        .sort(["customer_id", "brand_qty"], descending=[False, True]).group_by("customer_id").head(top_brands_per_user)
        .with_columns(pl.int_range(1, pl.len() + 1).over("customer_id").cast(pl.Int64).alias("brand_rank"))
        .select(["customer_id", "brand", "brand_rank"])
    )
    max_ts = hist_tx["event_ts"].max()
    recent_tx = hist_tx.filter(pl.col("event_ts") >= max_ts - pl.duration(days=lookback_days))
    brand_bestsellers = (
        recent_tx.join(item_brand, on="item_id", how="left").filter((pl.col("brand") != "Unknown") & (pl.col("brand") != "Không xác định"))
        .group_by(["brand", "item_id"]).agg(pl.col("quantity").sum().alias("qty"))
        .sort(["brand", "qty"], descending=[False, True]).group_by("brand").head(bestsellers_per_brand)
        .with_columns(pl.int_range(1, pl.len() + 1).over("brand").cast(pl.Int64).alias("item_rank"))
        .select(["brand", "item_id", "item_rank"])
    )
    return (
        user_brands.join(brand_bestsellers, on="brand", how="inner").sort(["customer_id", "brand_rank", "item_rank"])
        .group_by("customer_id").head(bestsellers_per_brand * top_brands_per_user)
        .with_columns(pl.int_range(1, pl.len() + 1).over("customer_id").cast(pl.Int64).alias("rank"))
        .select(["customer_id", "item_id", "rank"])
    )

def channel_global_popular(hist_tx: pl.DataFrame, target_users: pl.DataFrame, global_k: int = 200) -> pl.DataFrame:
    print("Running Channel G: Global Fallback...")
    max_ts = hist_tx["event_ts"].max()
    recent = hist_tx.filter(pl.col("event_ts") >= max_ts - pl.duration(days=30))
    global_top = (
        recent.group_by("item_id").agg(pl.col("quantity").sum().alias("qty"))
        .sort("qty", descending=True).head(global_k)
        .with_columns(pl.int_range(1, pl.len() + 1).cast(pl.Int64).alias("rank"))
        .select(["item_id", "rank"])
    )
    return target_users.join(global_top.with_columns(pl.lit(1).alias("_k")), how="cross").drop("_k")

def channel_category_trending(hist_tx: pl.DataFrame, items: pl.DataFrame, target_users: pl.DataFrame, top_cats: int = 3, trending: int = 30) -> pl.DataFrame:
    print("Running Channel H: Category Trending...")
    item_cat = items.select(["item_id", "category_l1"])
    user_cats = (
        hist_tx.join(target_users, on="customer_id", how="inner").join(item_cat, on="item_id", how="left")
        .filter(pl.col("category_l1") != "Unknown").group_by(["customer_id", "category_l1"]).agg(pl.col("quantity").sum().alias("cat_qty"))
        .sort(["customer_id", "cat_qty"], descending=[False, True]).group_by("customer_id").head(top_cats)
        .with_columns(pl.int_range(1, pl.len() + 1).over("customer_id").cast(pl.Int64).alias("cat_rank"))
        .select(["customer_id", "category_l1", "cat_rank"])
    )
    max_ts = hist_tx["event_ts"].max()
    t_recent = max_ts - pl.duration(days=30)
    t_prior = max_ts - pl.duration(days=60)
    
    recent_sales = hist_tx.filter(pl.col("event_ts") >= t_recent).group_by("item_id").agg(pl.col("quantity").sum().alias("qty_recent"))
    prior_sales = hist_tx.filter((pl.col("event_ts") >= t_prior) & (pl.col("event_ts") < t_recent)).group_by("item_id").agg(pl.col("quantity").sum().alias("qty_prior"))
    
    momentum = (
        recent_sales.join(prior_sales, on="item_id", how="outer").fill_null(0.0)
        .with_columns((pl.col("qty_recent") - pl.col("qty_prior")).alias("momentum"))
        .join(item_cat, on="item_id", how="left").filter(pl.col("category_l1") != "Unknown")
    )
    cat_trending = (
        momentum.sort(["category_l1", "momentum"], descending=[False, True]).group_by("category_l1").head(trending)
        .with_columns(pl.int_range(1, pl.len() + 1).over("category_l1").cast(pl.Int64).alias("item_rank"))
        .select(["category_l1", "item_id", "item_rank"])
    )
    return (
        user_cats.join(cat_trending, on="category_l1", how="inner").sort(["customer_id", "cat_rank", "item_rank"])
        .group_by("customer_id").head(trending * top_cats)
        .with_columns(pl.int_range(1, pl.len() + 1).over("customer_id").cast(pl.Int64).alias("rank"))
        .select(["customer_id", "item_id", "rank"])
    )

def channel_cf_latent(
    hist_tx: pl.DataFrame,
    training_users: pl.DataFrame,
    target_users: pl.DataFrame,
    svd_k: int = 400,
    i2i_k: int = 400,
    svd_components: int = 100,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    print(f"Running Channel C & D: Latent CF (svd_k={svd_k}, i2i_k={i2i_k})...")
    train_users = training_users["customer_id"].unique().to_list()
    hist_train = hist_tx.filter(pl.col("customer_id").is_in(train_users))
    if hist_train.is_empty():
        empty = pl.DataFrame(schema={"customer_id": pl.Int32, "item_id": pl.Utf8, "rank": pl.Int64})
        return empty, empty
    
    max_ts = hist_train["event_ts"].max()
    tx_weighted = (
        hist_train
        .with_columns(((pl.lit(max_ts) - pl.col("event_ts")).dt.total_days() / 30.0).alias("months_ago"))
        .with_columns((pl.col("quantity") * pl.lit(0.70).pow(pl.col("months_ago"))).cast(pl.Float32).alias("weight"))
        .group_by(["customer_id", "item_id"]).agg(pl.col("weight").sum().alias("weight"))
    )
    
    u_map = tx_weighted["customer_id"].unique()
    i_map = hist_tx["item_id"].unique()
    u_df = pl.DataFrame({"customer_id": u_map, "u_idx": np.arange(len(u_map), dtype=np.int32)})
    i_df = pl.DataFrame({"item_id": i_map, "i_idx": np.arange(len(i_map), dtype=np.int32)})
    hybrid_indexed = tx_weighted.join(u_df, on="customer_id", how="inner").join(i_df, on="item_id", how="inner")
    
    rows = hybrid_indexed["u_idx"].to_numpy()
    cols = hybrid_indexed["i_idx"].to_numpy()
    data = hybrid_indexed["weight"].to_numpy()
    mtx = csr_matrix((data, (rows, cols)), shape=(len(u_map), len(i_map)))
    u2idx = dict(zip(u_df["customer_id"], u_df["u_idx"]))
    idx2i = i_map.to_list()
    i_arr = np.array(idx2i)
    
    print("Fitting SVD...")
    svd = TruncatedSVD(n_components=svd_components, random_state=42)
    u_emb = svd.fit_transform(mtx)
    i_emb = svd.components_.T
    
    print("Computing I2I...")
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
        c_i2i.append(pl.DataFrame({
            "customer_id": pl.Series(np.repeat(chunk_users, i2i_k)[mask.flatten()], dtype=pl.Int32),
            "item_id": i_arr[top_i2i.flatten()][mask.flatten()],
            "rank": pl.Series(i2i_ranks[mask.flatten()], dtype=pl.Int64)
        }))
        
    df_svd = pl.concat(c_svd).unique() if c_svd else pl.DataFrame(schema={"customer_id": pl.Int32, "item_id": pl.Utf8, "rank": pl.Int64})
    df_i2i = pl.concat(c_i2i).unique() if c_i2i else pl.DataFrame(schema={"customer_id": pl.Int32, "item_id": pl.Utf8, "rank": pl.Int64})
    return df_svd, df_i2i

def evaluate_and_fuse(channels: dict, truth: pl.DataFrame, user_archetypes: pl.DataFrame, params: dict):
    print("Fusing candidates dynamically based on archetype...")
    
    all_cands = []
    for c_name, df in channels.items():
        if df is not None and not df.is_empty():
            all_cands.append(df.with_columns(pl.lit(c_name).alias("channel")))
            
    stacked = pl.concat(all_cands)
    unique_users = user_archetypes["customer_id"].unique().to_list()
    chunk_size = 5000
    final_chunks = []
    
    for i in range(0, len(unique_users), chunk_size):
        chunk_u = unique_users[i:i+chunk_size]
        stacked_chunk = stacked.filter(pl.col("customer_id").is_in(chunk_u))
        arch_chunk = user_archetypes.filter(pl.col("customer_id").is_in(chunk_u))
        
        merged = stacked_chunk.join(arch_chunk.select(["customer_id", "archetype"]), on="customer_id", how="inner")
        
        weight_map = {
            "Habitual": {"A_history": 5.0, "B_local": 1.0, "C_svd": 0.5, "D_i2i": 0.5, "E_cat": 2.0, "F_brand": 3.0, "G_global": 0.1, "H_trend": 0.5},
            "Explorer": {"A_history": 1.0, "B_local": 1.0, "C_svd": 4.0, "D_i2i": 4.0, "E_cat": 3.0, "F_brand": 1.0, "G_global": 1.0, "H_trend": 2.0},
            "Dormant":  {"A_history": 3.0, "B_local": 4.0, "C_svd": 0.5, "D_i2i": 0.5, "E_cat": 1.0, "F_brand": 2.0, "G_global": 5.0, "H_trend": 1.0},
            "New":      {"A_history": 0.5, "B_local": 3.0, "C_svd": 2.5, "D_i2i": 2.5, "E_cat": 3.0, "F_brand": 1.0, "G_global": 4.0, "H_trend": 4.0},
            "Standard": {"A_history": 2.0, "B_local": 2.0, "C_svd": 1.5, "D_i2i": 1.5, "E_cat": 2.0, "F_brand": 2.0, "G_global": 1.0, "H_trend": 1.5},
        }
        
        w_df = pl.DataFrame([
            {"archetype": a, "channel": c, "ch_weight": w}
            for a, cw in weight_map.items() for c, w in cw.items()
        ])
        
        scored = (
            merged.join(w_df, on=["archetype", "channel"], how="left")
            .with_columns(pl.col("ch_weight").fill_null(1.0))
            .with_columns((pl.col("ch_weight") / (pl.col("rank") + 60)).alias("score"))
            .group_by(["customer_id", "item_id", "archetype"])
            .agg(pl.col("score").sum().alias("final_score"))
            .sort(["customer_id", "final_score"], descending=[False, True])
        )
        
        budget_map = {
            "Habitual": 250,
            "Explorer": 800,
            "Dormant": 120,
            "New": 350,
            "Standard": 450
        }
        b_df = pl.DataFrame([{"archetype": k, "budget": v} for k, v in budget_map.items()])
        
        final_chunk = (
            scored.with_columns(pl.int_range(1, pl.len() + 1).over("customer_id").cast(pl.Int64).alias("final_rank"))
            .join(b_df, on="archetype", how="left")
            .with_columns(pl.col("budget").fill_null(100))
            .filter(pl.col("final_rank") <= pl.col("budget"))
            .select(["customer_id", "item_id", "final_rank", "archetype"])
        )
        final_chunks.append(final_chunk)

    final = pl.concat(final_chunks)
    
    avg_cand = final.group_by("customer_id").len().select(pl.col("len").mean()).item()
    print(f"\nFinal Blended Candidates: {avg_cand:.1f} per user")
    for row in final.group_by("archetype").agg(pl.len().alias("total"), pl.col("customer_id").n_unique().alias("users")).with_columns((pl.col("total") / pl.col("users")).alias("avg")).iter_rows(named=True):
        print(f"  Archetype {row['archetype']}: {row['avg']:.1f} candidates/user")
    
    eval_df = final.join(truth, on=["customer_id", "item_id"], how="inner")
    
    total_users = truth["customer_id"].n_unique()
    users_with_hits = eval_df["customer_id"].n_unique()
    recall_user = users_with_hits / total_users
    
    total_truth_items = truth.height
    total_hits = eval_df.height
    recall_item = total_hits / total_truth_items
    
    print("=" * 40)
    print("RRF FINAL EVALUATION:")
    print(f"Users Evaluated: {total_users}")
    print(f"Retrieval Recall (User-level): {recall_user:.4f}")
    print(f"Retrieval Recall (Item-level): {recall_item:.4f}")
    print("=" * 40)
    
    return final

def run_pipeline(sample_n: int):
    print(f"--- Starting PIR Pipeline v2 (Sub-{sample_n}) ---")
    items = load_items()
    unique_users, target_users, truth = sample_active_december_users(sample_n=sample_n)
    hist_tx = load_history_data(target_users)
    
    archetypes = compute_user_archetypes(hist_tx, items, target_users)
    
    channels = {}
    channels["A_history"] = channel_history(hist_tx, target_users)
    channels["B_local"] = channel_local_popular(hist_tx, target_users, top_k=500)
    
    all_users = hist_tx.select("customer_id").unique()
    c_svd, c_i2i = channel_cf_latent(hist_tx, all_users, target_users, svd_k=600, i2i_k=600)
    channels["C_svd"] = c_svd
    channels["D_i2i"] = c_i2i
    
    channels["E_cat"] = channel_category_popular(hist_tx, items, target_users, top_categories_per_user=5, bestsellers_per_category=80)
    channels["F_brand"] = channel_brand_popular(hist_tx, items, target_users, top_brands_per_user=5, bestsellers_per_brand=50)
    channels["G_global"] = channel_global_popular(hist_tx, target_users, global_k=300)
    channels["H_trend"] = channel_category_trending(hist_tx, items, target_users, top_cats=3, trending=50)
    
    evaluate_and_fuse(channels, truth, archetypes, params={})

if __name__ == "__main__":
    MAX_CANDS = 150
    run_pipeline(max_cands=MAX_CANDS)
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=int, default=5000, help="Number of users for quick eval")
    args = parser.parse_args()
    
    run_pipeline(sample_n=args.subset)
