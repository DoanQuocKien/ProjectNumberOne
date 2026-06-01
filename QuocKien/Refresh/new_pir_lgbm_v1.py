import os
import gc
import argparse
import numpy as np
import polars as pl
from datetime import datetime
from pathlib import Path
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb
from sklearn.preprocessing import normalize

# Base paths
ROOT = Path("d:/CS116/ProjectNumberOne")
TRANSACTION_PATH = ROOT / "transaction_full_2025.parquet"
ITEMS_PATH = ROOT / "items.parquet"

def load_items() -> pl.DataFrame:
    print("Loading items...")
    return pl.read_parquet(ITEMS_PATH).select([
        pl.col("item_id").cast(pl.Utf8),
        pl.col("category_l1").cast(pl.Utf8).fill_null("Unknown"),
        pl.col("brand").cast(pl.Utf8).fill_null("Unknown"),
        pl.col("price").cast(pl.Float32).fill_null(0.0),
    ])

def extract_features_and_candidates(cutoff_date: datetime, end_target_date: datetime, sample_n: int, items: pl.DataFrame):
    print(f"\n--- Extracting dataset: Cutoff={cutoff_date.date()} Target=[{cutoff_date.date()} to {end_target_date.date()}] ---")
    
    # 1. Target Data
    target_tx = (
        pl.scan_parquet(TRANSACTION_PATH)
        .select(["customer_id", "item_id", "updated_date"])
        .filter((pl.col("updated_date") >= cutoff_date) & (pl.col("updated_date") < end_target_date))
        .collect()
    )
    
    unique_target_users = target_tx["customer_id"].unique()
    sampled_users = unique_target_users.sample(n=min(sample_n, len(unique_target_users)), seed=42)
    target_tx = target_tx.join(sampled_users.to_frame(), on="customer_id", how="inner")
    
    truth = target_tx.select(["customer_id", "item_id"]).unique().with_columns(pl.lit(1).cast(pl.Int8).alias("label"))
    
    # 2. History Data
    hist_tx = (
        pl.scan_parquet(TRANSACTION_PATH)
        .select(["customer_id", "item_id", "quantity", "location", "updated_date"])
        .filter(pl.col("updated_date") < cutoff_date)
        .collect()
    )
    # Filter to only the sampled users to save memory during feature engineering
    hist_user = hist_tx.join(sampled_users.to_frame(), on="customer_id", how="inner")
    
    # --------------------------
    # FEATURE ENGINEERING
    # --------------------------
    print("Building User Features...")
    u_loc_hhi = (
        hist_user.group_by(["customer_id", "location"]).agg(pl.len().alias("loc_qty"))
        .with_columns((pl.col("loc_qty") / pl.col("loc_qty").sum().over("customer_id")).alias("share"))
        .group_by("customer_id").agg([(pl.col("share") * pl.col("share")).sum().alias("u_loc_hhi")])
    )
    
    u_cat_hhi = (
        hist_user.join(items, on="item_id", how="left")
        .group_by(["customer_id", "category_l1"]).agg(pl.len().alias("cat_qty"))
        .with_columns((pl.col("cat_qty") / pl.col("cat_qty").sum().over("customer_id")).alias("share"))
        .group_by("customer_id").agg([
            (pl.col("share") * pl.col("share")).sum().alias("u_cat_hhi"),
            pl.col("category_l1").n_unique().alias("u_unique_cats")
        ])
    )
    
    user_features = (
        hist_user.group_by("customer_id").agg([
            pl.len().alias("u_total_tx"),
            pl.col("item_id").n_unique().alias("u_unique_items"),
            (cutoff_date - pl.col("updated_date").min()).dt.total_days().alias("u_tenure_days"),
            (cutoff_date - pl.col("updated_date").max()).dt.total_days().alias("u_recency_days"),
        ])
        .join(u_loc_hhi, on="customer_id", how="left")
        .join(u_cat_hhi, on="customer_id", how="left")
        .with_columns([
            pl.col("u_loc_hhi").fill_null(1.0),
            pl.col("u_cat_hhi").fill_null(1.0),
            pl.col("u_unique_cats").fill_null(1),
            pl.col("u_recency_days").fill_null(999.0)
        ])
    )
    
    print("Building Item Features...")
    t_30 = cutoff_date - pl.duration(days=30)
    t_60 = cutoff_date - pl.duration(days=60)
    
    item_sales_total = hist_tx.group_by("item_id").agg(pl.len().alias("i_total_sales"))
    item_sales_30d = hist_tx.filter(pl.col("updated_date") >= t_30).group_by("item_id").agg(pl.len().alias("i_sales_30d"))
    item_sales_60d = hist_tx.filter((pl.col("updated_date") >= t_60) & (pl.col("updated_date") < t_30)).group_by("item_id").agg(pl.len().alias("i_sales_60_to_30d"))
    
    item_features = (
        items
        .join(item_sales_total, on="item_id", how="left")
        .join(item_sales_30d, on="item_id", how="left")
        .join(item_sales_60d, on="item_id", how="left")
        .fill_null(0)
        .with_columns((pl.col("i_sales_30d") - pl.col("i_sales_60_to_30d")).alias("i_momentum_30d"))
    )
    
    # Label Encode Categoricals
    cat_mapping = {cat: i for i, cat in enumerate(item_features["category_l1"].unique())}
    brand_mapping = {b: i for i, b in enumerate(item_features["brand"].unique())}
    item_features = item_features.with_columns([
        pl.col("category_l1").replace(cat_mapping).cast(pl.Int32).alias("i_category_l1_idx"),
        pl.col("brand").replace(brand_mapping).cast(pl.Int32).alias("i_brand_idx")
    ])
    
    print("Building User-Item Interactions...")
    user_item_hist = (
        hist_user.group_by(["customer_id", "item_id"])
        .agg([
            pl.len().alias("ui_purchase_count"),
            (cutoff_date - pl.col("updated_date").max()).dt.total_days().alias("ui_days_since_last")
        ])
    )
    
    print("Generating Candidates...")
    # Fast Candidate Generation (History, Local Top, Global Top, CF)
    # 1. History
    cand_hist = user_item_hist.select(["customer_id", "item_id"])
    
    # 2. Local Top 100
    user_loc = hist_user.group_by(["customer_id", "location"]).agg(pl.len().alias("qty")).sort(["customer_id", "qty"], descending=True).group_by("customer_id").head(1).select(["customer_id", "location"])
    local_top = hist_tx.filter(pl.col("updated_date") >= t_60).group_by(["location", "item_id"]).agg(pl.len().alias("qty")).sort(["location", "qty"], descending=True).group_by("location").head(100)
    cand_local = user_loc.join(local_top, on="location", how="inner").select(["customer_id", "item_id"])
    
    # 3. Global Top 100
    global_top = item_sales_30d.sort("i_sales_30d", descending=True).head(100).select("item_id")
    cand_global = sampled_users.to_frame().join(global_top, how="cross")
    
    # Combine candidates
    candidates = pl.concat([cand_hist, cand_local, cand_global]).unique()
    print(f"Total Unique Candidates: {candidates.height} (avg {candidates.height/sample_n:.1f} per user)")
    
    print("Assembling final dataset...")
    df = (
        candidates
        .join(truth, on=["customer_id", "item_id"], how="left").with_columns(pl.col("label").fill_null(0))
        .join(user_features, on="customer_id", how="inner")
        .join(item_features, on="item_id", how="inner")
        .join(user_item_hist, on=["customer_id", "item_id"], how="left")
        .with_columns([
            pl.col("ui_purchase_count").fill_null(0),
            pl.col("ui_days_since_last").fill_null(999.0)
        ])
        .sort(["customer_id"])
    )
    
    return df

def train_and_evaluate():
    items = load_items()
    
    # Train Split: History up to Nov 1. Target: Nov.
    train_df = extract_features_and_candidates(datetime(2025, 11, 1), datetime(2025, 12, 1), 10000, items)
    
    # Test Split: History up to Dec 1. Target: Dec.
    test_df = extract_features_and_candidates(datetime(2025, 12, 1), datetime(2026, 1, 1), 5000, items)
    
    features = [
        "u_total_tx", "u_unique_items", "u_tenure_days", "u_recency_days", "u_loc_hhi", "u_cat_hhi", "u_unique_cats",
        "price", "i_category_l1_idx", "i_brand_idx", "i_total_sales", "i_sales_30d", "i_sales_60_to_30d", "i_momentum_30d",
        "ui_purchase_count", "ui_days_since_last"
    ]
    
    categorical_features = ["i_category_l1_idx", "i_brand_idx"]
    
    print("\nPreparing LightGBM Datasets...")
    # Group counts
    q_train = train_df.group_by("customer_id", maintain_order=True).len()["len"].to_numpy()
    q_test = test_df.group_by("customer_id", maintain_order=True).len()["len"].to_numpy()
    
    X_train = train_df.select(features).to_pandas()
    y_train = train_df["label"].to_pandas()
    
    X_test = test_df.select(features).to_pandas()
    y_test = test_df["label"].to_pandas()
    
    lgb_train = lgb.Dataset(X_train, label=y_train, group=q_train, categorical_feature=categorical_features, free_raw_data=False)
    lgb_test = lgb.Dataset(X_test, label=y_test, group=q_test, categorical_feature=categorical_features, reference=lgb_train, free_raw_data=False)
    
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'eval_at': 10,
        'learning_rate': 0.1,
        'num_leaves': 31,
        'min_data_in_leaf': 50,
        'feature_fraction': 0.8,
        'random_state': 42,
        'n_jobs': -1
    }
    
    print("Training Model...")
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=300,
        valid_sets=[lgb_train, lgb_test],
        valid_names=['train', 'valid'],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(10)]
    )
    
    print("\nFeature Importances:")
    importances = model.feature_importance(importance_type='gain')
    feat_imp = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
    for feat, imp in feat_imp:
        print(f"  {feat}: {imp:.2f}")
    
    print("\nEvaluating Precision@10 on Test Set...")
    preds = model.predict(X_test)
    test_eval = test_df.select(["customer_id", "item_id", "label"]).with_columns(pl.Series("pred", preds))
    
    # Sort and take top 10
    top_10 = (
        test_eval
        .sort(["customer_id", "pred"], descending=[False, True])
        .group_by("customer_id").head(10)
    )
    
    all_customers = test_df.select("customer_id").unique()
    
    hits = (
        all_customers
        .join(
            top_10.group_by("customer_id").agg(pl.col("label").sum().alias("hits")),
            on="customer_id",
            how="left"
        )
        .with_columns(pl.col("hits").fill_null(0))
    )
    p_at_10 = hits["hits"].mean() / 10.0
    
    total_users = hits.height
    print("=" * 40)
    print(f"Users Evaluated: {total_users}")
    print(f"Reranker Precision@10: {p_at_10:.4f}")
    print(f"Avg hits per user in Top 10: {hits['hits'].mean():.2f}")
    print("=" * 40)

if __name__ == "__main__":
    train_and_evaluate()
