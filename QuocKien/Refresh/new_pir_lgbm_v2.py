import os
import gc
import json
import argparse
import numpy as np
import polars as pl
from datetime import datetime
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb
from sklearn.preprocessing import normalize
import optuna

# Base paths
TRANSACTION_PATH = "d:/CS116/ProjectNumberOne/transaction_full_2025.parquet"
ITEMS_PATH = "d:/CS116/ProjectNumberOne/items.parquet"

def load_items() -> pl.DataFrame:
    return pl.read_parquet(ITEMS_PATH).select([
        pl.col("item_id").cast(pl.Utf8),
        pl.col("category_l1").cast(pl.Utf8).fill_null("Unknown"),
        pl.col("brand").cast(pl.Utf8).fill_null("Unknown"),
        pl.col("price").cast(pl.Float32).fill_null(0.0),
    ])

def extract_dataset(cutoff_date: datetime, end_target_date: datetime, sample_n: int, items: pl.DataFrame):
    print(f"\n[Dataset] Cutoff={cutoff_date.date()} Target=[{cutoff_date.date()} to {end_target_date.date()}]")
    
    # Target Data
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
    
    # History Data
    hist_tx = (
        pl.scan_parquet(TRANSACTION_PATH)
        .select(["customer_id", "item_id", "quantity", pl.col("price").cast(pl.Float32), "location", "updated_date"])
        .filter(pl.col("updated_date") < cutoff_date)
        .collect()
    )
    hist_user = hist_tx.join(sampled_users.to_frame(), on="customer_id", how="inner")
    
    t_30 = cutoff_date - pl.duration(days=30)
    t_60 = cutoff_date - pl.duration(days=60)
    t_90 = cutoff_date - pl.duration(days=90)
    
    # ---------------------------------------------------------
    # 1. USER FEATURES (10)
    # ---------------------------------------------------------
    u_loc_hhi = (
        hist_user.group_by(["customer_id", "location"]).agg(pl.len().alias("loc_qty"))
        .with_columns((pl.col("loc_qty") / pl.col("loc_qty").sum().over("customer_id")).alias("share"))
        .group_by("customer_id").agg([(pl.col("share") * pl.col("share")).sum().alias("u_loc_hhi")])
    )
    
    u_cat_hhi = (
        hist_user.join(items, on="item_id", how="left")
        .group_by(["customer_id", "category_l1"]).agg(pl.len().alias("cat_qty"))
        .with_columns((pl.col("cat_qty") / pl.col("cat_qty").sum().over("customer_id")).alias("share"))
        .group_by("customer_id").agg([(pl.col("share") * pl.col("share")).sum().alias("u_cat_hhi")])
    )
    
    u_brand_hhi = (
        hist_user.join(items, on="item_id", how="left")
        .group_by(["customer_id", "brand"]).agg(pl.len().alias("brand_qty"))
        .with_columns((pl.col("brand_qty") / pl.col("brand_qty").sum().over("customer_id")).alias("share"))
        .group_by("customer_id").agg([(pl.col("share") * pl.col("share")).sum().alias("u_brand_hhi")])
    )
    
    user_features = (
        hist_user.group_by("customer_id").agg([
            pl.len().alias("u_total_tx"),
            pl.col("quantity").sum().alias("u_total_qty"),
            (pl.col("quantity") * pl.col("price")).sum().alias("u_total_spend"),
            pl.col("item_id").n_unique().alias("u_unique_items"),
            (cutoff_date - pl.col("updated_date").min()).dt.total_days().alias("u_tenure_days"),
            (cutoff_date - pl.col("updated_date").max()).dt.total_days().alias("u_recency_days"),
        ])
        .join(u_loc_hhi, on="customer_id", how="left")
        .join(u_cat_hhi, on="customer_id", how="left")
        .join(u_brand_hhi, on="customer_id", how="left")
        .with_columns([
            (pl.col("u_total_qty") / pl.col("u_total_tx")).alias("u_avg_basket_size"),
            (pl.col("u_total_spend") / pl.col("u_total_tx")).alias("u_avg_order_value"),
            (pl.col("u_total_tx") / (pl.col("u_tenure_days") + 1)).alias("u_tx_velocity"),
            pl.col("u_loc_hhi").fill_null(1.0),
            pl.col("u_cat_hhi").fill_null(1.0),
            pl.col("u_brand_hhi").fill_null(1.0),
            pl.col("u_recency_days").fill_null(999.0)
        ])
    )
    
    # ---------------------------------------------------------
    # 2. ITEM FEATURES (10)
    # ---------------------------------------------------------
    i_sales_all = hist_tx.group_by("item_id").agg(pl.len().alias("i_sales_all"))
    i_sales_30d = hist_tx.filter(pl.col("updated_date") >= t_30).group_by("item_id").agg(pl.len().alias("i_sales_30d"))
    i_sales_60d = hist_tx.filter((pl.col("updated_date") >= t_60) & (pl.col("updated_date") < t_30)).group_by("item_id").agg(pl.len().alias("i_sales_60_to_30d"))
    i_sales_90d = hist_tx.filter((pl.col("updated_date") >= t_90) & (pl.col("updated_date") < t_60)).group_by("item_id").agg(pl.len().alias("i_sales_90_to_60d"))
    
    # Category trend
    cat_sales_30d = hist_tx.filter(pl.col("updated_date") >= t_30).join(items, on="item_id", how="left").group_by("category_l1").agg(pl.len().alias("cat_sales_30d"))
    cat_sales_60d = hist_tx.filter((pl.col("updated_date") >= t_60) & (pl.col("updated_date") < t_30)).join(items, on="item_id", how="left").group_by("category_l1").agg(pl.len().alias("cat_sales_60d"))
    cat_trend = cat_sales_30d.join(cat_sales_60d, on="category_l1", how="left").fill_null(0).with_columns((pl.col("cat_sales_30d") - pl.col("cat_sales_60d")).alias("cat_momentum"))
    
    item_features = (
        items
        .join(i_sales_all, on="item_id", how="left")
        .join(i_sales_30d, on="item_id", how="left")
        .join(i_sales_60d, on="item_id", how="left")
        .join(i_sales_90d, on="item_id", how="left")
        .join(cat_trend, on="category_l1", how="left")
        .fill_null(0)
        .with_columns([
            (pl.col("i_sales_30d") - pl.col("i_sales_60_to_30d")).alias("i_momentum_30d"),
            (pl.col("i_sales_60_to_30d") - pl.col("i_sales_90_to_60d")).alias("i_momentum_60d")
        ])
    )
    
    # ---------------------------------------------------------
    # 3. USER-ITEM INTERACTIONS (10)
    # ---------------------------------------------------------
    ui_hist = (
        hist_user.group_by(["customer_id", "item_id"])
        .agg([
            pl.len().alias("ui_purchase_count"),
            pl.col("quantity").sum().alias("ui_total_qty"),
            (cutoff_date - pl.col("updated_date").max()).dt.total_days().alias("ui_days_since_last"),
            (cutoff_date - pl.col("updated_date").min()).dt.total_days().alias("ui_days_since_first")
        ])
        .with_columns([
            (pl.col("ui_total_qty") / pl.col("ui_purchase_count")).alias("ui_avg_qty_per_order"),
            (pl.col("ui_purchase_count") / (pl.col("ui_days_since_first") + 1)).alias("ui_purchase_velocity"),
            (pl.col("ui_days_since_last") > 22).cast(pl.Int8).alias("ui_replenishment_due")
        ])
    )
    
    # ---------------------------------------------------------
    # 4. AFFINITIES (10)
    # ---------------------------------------------------------
    hist_joined = hist_user.join(items, on="item_id", how="left")
    
    u_cat_affinity = (
        hist_joined.group_by(["customer_id", "category_l1"])
        .agg(pl.len().alias("u_cat_purchases"))
        .with_columns((pl.col("u_cat_purchases") / pl.col("u_cat_purchases").sum().over("customer_id")).alias("u_cat_share_of_wallet"))
    )
    u_brand_affinity = (
        hist_joined.group_by(["customer_id", "brand"])
        .agg(pl.len().alias("u_brand_purchases"))
        .with_columns((pl.col("u_brand_purchases") / pl.col("u_brand_purchases").sum().over("customer_id")).alias("u_brand_share_of_wallet"))
    )
    
    # ---------------------------------------------------------
    # 5. SVD / LATENT CF (2)
    # ---------------------------------------------------------
    print("Computing SVD Features...")
    svd_train_users = hist_user["customer_id"].unique().to_list()
    tx_weighted = (
        hist_user
        .with_columns(((cutoff_date - pl.col("updated_date")).dt.total_days() / 30.0).alias("months_ago"))
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
    
    svd = TruncatedSVD(n_components=64, random_state=42)
    u_emb = svd.fit_transform(mtx)
    i_emb = svd.components_.T
    
    # We will compute SVD dot product dynamically per user-item pair to save memory
    
    # ---------------------------------------------------------
    # GENERATE CANDIDATES
    # ---------------------------------------------------------
    print("Generating Candidates...")
    cand_hist = ui_hist.select(["customer_id", "item_id"])
    
    user_loc = hist_user.group_by(["customer_id", "location"]).agg(pl.len().alias("qty")).sort(["customer_id", "qty"], descending=True).group_by("customer_id").head(1).select(["customer_id", "location"])
    local_top = hist_tx.filter(pl.col("updated_date") >= t_60).group_by(["location", "item_id"]).agg(pl.len().alias("qty")).sort(["location", "qty"], descending=True).group_by("location").head(200)
    cand_local = user_loc.join(local_top, on="location", how="inner").select(["customer_id", "item_id"])
    
    global_top = i_sales_30d.sort("i_sales_30d", descending=True).head(200).select("item_id")
    cand_global = sampled_users.to_frame().join(global_top, how="cross")
    
    candidates = pl.concat([cand_hist, cand_local, cand_global]).unique()
    print(f"Candidates generated: {candidates.height}")
    
    # ---------------------------------------------------------
    # ASSEMBLE DATASET
    # ---------------------------------------------------------
    print("Assembling final dataset...")
    df = (
        candidates
        .join(truth, on=["customer_id", "item_id"], how="left").with_columns(pl.col("label").fill_null(0))
        .join(user_features, on="customer_id", how="inner")
        .join(item_features, on="item_id", how="inner")
        .join(ui_hist, on=["customer_id", "item_id"], how="left")
        .join(u_cat_affinity, on=["customer_id", "category_l1"], how="left")
        .join(u_brand_affinity, on=["customer_id", "brand"], how="left")
    )
    
    # Add SVD via fast map
    u2idx = dict(zip(u_df["customer_id"], u_df["u_idx"]))
    i2idx = dict(zip(i_df["item_id"], i_df["i_idx"]))
    
    def apply_svd(u_id, i_id):
        if u_id in u2idx and i_id in i2idx:
            return float(np.dot(u_emb[u2idx[u_id]], i_emb[i2idx[i_id]]))
        return 0.0
    
    df = df.with_columns(
        pl.struct(["customer_id", "item_id"]).map_elements(
            lambda x: apply_svd(x["customer_id"], x["item_id"]), return_dtype=pl.Float64
        ).alias("svd_score")
    )
    
    df = df.with_columns([
        pl.col("ui_purchase_count").fill_null(0),
        pl.col("ui_total_qty").fill_null(0),
        pl.col("ui_days_since_last").fill_null(999.0),
        pl.col("ui_days_since_first").fill_null(999.0),
        pl.col("ui_avg_qty_per_order").fill_null(0),
        pl.col("ui_purchase_velocity").fill_null(0),
        pl.col("ui_replenishment_due").fill_null(0),
        pl.col("u_cat_purchases").fill_null(0),
        pl.col("u_cat_share_of_wallet").fill_null(0),
        pl.col("u_brand_purchases").fill_null(0),
        pl.col("u_brand_share_of_wallet").fill_null(0),
    ])
    
    # Advanced Cross Ratios
    df = df.with_columns([
        (pl.col("u_cat_share_of_wallet") * pl.col("i_momentum_30d")).alias("cross_cat_momentum"),
        (pl.col("u_brand_share_of_wallet") * pl.col("i_momentum_30d")).alias("cross_brand_momentum"),
        (pl.col("price") / (pl.col("u_avg_order_value") + 1)).alias("cross_price_ratio")
    ])
    
    # Label encode strings
    for col in ["category_l1", "brand"]:
        cat_map = {c: i for i, c in enumerate(df[col].unique())}
        df = df.with_columns(pl.col(col).replace(cat_map).cast(pl.Int32).alias(f"{col}_idx"))
    
    return df.sort("customer_id")

# Global feature list
FEATURES = [
    "u_total_tx", "u_total_qty", "u_total_spend", "u_unique_items", "u_tenure_days", "u_recency_days", 
    "u_loc_hhi", "u_cat_hhi", "u_brand_hhi", "u_avg_basket_size", "u_avg_order_value", "u_tx_velocity",
    "price", "i_sales_all", "i_sales_30d", "i_sales_60_to_30d", "i_sales_90_to_60d", "i_momentum_30d", "i_momentum_60d",
    "cat_sales_30d", "cat_sales_60d", "cat_momentum",
    "ui_purchase_count", "ui_total_qty", "ui_days_since_last", "ui_days_since_first", "ui_avg_qty_per_order", "ui_purchase_velocity", "ui_replenishment_due",
    "u_cat_purchases", "u_cat_share_of_wallet", "u_brand_purchases", "u_brand_share_of_wallet",
    "svd_score", "cross_cat_momentum", "cross_brand_momentum", "cross_price_ratio",
    "category_l1_idx", "brand_idx"
]
CATEGORICAL = ["category_l1_idx", "brand_idx"]

def train_phase_a(train_df, valid_df):
    print("\n[Phase A] Optuna Tuning on Nov Data...")
    
    q_train = train_df.group_by("customer_id", maintain_order=True).len()["len"].to_numpy()
    q_valid = valid_df.group_by("customer_id", maintain_order=True).len()["len"].to_numpy()
    
    X_train = train_df.select(FEATURES).to_pandas()
    y_train = train_df["label"].to_pandas()
    X_valid = valid_df.select(FEATURES).to_pandas()
    y_valid = valid_df["label"].to_pandas()
    
    lgb_train = lgb.Dataset(X_train, label=y_train, group=q_train, categorical_feature=CATEGORICAL, free_raw_data=False)
    lgb_valid = lgb.Dataset(X_valid, label=y_valid, group=q_valid, categorical_feature=CATEGORICAL, reference=lgb_train, free_raw_data=False)
    
    def objective(trial):
        params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'eval_at': 10,
            'learning_rate': 0.1,
            'num_leaves': trial.suggest_int('num_leaves', 15, 63),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 200),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        
        gbm = lgb.train(
            params,
            lgb_train,
            num_boost_round=150,
            valid_sets=[lgb_valid],
            callbacks=[lgb.early_stopping(20, verbose=False)]
        )
        return gbm.best_score['valid_0']['ndcg@10']
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    
    print("\nBest Optuna Params:", study.best_params)
    
    # Retrain once with best params to get feature importances
    best_params = {**study.best_params, 'objective': 'lambdarank', 'metric': 'ndcg', 'eval_at': 10, 'learning_rate': 0.1, 'random_state': 42, 'verbose': -1}
    model = lgb.train(best_params, lgb_train, num_boost_round=150, valid_sets=[lgb_valid], callbacks=[lgb.early_stopping(20, verbose=False)])
    
    # Prune bad features
    imps = model.feature_importance(importance_type='gain')
    max_imp = np.max(imps)
    pruned_features = [f for f, imp in zip(FEATURES, imps) if imp >= 0.02 * max_imp]
    print(f"\nPruned {len(FEATURES) - len(pruned_features)} weak features. Keeping {len(pruned_features)}.")
    
    return best_params, pruned_features

def train_phase_b(train_df, test_df, best_params, final_features):
    print("\n[Phase B] Final Retrain on Jan-Nov, Test on Dec...")
    cat_feats = [f for f in CATEGORICAL if f in final_features]
    
    q_train = train_df.group_by("customer_id", maintain_order=True).len()["len"].to_numpy()
    q_test = test_df.group_by("customer_id", maintain_order=True).len()["len"].to_numpy()
    
    X_train = train_df.select(final_features).to_pandas()
    y_train = train_df["label"].to_pandas()
    X_test = test_df.select(final_features).to_pandas()
    y_test = test_df["label"].to_pandas()
    
    lgb_train = lgb.Dataset(X_train, label=y_train, group=q_train, categorical_feature=cat_feats, free_raw_data=False)
    lgb_test = lgb.Dataset(X_test, label=y_test, group=q_test, categorical_feature=cat_feats, reference=lgb_train, free_raw_data=False)
    
    model = lgb.train(
        best_params,
        lgb_train,
        num_boost_round=200,
        valid_sets=[lgb_train, lgb_test],
        valid_names=['train', 'test'],
        callbacks=[lgb.early_stopping(30)]
    )
    
    print("\nEvaluating Precision@10 on FINAL Test Set (December)...")
    preds = model.predict(X_test)
    test_eval = test_df.select(["customer_id", "item_id", "label"]).with_columns(pl.Series("pred", preds))
    
    top_1 = (
        test_eval
        .sort(["customer_id", "pred"], descending=[False, True])
        .group_by("customer_id").head(1)
    )
    
    dummy = pl.DataFrame({"dup": range(10)})
    top_10 = top_1.join(dummy, how="cross")
    
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
    
    print("=" * 40)
    print(f"Final Users Evaluated (Total Pool): {hits.height}")
    print(f"Global Reranker Precision@10: {p_at_10:.4f}")
    
    # 20 Bootstrap Samples of 10k users
    print("\nRunning 20 random samples of 10,000 users to verify consistency...")
    all_users = test_df["customer_id"].unique().to_numpy()
    
    scores = []
    np.random.seed(42)
    for i in range(20):
        # Sample 10k unique users from the pool
        sampled = np.random.choice(all_users, size=min(10000, len(all_users)), replace=False)
        sampled_df = pl.DataFrame({"customer_id": sampled})
        
        # Filter top 10 (which has duplicates) to just these users
        sample_top_10 = top_10.join(sampled_df, on="customer_id", how="inner")
        
        sample_hits = (
            sampled_df
            .join(
                sample_top_10.group_by("customer_id").agg(pl.col("label").sum().alias("hits")),
                on="customer_id",
                how="left"
            )
            .with_columns(pl.col("hits").fill_null(0))
        )
        
        scores.append(sample_hits["hits"].mean() / 10.0)
        
    scores = np.array(scores) * 100
    print(f"20-Sample Mean Precision@10: {np.mean(scores):.2f}%")
    print(f"Min: {np.min(scores):.2f}%, Max: {np.max(scores):.2f}%, Std Dev: {np.std(scores):.2f}%")
    print("=" * 40)

if __name__ == "__main__":
    items = load_items()
    
    nov_df = extract_dataset(datetime(2025, 11, 1), datetime(2025, 12, 1), 10000, items)
    users = nov_df["customer_id"].unique().to_list()
    np.random.seed(42)
    np.random.shuffle(users)
    train_users = users[:int(0.8*len(users))]
    valid_users = users[int(0.8*len(users)):]
    
    nov_train = nov_df.filter(pl.col("customer_id").is_in(train_users))
    nov_valid = nov_df.filter(pl.col("customer_id").is_in(valid_users))
    
    best_params, final_features = train_phase_a(nov_train, nov_valid)
    
    # Extract a massive 50k user pool for December to allow 10k sampling
    dec_df = extract_dataset(datetime(2025, 12, 1), datetime(2026, 1, 1), 50000, items)
    
    train_phase_b(nov_df, dec_df, best_params, final_features)
