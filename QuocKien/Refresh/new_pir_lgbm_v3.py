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
        pl.col("category_l2").cast(pl.Utf8).fill_null("Unknown"),
        pl.col("category_l3").cast(pl.Utf8).fill_null("Unknown"),
        pl.col("brand").cast(pl.Utf8).fill_null("Unknown"),
        pl.col("manufacturer").cast(pl.Utf8).fill_null("Unknown"),
        pl.col("size").cast(pl.Utf8).fill_null("Unknown"),
        pl.col("price").cast(pl.Float32).fill_null(0.0),
    ])

def extract_dataset(cutoff_date: datetime, end_target_date: datetime, sample_n: int, items: pl.DataFrame):
    print(f"\n[Dataset] Cutoff={cutoff_date.date()} Target=[{cutoff_date.date()} to {end_target_date.date()}]")
    
    # ---------------------------------------------------------
    # DATA LOADING & FILTERING
    # ---------------------------------------------------------
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
    
    hist_tx = (
        pl.scan_parquet(TRANSACTION_PATH)
        .select(["customer_id", "item_id", "quantity", pl.col("price").cast(pl.Float32), "discount", "bill_id", "location", "updated_date"])
        .filter(pl.col("updated_date") < cutoff_date)
        .collect()
    )
    hist_tx = hist_tx.with_columns(pl.col("discount").cast(pl.Float32).fill_null(0.0))
    hist_user = hist_tx.join(sampled_users.to_frame(), on="customer_id", how="inner")
    
    t_30 = cutoff_date - pl.duration(days=30)
    t_60 = cutoff_date - pl.duration(days=60)
    t_90 = cutoff_date - pl.duration(days=90)
    
    # Pre-merge items for categoricals
    hist_joined = hist_user.join(items, on="item_id", how="left")
    
    # ---------------------------------------------------------
    # 1. USER BEHAVIOR & HHI (12)
    # ---------------------------------------------------------
    print("Computing User Features...")
    def calc_hhi(df, col_name, prefix):
        counts = df.group_by(["customer_id", col_name]).agg(pl.len().alias("qty"))
        return (
            counts.with_columns((pl.col("qty") / pl.col("qty").sum().over("customer_id")).alias("share"))
            .group_by("customer_id").agg([(pl.col("share") * pl.col("share")).sum().alias(f"{prefix}_{col_name}_hhi")])
        )
    
    u_loc_hhi = calc_hhi(hist_joined, "location", "u")
    u_cat1_hhi = calc_hhi(hist_joined, "category_l1", "u")
    u_brand_hhi = calc_hhi(hist_joined, "brand", "u")
    u_cat2_hhi = calc_hhi(hist_joined, "category_l2", "u")
    
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
        .join(u_cat1_hhi, on="customer_id", how="left")
        .join(u_brand_hhi, on="customer_id", how="left")
        .join(u_cat2_hhi, on="customer_id", how="left")
        .with_columns([
            (pl.col("u_total_qty") / pl.col("u_total_tx")).alias("u_avg_basket_size"),
            (pl.col("u_total_spend") / pl.col("u_total_tx")).alias("u_avg_order_value"),
            (pl.col("u_total_tx") / (pl.col("u_tenure_days") + 1)).alias("u_tx_velocity"),
            pl.col("u_recency_days").fill_null(999.0)
        ])
    )
    
    # ---------------------------------------------------------
    # 2. ITEM & CATEGORY MOMENTUM (12)
    # ---------------------------------------------------------
    print("Computing Item Features...")
    i_sales_all = hist_tx.group_by("item_id").agg(pl.len().alias("i_sales_all"))
    i_sales_30d = hist_tx.filter(pl.col("updated_date") >= t_30).group_by("item_id").agg(pl.len().alias("i_sales_30d"))
    i_sales_60d = hist_tx.filter((pl.col("updated_date") >= t_60) & (pl.col("updated_date") < t_30)).group_by("item_id").agg(pl.len().alias("i_sales_60_to_30d"))
    
    item_features = (
        items
        .join(i_sales_all, on="item_id", how="left")
        .join(i_sales_30d, on="item_id", how="left")
        .join(i_sales_60d, on="item_id", how="left")
        .fill_null(0)
        .with_columns([
            (pl.col("i_sales_30d") - pl.col("i_sales_60_to_30d")).alias("i_momentum_30d")
        ])
    )
    
    # ---------------------------------------------------------
    # 3. DISCOUNT & PROMO (8)
    # ---------------------------------------------------------
    print("Computing Discount Features...")
    hist_joined = hist_joined.with_columns(
        (pl.col("discount") / (pl.col("price") + 1e-5)).alias("discount_rate")
    ).with_columns(
        (pl.col("discount_rate") > 0.05).cast(pl.Int8).alias("is_promo")
    )
    
    u_promo = hist_joined.group_by("customer_id").agg([
        pl.col("discount_rate").mean().alias("u_avg_discount_rate"),
        pl.col("is_promo").mean().alias("u_promo_purchase_ratio")
    ])
    
    i_promo = hist_joined.group_by("item_id").agg([
        pl.col("discount_rate").mean().alias("i_avg_discount_rate"),
        pl.col("is_promo").mean().alias("i_promo_sales_ratio")
    ])
    
    user_features = user_features.join(u_promo, on="customer_id", how="left").with_columns(pl.exclude("customer_id").fill_null(0.0))
    item_features = item_features.join(i_promo, on="item_id", how="left").with_columns(pl.exclude("item_id").fill_null(0.0))
    
    # ---------------------------------------------------------
    # 4. BASKET / SESSION (6)
    # ---------------------------------------------------------
    print("Computing Basket Features...")
    u_basket = (
        hist_joined.group_by(["customer_id", "bill_id"]).agg(pl.col("item_id").n_unique().alias("bill_items"))
        .group_by("customer_id").agg([
            pl.col("bill_items").mean().alias("u_avg_items_per_bill"),
            pl.col("bill_items").max().alias("u_max_items_per_bill")
        ])
    )
    
    i_basket = (
        hist_joined.group_by(["item_id", "bill_id"]).agg(pl.len().alias("qty"))
        .group_by("item_id").agg(pl.col("qty").mean().alias("i_avg_items_in_its_bills"))
    )
    
    user_features = user_features.join(u_basket, on="customer_id", how="left").with_columns(pl.exclude("customer_id").fill_null(1.0))
    item_features = item_features.join(i_basket, on="item_id", how="left").with_columns(pl.exclude("item_id").fill_null(1.0))
    
    # ---------------------------------------------------------
    # 5. TEMPORAL / SEASONALITY (10)
    # ---------------------------------------------------------
    print("Computing Temporal Features...")
    hist_joined = hist_joined.with_columns([
        pl.col("updated_date").dt.hour().alias("hour"),
        pl.col("updated_date").dt.weekday().alias("weekday") # 1=Mon, 7=Sun
    ]).with_columns(
        (pl.col("weekday") >= 6).cast(pl.Int8).alias("is_weekend")
    )
    
    u_time = hist_joined.group_by("customer_id").agg([
        pl.col("is_weekend").mean().alias("u_weekend_ratio"),
        pl.col("hour").mean().alias("u_avg_hour")
    ])
    i_time = hist_joined.group_by("item_id").agg([
        pl.col("is_weekend").mean().alias("i_weekend_ratio"),
        pl.col("hour").mean().alias("i_avg_hour")
    ])
    
    user_features = user_features.join(u_time, on="customer_id", how="left").with_columns(pl.exclude("customer_id").fill_null(0.0))
    item_features = item_features.join(i_time, on="item_id", how="left").with_columns(pl.exclude("item_id").fill_null(0.0))
    
    # ---------------------------------------------------------
    # 6. DIRECT UI INTERACTIONS & REPLENISHMENT (12)
    # ---------------------------------------------------------
    print("Computing Direct UI & Replenishment Features...")
    
    # Calculate exact median gap per item across entire history
    i_dates = hist_tx.select(["customer_id", "item_id", "updated_date"]).sort(["customer_id", "item_id", "updated_date"])
    i_gaps = i_dates.with_columns(
        (pl.col("updated_date") - pl.col("updated_date").shift(1).over(["customer_id", "item_id"])).dt.total_days().alias("gap")
    ).filter(pl.col("gap").is_not_null() & (pl.col("gap") > 1))
    
    item_median_gap = i_gaps.group_by("item_id").agg(pl.col("gap").median().alias("i_median_replenish_gap")).fill_null(999.0)
    item_features = item_features.join(item_median_gap, on="item_id", how="left")
    
    ui_hist = (
        hist_user.group_by(["customer_id", "item_id"])
        .agg([
            pl.len().alias("ui_purchase_count"),
            pl.col("quantity").sum().alias("ui_total_qty"),
            (cutoff_date - pl.col("updated_date").max()).dt.total_days().alias("ui_days_since_last"),
            (cutoff_date - pl.col("updated_date").min()).dt.total_days().alias("ui_days_since_first")
        ])
    )
    
    # ---------------------------------------------------------
    # 7. DEEP AFFINITIES (20)
    # ---------------------------------------------------------
    print("Computing Deep Affinities...")
    def calc_affinity(col):
        return (
            hist_joined.group_by(["customer_id", col])
            .agg(pl.len().alias(f"u_{col}_purchases"))
            .with_columns((pl.col(f"u_{col}_purchases") / pl.col(f"u_{col}_purchases").sum().over("customer_id")).alias(f"u_{col}_share"))
        )
    
    u_cat1_aff = calc_affinity("category_l1")
    u_cat2_aff = calc_affinity("category_l2")
    u_brand_aff = calc_affinity("brand")
    u_manuf_aff = calc_affinity("manufacturer")
    
    # ---------------------------------------------------------
    # 8. SVD / LATENT CF (2)
    # ---------------------------------------------------------
    print("Computing SVD Features...")
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
        .join(u_cat1_aff, on=["customer_id", "category_l1"], how="left")
        .join(u_cat2_aff, on=["customer_id", "category_l2"], how="left")
        .join(u_brand_aff, on=["customer_id", "brand"], how="left")
        .join(u_manuf_aff, on=["customer_id", "manufacturer"], how="left")
    )
    
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
        pl.col(c).fill_null(0) for c in df.columns if c not in ["label", "customer_id", "item_id"]
    ])
    
    # Advanced Cross Ratios & Combinations
    df = df.with_columns([
        (pl.col("u_category_l1_share") * pl.col("i_momentum_30d")).alias("cross_cat_momentum"),
        (pl.col("u_promo_purchase_ratio") * pl.col("i_promo_sales_ratio")).alias("cross_promo_affinity"),
        (pl.col("u_avg_items_per_bill") - pl.col("i_avg_items_in_its_bills")).abs().alias("basket_size_mismatch"),
        (pl.col("u_weekend_ratio") * pl.col("i_weekend_ratio")).alias("weekend_shopper_match"),
        (pl.col("ui_days_since_last") - pl.col("i_median_replenish_gap")).alias("replenishment_overdue_days")
    ])
    
    # Label encode strings (Categorical features for LGBM)
    cat_cols = ["category_l1", "category_l2", "category_l3", "brand", "manufacturer", "size"]
    for col in cat_cols:
        cat_map = {c: i for i, c in enumerate(df[col].unique())}
        df = df.with_columns(pl.col(col).replace(cat_map).cast(pl.Int32).alias(f"{col}_idx"))
    
    return df.sort("customer_id")

# Global feature extraction list
def extract_feature_names(df):
    excluded = ["customer_id", "item_id", "label", "category_l1", "category_l2", "category_l3", "brand", "manufacturer", "size"]
    return [c for c in df.columns if c not in excluded]

CATEGORICAL = ["category_l1_idx", "category_l2_idx", "category_l3_idx", "brand_idx", "manufacturer_idx", "size_idx"]

def train_phase_a(train_df, valid_df):
    print("\n[Phase A] Optuna Tuning on Nov Data...")
    FEATURES = extract_feature_names(train_df)
    print(f"Total Features fed to LGBM: {len(FEATURES)}")
    
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
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 255),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 500),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.3, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.3, 1.0),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'bagging_freq': 1,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1,
            'feature_pre_filter': False
        }
        gbm = lgb.train(params, lgb_train, num_boost_round=1000, valid_sets=[lgb_valid], callbacks=[lgb.early_stopping(50, verbose=False)])
        return gbm.best_score['valid_0']['ndcg@10']
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)
    print("\nBest Optuna Params:", study.best_params)
    
    best_params = {**study.best_params, 'objective': 'lambdarank', 'metric': 'ndcg', 'eval_at': 10, 'bagging_freq': 1, 'random_state': 42, 'verbose': -1, 'feature_pre_filter': False}
    model = lgb.train(best_params, lgb_train, num_boost_round=1000, valid_sets=[lgb_valid], callbacks=[lgb.early_stopping(50, verbose=False)])
    
    imps = model.feature_importance(importance_type='gain')
    max_imp = np.max(imps)
    pruned_features = [f for f, imp in zip(FEATURES, imps) if imp >= 0.005 * max_imp]
    
    print(f"\nPruned {len(FEATURES) - len(pruned_features)} weak features. Keeping {len(pruned_features)}.")
    return best_params, pruned_features

def train_phase_b(train_df, test_df, best_params, final_features):
    print("\n[Phase B] Final Retrain on Jan-Nov, Test on Dec...")
    cat_feats = [f for f in CATEGORICAL if f in final_features]
    
    q_train = train_df.group_by("customer_id", maintain_order=True).len()["len"].to_numpy()
    X_train = train_df.select(final_features).to_pandas()
    y_train = train_df["label"].to_pandas()
    
    X_test = test_df.select(final_features).to_pandas()
    
    lgb_train = lgb.Dataset(X_train, label=y_train, group=q_train, categorical_feature=cat_feats, free_raw_data=False)
    
    model = lgb.train(
        best_params,
        lgb_train,
        num_boost_round=1000,
        valid_sets=[lgb_train],
        valid_names=['train'],
        callbacks=[lgb.early_stopping(50)]
    )
    
    print("\nEvaluating Precision@10 on FINAL Test Set (December)...")
    preds = model.predict(X_test)
    test_eval = test_df.select(["customer_id", "item_id", "label"]).with_columns(pl.Series("pred", preds))
    
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
    
    print("=" * 40)
    print(f"Final Users Evaluated: {hits.height}")
    print(f"Final Reranker Precision@10: {p_at_10:.4f}")
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
    
    dec_df = extract_dataset(datetime(2025, 12, 1), datetime(2026, 1, 1), 5000, items)
    train_phase_b(nov_df, dec_df, best_params, final_features)
