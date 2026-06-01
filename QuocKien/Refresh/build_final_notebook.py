import nbformat as nbf

nb = nbf.v4.new_notebook()

cell_imports = """
import polars as pl
import numpy as np
import lightgbm as lgb
import pickle
from datetime import datetime
from scipy.sparse.linalg import svds
from scipy.sparse import coo_matrix

# ==============================================================
# 1. FILE PATHS (REPLACE THESE VARIABLES ON KAGGLE)
# ==============================================================
TRANSACTION_PATH = "transaction_full_2025.parquet"
ITEM_PATH = "items.parquet"
OUTPUT_SUBMISSION_PATH = "submission.pkl"

# Parameters
CHUNK_SIZE = 50000  # Number of users per inference chunk to prevent OOM
"""

cell_helpers = """
# ==============================================================
# 2. HELPER FUNCTIONS
# ==============================================================

def load_items():
    return pl.scan_parquet(ITEM_PATH).select(["item_id", "category_l1", "category_l2", "sale_status"]).collect()

def calc_hhi(df, col_name, prefix):
    counts = df.group_by(["customer_id", col_name]).agg(pl.len().alias("qty"))
    return (
        counts.with_columns((pl.col("qty") / pl.col("qty").sum().over("customer_id")).alias("share"))
        .group_by("customer_id").agg([(pl.col("share") * pl.col("share")).sum().alias(f"{prefix}_{col_name}_hhi")])
        .with_columns(pl.col("customer_id").cast(pl.Int64))
    )

def build_precomputed_tables(history_tx, items):
    \"\"\"
    Precomputes all expensive tables (SVD, user_features, item_features, ui_hist, global_top)
    ONCE from the full history. Used for both training dataset extraction and inference.
    \"\"\"
    print("  [Precompute] Joining items metadata...")
    hist_joined = history_tx.join(items, on="item_id", how="left")

    print("  [Precompute] Computing SVD embeddings...")
    user_cat_matrix = (
        hist_joined.group_by(["customer_id", "category_l2"])
        .agg(pl.len().alias("purchases"))
        .to_pandas()
    )
    unique_users = user_cat_matrix["customer_id"].unique()
    unique_cats  = user_cat_matrix["category_l2"].dropna().unique()
    user_map = {u: i for i, u in enumerate(unique_users)}
    cat_map  = {c: i for i, c in enumerate(unique_cats)}

    valid_mask = user_cat_matrix["category_l2"].notna()
    uc_valid = user_cat_matrix[valid_mask]
    row  = uc_valid["customer_id"].map(user_map).values
    col  = uc_valid["category_l2"].map(cat_map).values
    data = uc_valid["purchases"].values
    mat  = coo_matrix((data, (row, col)), shape=(len(unique_users), len(unique_cats))).astype(float)
    k    = min(10, min(mat.shape) - 1)

    if k > 0:
        U, S, Vt = svds(mat, k=k)
        U_df = pl.DataFrame(U, schema=[f"u_svd_{i}" for i in range(k)]).with_columns(
            pl.Series("customer_id", unique_users, dtype=pl.Int64))
        V_df = pl.DataFrame(Vt.T, schema=[f"c_svd_{i}" for i in range(k)]).with_columns(
            pl.Series("category_l2", unique_cats))
    else:
        U_df = pl.DataFrame({"customer_id": pl.Series(unique_users, dtype=pl.Int64)})
        V_df = pl.DataFrame({"category_l2": pl.Series(unique_cats)})

    print("  [Precompute] Computing user features...")
    user_features = (
        history_tx.group_by("customer_id").agg([
            pl.len().alias("u_total_purchases"),
            pl.col("item_id").n_unique().alias("u_unique_items"),
            pl.col("price").mean().alias("u_avg_price"),
        ])
        .with_columns(pl.col("customer_id").cast(pl.Int64))
        .join(calc_hhi(hist_joined, "category_l1", "u"), on="customer_id", how="left")
        .join(U_df, on="customer_id", how="left")
        .with_columns(pl.exclude("customer_id").fill_null(0.0))
    )

    print("  [Precompute] Computing item features...")
    item_features = (
        history_tx.group_by("item_id").agg([
            pl.len().alias("i_total_sales"),
            pl.col("customer_id").n_unique().alias("i_unique_buyers"),
        ])
        .with_columns(pl.col("item_id").cast(pl.Utf8))
        .join(items.select(["item_id", "category_l2"]).with_columns(pl.col("item_id").cast(pl.Utf8)), on="item_id", how="left")
        .join(V_df, on="category_l2", how="left")
        .drop("category_l2")
        .with_columns(pl.exclude("item_id").fill_null(0.0))
    )

    print("  [Precompute] Computing ui_hist & global_top...")
    ui_hist = (
        history_tx.group_by(["customer_id", "item_id"]).agg([
            pl.len().alias("ui_purchases"),
            (pl.col("updated_date").max() - pl.col("updated_date").min()).dt.total_days().alias("ui_duration"),
        ])
        .with_columns([
            pl.col("customer_id").cast(pl.Int64),
            pl.col("item_id").cast(pl.Utf8),
        ])
    )

    global_top = (
        history_tx.group_by("item_id").len()
        .sort("len", descending=True).head(100)
        .select(pl.col("item_id").cast(pl.Utf8))
    )

    return user_features, item_features, ui_hist, global_top


def extract_training_dataset(history_tx, target_tx, sample_n, items,
                              user_features, item_features, ui_hist, global_top):
    \"\"\"Builds a labeled training DataFrame using precomputed tables.\"\"\"
    active_users = target_tx["customer_id"].unique()
    if sample_n and sample_n < len(active_users):
        active_users = active_users.sample(n=sample_n, seed=42)
    active_users = active_users.cast(pl.Int64)

    cand_hist = (
        ui_hist.filter(pl.col("customer_id").is_in(active_users))
        .select([pl.col("customer_id"), pl.col("item_id")])
    )
    cand_global = active_users.to_frame().join(global_top, how="cross")
    candidates = pl.concat([cand_hist, cand_global]).unique()

    truth = (
        target_tx
        .filter(pl.col("customer_id").is_in(active_users))
        .with_columns([pl.col("customer_id").cast(pl.Int64), pl.col("item_id").cast(pl.Utf8), pl.lit(1).alias("label")])
        .select(["customer_id", "item_id", "label"]).unique()
    )

    df = (
        candidates
        .join(truth, on=["customer_id", "item_id"], how="left")
        .with_columns(pl.col("label").fill_null(0))
        .join(user_features, on="customer_id", how="inner")
        .join(item_features, on="item_id", how="inner")
        .join(ui_hist.select(["customer_id", "item_id", "ui_purchases", "ui_duration"]),
              on=["customer_id", "item_id"], how="left")
        .with_columns([pl.col("ui_purchases").fill_null(0), pl.col("ui_duration").fill_null(0)])
    )
    return df


def infer_chunk(chunk_users, ui_hist, global_top, user_features, item_features, final_model, FINAL_FEATURES):
    \"\"\"Fast inference for a single chunk: only filter + join + predict, no heavy aggregation.\"\"\"
    chunk_series = pl.Series("customer_id", chunk_users, dtype=pl.Int64)

    cand_hist   = ui_hist.filter(pl.col("customer_id").is_in(chunk_series)).select(["customer_id", "item_id"])
    cand_global = chunk_series.to_frame().join(global_top, how="cross")
    candidates  = pl.concat([cand_hist, cand_global]).unique()

    df = (
        candidates
        .join(user_features, on="customer_id", how="inner")
        .join(item_features, on="item_id", how="inner")
        .join(ui_hist.select(["customer_id", "item_id", "ui_purchases", "ui_duration"]),
              on=["customer_id", "item_id"], how="left")
        .with_columns([pl.col("ui_purchases").fill_null(0), pl.col("ui_duration").fill_null(0)])
    )

    preds  = final_model.predict(df[FINAL_FEATURES].to_numpy())
    res_df = df.select(["customer_id", "item_id"]).with_columns(pl.Series("pred", preds))

    top_1 = (
        res_df.sort(["customer_id", "pred"], descending=[False, True])
        .group_by("customer_id").head(1)
    )

    return {
        int(row["customer_id"]): tuple([str(row["item_id"])] * 10)
        for row in top_1.iter_rows(named=True)
    }


def get_data_splits():
    print("Loading full transaction history...")
    tx = pl.scan_parquet(TRANSACTION_PATH).collect()

    hist_nov = tx.filter(pl.col("updated_date") < datetime(2025, 11, 1))
    targ_nov = tx.filter((pl.col("updated_date") >= datetime(2025, 11, 1)) & (pl.col("updated_date") < datetime(2025, 12, 1)))

    hist_dec = tx.filter(pl.col("updated_date") < datetime(2025, 12, 1))
    targ_dec = tx.filter((pl.col("updated_date") >= datetime(2025, 12, 1)) & (pl.col("updated_date") < datetime(2026, 1, 1)))

    hist_jan     = tx
    all_2025_users = tx["customer_id"].unique().cast(pl.Int64).to_list()

    return hist_nov, targ_nov, hist_dec, targ_dec, hist_jan, all_2025_users
"""

cell_phase_a = """
# ==============================================================
# 3. PHASE A: FEATURE PRUNING (TARGET = NOV)
# ==============================================================
items = load_items()
hist_nov, targ_nov, hist_dec, targ_dec, hist_jan, all_2025_users = get_data_splits()

print("Precomputing tables for Nov history...")
uf_nov, if_nov, ui_nov, gt_nov = build_precomputed_tables(hist_nov, items)

print("Extracting Nov training dataset (20,000 users)...")
nov_df = extract_training_dataset(hist_nov, targ_nov, sample_n=20000, items=items,
                                   user_features=uf_nov, item_features=if_nov,
                                   ui_hist=ui_nov, global_top=gt_nov)

users_nov = nov_df["customer_id"].unique().to_list()
np.random.seed(42); np.random.shuffle(users_nov)
train_users = users_nov[:int(0.8 * len(users_nov))]
valid_users = users_nov[int(0.8 * len(users_nov)):]

nov_train = nov_df.filter(pl.col("customer_id").is_in(train_users)).sort("customer_id")
nov_valid = nov_df.filter(pl.col("customer_id").is_in(valid_users)).sort("customer_id")

FEATURES = [c for c in nov_train.columns if c not in ["customer_id", "item_id", "label"]]

q_train = nov_train.group_by("customer_id", maintain_order=True).len()["len"].to_numpy()
q_valid = nov_valid.group_by("customer_id", maintain_order=True).len()["len"].to_numpy()

lgb_train = lgb.Dataset(nov_train[FEATURES].to_numpy(), label=nov_train["label"].to_numpy(), group=q_train)
lgb_valid = lgb.Dataset(nov_valid[FEATURES].to_numpy(), label=nov_valid["label"].to_numpy(), group=q_valid, reference=lgb_train)

# Hardcoded optimal params from local 30-trial Optuna run (NDCG@10: 0.5618)
best_params = {
    'objective': 'lambdarank', 'metric': 'ndcg', 'eval_at': 10,
    'learning_rate': 0.1, 'num_leaves': 24, 'min_data_in_leaf': 193,
    'feature_fraction': 0.8178, 'bagging_fraction': 0.9, 'bagging_freq': 1,
    'random_state': 42, 'n_jobs': -1, 'verbose': -1, 'feature_pre_filter': False
}

print("Training Phase A model for feature importance pruning...")
model_a = lgb.train(best_params, lgb_train, num_boost_round=150,
                    valid_sets=[lgb_valid], callbacks=[lgb.early_stopping(20, verbose=False)])

imps = model_a.feature_importance(importance_type='gain')
max_imp = np.max(imps)
FINAL_FEATURES = [f for f, imp in zip(FEATURES, imps) if imp >= 0.01 * max_imp]
print(f"Pruned {len(FEATURES) - len(FINAL_FEATURES)} weak features. Keeping {len(FINAL_FEATURES)}.")
"""

cell_phase_b = """
# ==============================================================
# 4. PHASE B: FINAL TRAINING (TARGET = DEC)
# ==============================================================
print("Precomputing tables for Dec history...")
uf_dec, if_dec, ui_dec, gt_dec = build_precomputed_tables(hist_dec, items)

print("Extracting Dec training dataset (100,000 users)...")
dec_df = extract_training_dataset(hist_dec, targ_dec, sample_n=100000, items=items,
                                   user_features=uf_dec, item_features=if_dec,
                                   ui_hist=ui_dec, global_top=gt_dec)

dec_train = dec_df.sort("customer_id")
q_dec = dec_train.group_by("customer_id", maintain_order=True).len()["len"].to_numpy()
final_lgb_train = lgb.Dataset(dec_train[FINAL_FEATURES].to_numpy(), label=dec_train["label"].to_numpy(), group=q_dec)

print("Training Final Reranker on December data...")
final_model = lgb.train(
    best_params, final_lgb_train, num_boost_round=150,
    valid_sets=[final_lgb_train], valid_names=['train'],
    callbacks=[lgb.early_stopping(30)]
)
print("Final Model Ready!")
"""

cell_inference = """
# ==============================================================
# 5. INFERENCE: PRECOMPUTE ONCE, THEN CHUNK (FAST)
# ==============================================================
print("Precomputing inference tables from full 2025 history (done ONCE)...")
uf_jan, if_jan, ui_jan, gt_jan = build_precomputed_tables(hist_jan, items)

print(f"Total users to predict: {len(all_2025_users)}")
final_submission = {}

for i in range(0, len(all_2025_users), CHUNK_SIZE):
    chunk_users = all_2025_users[i : i + CHUNK_SIZE]
    chunk_num   = i // CHUNK_SIZE + 1
    total_chunks = (len(all_2025_users) + CHUNK_SIZE - 1) // CHUNK_SIZE
    print(f"Chunk {chunk_num}/{total_chunks} ({len(chunk_users)} users)...", end=" ")

    chunk_result = infer_chunk(chunk_users, ui_jan, gt_jan, uf_jan, if_jan, final_model, FINAL_FEATURES)
    final_submission.update(chunk_result)
    print(f"done. Total: {len(final_submission)}")
"""

cell_export = """
# ==============================================================
# 6. EXPORT TO PICKLE
# ==============================================================
print(f"Exporting {len(final_submission)} users to {OUTPUT_SUBMISSION_PATH}...")
with open(OUTPUT_SUBMISSION_PATH, "wb") as f:
    pickle.dump(final_submission, f)
print("SUCCESS! Upload submission.pkl to Kaggle.")
"""

nb['cells'] = [
    nbf.v4.new_code_cell(cell_imports),
    nbf.v4.new_code_cell(cell_helpers),
    nbf.v4.new_code_cell(cell_phase_a),
    nbf.v4.new_code_cell(cell_phase_b),
    nbf.v4.new_code_cell(cell_inference),
    nbf.v4.new_code_cell(cell_export),
]

with open("Final_PIR_Submission.ipynb", 'w') as f:
    nbf.write(nb, f)
print("Notebook generated successfully!")
