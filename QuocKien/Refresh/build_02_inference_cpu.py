import nbformat as nbf
import os

def run():
    with open("local_ablation_test.py", "r", encoding="utf-8") as f:
        local_code = f.read()

    start_idx = local_code.find("def standardize_age(text):")
    end_idx = local_code.find("def run_ablation", start_idx)
    helpers_code = local_code[start_idx:end_idx]

    flags_block = """
# ==============================================================
# HARDCODED FLAGS FOR CPU INFERENCE
# ==============================================================
# Must exactly match the flags used during GPU Training!
flags = {
    "filter_dead": False,
    "use_events": False,
    "copurchase": True,
    "rich_metadata": False,
    "discount_features": False
}
"""

    streaming_code = """
def infer_chunk(chunk_users, tables, model, final_features, flags):
    cands = candidates_for_chunk(chunk_users, tables, flags)
    df = assemble_dataset(cands, None, tables, is_inference=True, flags=flags)
    
    # Fill nulls for new features in case there are any
    for col in final_features:
        if col not in df.columns:
            df = df.with_columns(pl.lit(0.0).alias(col).cast(pl.Float32))
            
    df = df.with_columns(pl.col(pl.Float64).cast(pl.Float32))
    
    # Need to keep track of item ordering per customer
    df = df.with_columns(pl.col("item_id").cast(pl.Utf8))
    X = df[final_features].to_numpy()
    preds = model.predict(X)
    df = df.with_columns([
        pl.col(c).fill_null(0) for c in df.columns if c not in ["label", "customer_id", "item_id", "channel"]
    ])
    
    # Advanced Cross Ratios & Combinations from new_pir_lgbm_v3.py
    df = df.with_columns([
        (pl.col("u_category_l1_share") * pl.col("i_momentum_30d")).alias("cross_cat_momentum"),
        (pl.col("u_promo_purchase_ratio") * pl.col("i_promo_sales_ratio")).alias("cross_promo_affinity"),
        (pl.col("u_avg_items_per_bill") - pl.col("i_avg_items_in_its_bills")).abs().alias("basket_size_mismatch"),
        (pl.col("u_weekend_ratio") * pl.col("i_weekend_ratio")).alias("weekend_shopper_match"),
        (pl.col("ui_days_since_last") - pl.col("i_median_replenish_gap")).alias("replenishment_overdue_days")
    ])

    cat_cols = ["category_l1", "category_l2", "category_l3", "brand", "manufacturer", "size"]
    df = df.with_columns(pl.Series("score", preds))
    
    df = df.sort(["customer_id", "score"], descending=[False, True])
    agg_df = df.group_by("customer_id", maintain_order=True).agg(pl.col("item_id").alias("items"))
    
    out = {}
    for r in agg_df.iter_rows(named=True):
        cid = int(r["customer_id"])
        items = [str(x) for x in r["items"]]
        while len(items) < 10:
            items.append(items[-1] if items else "Unknown")
        out[cid] = tuple(items[:10])
    del cands, df, X, preds, agg_df; gc.collect()
    return out
"""

    full_helpers_code = (
        "import gc, pickle, re\n"
        "import polars as pl\n"
        "import numpy as np\n"
        "import lightgbm as lgb\n"
        "from datetime import datetime\n"
        "from scipy.sparse import csr_matrix\n"
        "from sklearn.decomposition import TruncatedSVD\n"
        "from sklearn.preprocessing import normalize\n"
        "from sklearn.feature_extraction.text import TfidfTransformer\n\n"
        + flags_block + "\n" + helpers_code + "\n" + streaming_code
    )

    # Apply Fast parameters
    full_helpers_code = full_helpers_code.replace('"Habitual": 250,', '"Habitual": 125,')
    full_helpers_code = full_helpers_code.replace('"Explorer": 800,', '"Explorer": 400,')
    full_helpers_code = full_helpers_code.replace('"Dormant": 120,', '"Dormant": 60,')
    full_helpers_code = full_helpers_code.replace('"New": 350,', '"New": 175,')
    full_helpers_code = full_helpers_code.replace('"Standard": 450', '"Standard": 225')

    full_helpers_code = full_helpers_code.replace('.head(400)', '.head(200)')
    full_helpers_code = full_helpers_code.replace('.head(250)', '.head(125)')
    full_helpers_code = full_helpers_code.replace('.head(150)', '.head(75)')

    cell_imports = """
import gc
import polars as pl
import numpy as np
import lightgbm as lgb
import pickle
from datetime import datetime
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

# ==============================================================
# 1. FILE PATHS  (replace on Kaggle)
# ==============================================================
TRANSACTION_PATH = "/kaggle/input/datasets/kinonquc/qkindataset2/transaction_full_2025.parquet"
ITEM_PATH        = "/kaggle/input/datasets/kinonquc/qkindataset2/items.parquet"
EVENT_PATH       = "/kaggle/input/datasets/kinonquc/qkindataset2/event_full_2025.parquet"

MODEL_IN_PATH    = "/kaggle/input/01-train-gpu/final_lgb_model.txt"
FEATURES_IN_PATH = "/kaggle/input/01-train-gpu/final_features.pkl"

OUTPUT_PATH      = "/kaggle/working/submission_jan.pkl"

CHUNK_SIZE = 10000   # 10k users per chunk to safely balance RAM against the 1000 SVD candidates
"""

    cell_load_model = """
print("Loading trained model and feature list...")
final_model = lgb.Booster(model_file=MODEL_IN_PATH)
with open(FEATURES_IN_PATH, "rb") as f:
    FINAL_FEATURES = pickle.load(f)
    
print(f"Loaded {len(FINAL_FEATURES)} features.")
"""

    cell_inference = """
# ==============================================================
# INFERENCE -- predict January 2026 for ALL 2025 users
# ==============================================================
items = load_items(flags)

print("Collecting full 2025 history for inference...")
hist_inf = scan_tx().collect()
ev_hist_inf = scan_events() if flags.get("use_events") else None
tx_full_inf = scan_tx_full() if flags.get("use_events") else None

all_users = hist_inf["customer_id"].unique().cast(pl.Int64).to_list()
print(f"Total users: {len(all_users)}")

print("Building lookup tables for full 2025 history (once)...")
tables_inf = precompute_lookup_tables(hist_inf, items, all_users, flags, hist_tx_full=tx_full_inf, events=ev_hist_inf)
del hist_inf, ev_hist_inf, tx_full_inf; gc.collect()

final_submission = {}
total_chunks = (len(all_users) + CHUNK_SIZE - 1) // CHUNK_SIZE

for i in range(0, len(all_users), CHUNK_SIZE):
    chunk = all_users[i : i + CHUNK_SIZE]
    result = infer_chunk(chunk, tables_inf, final_model, FINAL_FEATURES, flags=flags)
    final_submission.update(result)
    print(f"Chunk {i//CHUNK_SIZE+1}/{total_chunks} done. Total: {len(final_submission)}")
"""

    cell_export = """
print(f"Saving {len(final_submission)} users to {OUTPUT_PATH}...")
with open(OUTPUT_PATH, "wb") as f:
    pickle.dump(final_submission, f)
print("Done!")
"""

    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_code_cell(cell_imports),
        nbf.v4.new_code_cell(full_helpers_code),
        nbf.v4.new_code_cell(cell_load_model),
        nbf.v4.new_code_cell(cell_inference),
        nbf.v4.new_code_cell(cell_export),
    ]

    with open("02_Inference_CPU.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("02_Inference_CPU.ipynb generated successfully!")

if __name__ == "__main__":
    run()
