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
# HARDCODED FLAGS FOR FINAL JAN SUBMISSION
# ==============================================================
flags = {
    "filter_dead": True,
    "use_events": False,
    "copurchase": False,
    "rich_metadata": False,
    "discount_features": False
}
"""

    streaming_code = """
def build_lgb_dataset_streaming(target_tx, tables, sample_n, flags):
    '''OOM-safe Phase B training for large sample_n (100k users).
    Streams 2k-user chunks directly to narrow numpy arrays.
    '''
    sample_users = (
        target_tx["customer_id"].unique().cast(pl.Int64)
        .sample(n=min(sample_n, target_tx["customer_id"].n_unique()), seed=42).to_list()
    )
    sub_chunk = 2000
    total = (len(sample_users) + sub_chunk - 1) // sub_chunk
    all_X, all_y, all_groups = [], [], []
    final_features = None
    for i in range(0, len(sample_users), sub_chunk):
        sub = sample_users[i:i + sub_chunk]
        cands = candidates_for_chunk(sub, tables, flags)
        df    = assemble_dataset(cands, target_tx, tables, is_inference=False, flags=flags)
        del cands; gc.collect()
        df = df.sort("customer_id")
        if final_features is None:
            final_features = [c for c in df.columns if c not in ["customer_id", "item_id", "label", "channel", "rank", "final_rank"]]
        all_groups.append(df.group_by("customer_id", maintain_order=True).len()["len"].to_numpy())
        all_y.append(df["label"].to_numpy())
        df = df.select(final_features)
        all_X.append(df.to_numpy())
        del df; gc.collect()
        print(f"  Streamed chunk {i//sub_chunk+1}/{total} (RAM: ~safe)", end="\\r")
    print("")
    return lgb.Dataset(np.vstack(all_X), label=np.concatenate(all_y), group=np.concatenate(all_groups)), final_features

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
OUTPUT_PATH      = "/kaggle/working/submission_jan.pkl"

CHUNK_SIZE = 10000   # 10k users per chunk to safely balance RAM against the 1000 SVD candidates
"""

    cell_data = """
items = load_items(flags)

print("Collecting history splits (lazy scans)...")
"""

    cell_phase_b = """
# ==============================================================
# PHASE B -- final model (Nov history, Dec target, 80k users)
# ==============================================================
print("Collecting Dec history...")
hist_dec = scan_tx(cutoff_end=datetime(2025, 12, 1)).collect()
targ_dec = scan_tx(cutoff_start=datetime(2025, 12, 1)).collect()
ev_hist_dec = scan_events(cutoff_end=datetime(2025, 12, 1)) if flags.get("use_events") else None
tx_full_dec = scan_tx_full(cutoff_end=datetime(2025, 12, 1)) if flags.get("use_events") else None

all_users = targ_dec["customer_id"].unique().cast(pl.Int64).to_list()

print("Building lookup tables for Dec history...")
tables_dec = precompute_lookup_tables(hist_dec, items, all_users, flags, hist_tx_full=tx_full_dec, events=ev_hist_dec)
del hist_dec, tx_full_dec, ev_hist_dec; gc.collect()

final_lgb_train, FINAL_FEATURES = build_lgb_dataset_streaming(targ_dec, tables_dec, sample_n=100000, flags=flags)
del targ_dec; gc.collect()

best_params = {
    'objective': 'lambdarank', 'metric': 'ndcg', 'eval_at': 10,
    'learning_rate': 0.05234522483024746, 'num_leaves': 155, 'min_data_in_leaf': 156,
    'feature_fraction': 0.48585196246455786, 'bagging_fraction': 0.6681482272360668, 'bagging_freq': 5,
    'lambda_l1': 2.88737161982848, 'lambda_l2': 0.1863820285933072,
    'device_type': 'cpu',
    'random_state': 42, 'n_jobs': -1, 'verbose': -1, 'feature_pre_filter': False
}

final_model = lgb.train(best_params, final_lgb_train, num_boost_round=150,
                         valid_sets=[final_lgb_train], valid_names=["train"],
                         callbacks=[lgb.early_stopping(30)])
del final_lgb_train; gc.collect()
print("Final model ready!")
"""

    cell_inference = """
# ==============================================================
# INFERENCE -- predict January 2026 for ALL 2025 users
# ==============================================================
print("Collecting full 2025 history for inference...")
hist_inf = scan_tx().collect()
ev_hist_inf = scan_events()
tx_full_inf = scan_tx_full()

all_users = hist_inf["customer_id"].unique().cast(pl.Int64).to_list()
print(f"Total users: {len(all_users)}")

del tables_dec; gc.collect()
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
        nbf.v4.new_code_cell(cell_data),
        nbf.v4.new_code_cell(cell_phase_b),
        nbf.v4.new_code_cell(cell_inference),
        nbf.v4.new_code_cell(cell_export),
    ]

    with open("Final_PIR_Jan_Submission_Fast.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Final_PIR_Jan_Submission_Fast.ipynb generated successfully!")

if __name__ == "__main__":
    run()
