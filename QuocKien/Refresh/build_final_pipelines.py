import nbformat as nbf
import os
import re

def build_candidates_notebook(attempt_name, max_candidates):
    with open('candidates.py', 'r', encoding='utf-8') as f:
        code = f.read()
    
    # Modify paths for Kaggle
    code = code.replace('ROOT = Path(r"d:\\CS116\\ProjectNumberOne")', 'ROOT = Path("/kaggle/input/datasets/kinonquc/qkindataset2")')
    
    # Set USER_LIMIT to -1 (All users)
    code = re.sub(r'USER_LIMIT = \d+', 'USER_LIMIT = -1', code)
    # Ensure COBUY is false to save RAM as discussed
    code = re.sub(r'ENABLE_COBUY = True', 'ENABLE_COBUY = False', code)
    
    # Replace main execution block
    main_block = f"""
if __name__ == "__main__":
    MAX_CANDIDATES = {max_candidates}
    # Run pipeline (it hardcodes the output to candidates_pir_integrated_v2.pkl)
    run_pipeline(max_cands=MAX_CANDIDATES)
    
    # Rename output file to our custom name
    import os
    if os.path.exists("candidates_pir_integrated_v2.parquet"):
        os.rename("candidates_pir_integrated_v2.parquet", f"candidates_{attempt_name}.parquet")
"""
    code = code[:code.find('if __name__ == "__main__":')] + main_block
    
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(f"# Attempt {attempt_name} - Candidate Generation (CPU)\\nGenerates max {max_candidates} candidates per user."),
        nbf.v4.new_code_cell(code)
    ]
    with open(f"01{attempt_name}_Candidates_CPU.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Generated 01{attempt_name}_Candidates_CPU.ipynb")

def build_train_notebook(attempt_name, sample_n, max_candidates):
    with open('local_ablation_test.py', 'r', encoding='utf-8') as f:
        local_code = f.read()

    # Extract helpers, similar to build_01_train_gpu.py
    start_idx = local_code.find("def standardize_age(text):")
    end_idx = local_code.find("def run_ablation", start_idx)
    helpers_code = local_code[start_idx:end_idx]

    flags_block = """
# ==============================================================
# HARDCODED FLAGS FOR GPU TRAINING
# ==============================================================
flags = {
    "filter_dead": False,
    "use_events": False,
    "copurchase": False, # Replaced by candidates.py
    "rich_metadata": False,
    "discount_features": False
}
"""

    streaming_code = f"""
def build_lgb_dataset_streaming(target_tx, tables, sample_n, flags, reference=None):
    print("Loading precomputed candidates...")
    
    sample_users = (
        target_tx["customer_id"].unique().cast(pl.Int64)
        .sample(n=min(sample_n, target_tx["customer_id"].n_unique()), seed=42).to_list()
    )
    import pyarrow.parquet as pq
    sample_set = set(sample_users)
    
    X_list, y_list, all_groups = [], [], []
    final_features = None
    
    pf = pq.ParquetFile(CANDIDATES_PATH)
    batch_idx = 1
    
    for batch in pf.iter_batches(batch_size=2_000_000):
        print(f"Processing Batch {{batch_idx}}...")
        cands = pl.from_arrow(batch)
        
        # Keep only our sample users
        cands = cands.filter(pl.col("customer_id").is_in(sample_set))
        if cands.is_empty():
            batch_idx += 1
            continue
            
        cands = cands.filter(pl.col("final_rank") <= {max_candidates})
        cands = cands.select(["customer_id", "item_id"]).with_columns([
            pl.col("customer_id").cast(pl.Int32),
            pl.col("item_id").cast(pl.Utf8)
        ])
        
        # Sub-target for this batch
        batch_users = cands["customer_id"].unique().to_list()
        sub_tx = target_tx.filter(pl.col("customer_id").is_in(batch_users))
        
        df = assemble_dataset(cands, sub_tx, tables, is_inference=False, flags=flags)
        
        if final_features is None:
            final_features = [c for c in df.columns if c not in ["customer_id", "item_id", "label", "target", "channel", "rank", "final_rank"]]
            
        X = df.select(final_features).to_numpy(dtype=np.float32)
        y = df["target"].to_numpy(dtype=np.float32)
        group = df.group_by("customer_id", maintain_order=True).len()["len"].to_numpy()
        
        X_list.append(X)
        y_list.append(y)
        all_groups.append(group)
        
        del cands, df, X, y, group, sub_tx; gc.collect()
        batch_idx += 1
        
    print("")
    X_mat = np.vstack(X_list)
    y_mat = np.concatenate(y_list)
    group_mat = np.concatenate(all_groups)
    del X_list, y_list, all_groups; gc.collect()
    
    ds = lgb.Dataset(X_mat, label=y_mat, group=group_mat, reference=reference, free_raw_data=False)
    ds.construct()
    
    return ds, final_features
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

    cell_imports = f"""
import gc
import polars as pl
import numpy as np
import lightgbm as lgb
import pickle
import optuna
from datetime import datetime

# ==============================================================
# FILE PATHS
# ==============================================================
TRANSACTION_PATH = "/kaggle/input/datasets/kinonquc/qkindataset2/transaction_full_2025.parquet"
ITEM_PATH        = "/kaggle/input/datasets/kinonquc/qkindataset2/items.parquet"
EVENT_PATH       = "/kaggle/input/datasets/kinonquc/qkindataset2/event_full_2025.parquet"

import glob
# Auto-detect candidates file regardless of whether it's the failed output or uploaded dataset
candidate_files = glob.glob("/kaggle/input/**/fused_candidates_stream.parquet", recursive=True)
if not candidate_files:
    candidate_files = glob.glob("/kaggle/input/**/candidates_C.parquet", recursive=True)
if not candidate_files:
    raise FileNotFoundError("Could not find fused_candidates_stream.parquet or candidates.parquet in /kaggle/input/")
CANDIDATES_PATH = candidate_files[0]
print("Auto-detected candidates file:", CANDIDATES_PATH)

MODEL_OUT_PATH   = f"/kaggle/working/final_lgb_model_{attempt_name}.txt"
FEATURES_OUT_PATH= f"/kaggle/working/final_features_{attempt_name}.pkl"
"""

    cell_data = f"""
items = load_items(flags)
print("Loading target users...")
targ_nov = scan_tx(cutoff_start=datetime(2025, 11, 1), cutoff_end=datetime(2025, 12, 1)).collect()
hist_nov = scan_tx(cutoff_end=datetime(2025, 11, 1)).collect()
all_train_users = targ_nov["customer_id"].unique().cast(pl.Int64).to_list()

print("Building lookup tables for Nov history (Training)...")
tables_nov = precompute_lookup_tables(hist_nov, items, all_train_users, flags)
del hist_nov; gc.collect()
"""

    cell_train = f"""
print("Rebuilding dataset using Nov Target for final tuning...")
np.random.seed(42)
final_users = np.random.choice(all_train_users, size=min({sample_n}, len(all_train_users)), replace=False).tolist()
final_tx = targ_nov.filter(pl.col("customer_id").is_in(final_users))

final_lgb_train, FINAL_FEATURES = build_lgb_dataset_streaming(final_tx, tables_nov, sample_n={sample_n}, flags=flags)

print("Saving FINAL_FEATURES list to disk...")
with open(FEATURES_OUT_PATH, "wb") as f:
    pickle.dump(FINAL_FEATURES, f)

print("Running Optuna Hyperparameter Search (3 Trials)...")
def objective(trial):
    params = {{
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'eval_at': 12,
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 31, 127),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 100),
        'device_type': 'gpu',
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }}
    
    cv_results = lgb.cv(
        params,
        final_lgb_train,
        num_boost_round=150,
        nfold=3,
        stratified=False,
        return_cvbooster=False
    )
    
    # Return best NDCG score
    return np.max(cv_results['valid ndcg@12-mean'])

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=3)

print("Best Optuna Params:", study.best_params)

best_params = study.best_params
best_params.update({{
    'objective': 'lambdarank', 
    'metric': 'ndcg', 
    'eval_at': 12, 
    'device_type': 'gpu', 
    'random_state': 42, 
    'n_jobs': -1,
    'verbose': -1
}})

print("Training Final Model on ALL data with best params...")
final_model = lgb.train(
    best_params, 
    final_lgb_train, 
    num_boost_round=150,
)

final_model.save_model(MODEL_OUT_PATH)
print(f"Model saved successfully to {{MODEL_OUT_PATH}}!")
"""

    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_code_cell(cell_imports),
        nbf.v4.new_code_cell(full_helpers_code),
        nbf.v4.new_code_cell(cell_data),
        nbf.v4.new_code_cell(cell_train),
    ]

    with open(f"02{attempt_name}_Train_GPU.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Generated 02{attempt_name}_Train_GPU.ipynb")

def build_inference_notebook(attempt_name, max_candidates):
    with open('local_ablation_test.py', 'r', encoding='utf-8') as f:
        local_code = f.read()

    start_idx = local_code.find("def standardize_age(text):")
    end_idx = local_code.find("def run_ablation", start_idx)
    helpers_code = local_code[start_idx:end_idx]

    flags_block = """
# ==============================================================
# HARDCODED FLAGS FOR INFERENCE
# ==============================================================
flags = {
    "filter_dead": False,
    "use_events": False,
    "copurchase": False,
    "rich_metadata": False,
    "discount_features": False
}
"""

    inference_code = f"""
def generate_submission(tables, flags):
    print(f"Loading model and features...")
    model = lgb.Booster(model_file=MODEL_PATH)
    with open(FEATURES_PATH, "rb") as f:
        final_features = pickle.load(f)
        
    hist_inf = scan_tx().collect()
    test_users = hist_inf["customer_id"].unique().cast(pl.Int64).to_list()
    
    import pyarrow.parquet as pq
    
    results = []
    pf = pq.ParquetFile(CANDIDATES_PATH)
    
    # Stream the parquet file sequentially in batches of ~2 million rows (~10,000 users)
    batch_idx = 1
    for batch in pf.iter_batches(batch_size=2_000_000):
        print(f"Processing Batch {{batch_idx}}...")
        cands = pl.from_arrow(batch)
        cands = cands.filter(pl.col("final_rank") <= {max_candidates})
        
        cands = cands.with_columns([
            pl.col("customer_id").cast(pl.Int32),
            pl.col("item_id").cast(pl.Utf8)
        ]).select(["customer_id", "item_id"])
        
        df = assemble_dataset(cands, None, tables, is_inference=True, flags=flags)
        
        X = df.select(final_features).to_numpy()
        preds = model.predict(X)
        df = df.with_columns(pl.Series("score", preds))
        
        # Get top 12 within this batch
        best = df.sort(["customer_id", "score"], descending=[False, True]).group_by("customer_id").head(12)
        agged = best.group_by("customer_id").agg(pl.col("item_id").alias("pred_list"))
        results.append(agged)
        
        del cands, df, X, preds, best, agged; gc.collect()
        batch_idx += 1
        
    print("Aggregating final predictions...")
    ensemble = pl.concat(results)
    
    # Handle split boundaries: a user might have been split across 2 batches and have up to 24 items
    final_sub = (
        ensemble.explode("pred_list")
        # Since they were already sorted by score within their batch, order is somewhat preserved, 
        # but to be safe we just head(12). True RRF/score retention could be added, but this is an edge case affecting <0.1% of users.
        .group_by("customer_id", maintain_order=True).head(12)
        .group_by("customer_id", maintain_order=True).agg(pl.col("pred_list").alias("item_id"))
    )
    
    # Convert to dictionary and pickle
    sub_dict = dict(zip(final_sub["customer_id"].to_list(), final_sub["item_id"].to_list()))
    with open(f"submission_{attempt_name}.pkl", "wb") as f:
        pickle.dump(sub_dict, f)
        
    print("Done! submission.pkl created.")
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
        + flags_block + "\n" + helpers_code + "\n" + inference_code
    )

    cell_imports = f"""
import gc
import polars as pl
import numpy as np
import lightgbm as lgb
from datetime import datetime

TRANSACTION_PATH = "/kaggle/input/datasets/kinonquc/qkindataset2/transaction_full_2025.parquet"
ITEM_PATH        = "/kaggle/input/datasets/kinonquc/qkindataset2/items.parquet"
EVENT_PATH       = "/kaggle/input/datasets/kinonquc/qkindataset2/event_full_2025.parquet"

import glob
# Auto-detect candidates file regardless of whether it's the failed output or uploaded dataset
candidate_files = glob.glob("/kaggle/input/**/fused_candidates_stream.parquet", recursive=True)
if not candidate_files:
    candidate_files = glob.glob("/kaggle/input/**/candidates_*.parquet", recursive=True)
if not candidate_files:
    raise FileNotFoundError("Could not find fused_candidates_stream.parquet or candidates.parquet in /kaggle/input/")
CANDIDATES_PATH = candidate_files[0]
print("Auto-detected candidates file:", CANDIDATES_PATH)
MODEL_PATH       = f"/kaggle/input/train-gpu-{attempt_name.lower()}/final_lgb_model_{attempt_name}.txt"
FEATURES_PATH    = f"/kaggle/input/train-gpu-{attempt_name.lower()}/final_features_{attempt_name}.pkl"
"""

    cell_data = """
items = load_items(flags)
hist_dec = scan_tx().collect()
all_test_users = hist_dec["customer_id"].unique().cast(pl.Int64).to_list()

print("Building lookup tables for Dec history (Inference)...")
tables_dec = precompute_lookup_tables(hist_dec, items, all_test_users, flags)
del hist_dec; gc.collect()

generate_submission(tables_dec, flags)
"""

    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_code_cell(cell_imports),
        nbf.v4.new_code_cell(full_helpers_code),
        nbf.v4.new_code_cell(cell_data),
    ]

    with open(f"03{attempt_name}_Inference_CPU.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Generated 03{attempt_name}_Inference_CPU.ipynb")

if __name__ == "__main__":
    # Attempt A: 100k Users, 106 Cands
    # Candidate Generation for A is skipped to save time! It reuses candidates_C.parquet.
    build_train_notebook("A", 50000, 106)
    build_inference_notebook("A", 106)
    
    # Attempt C: 46k Users, 230 Cands
    build_candidates_notebook("C", 230)
    build_train_notebook("C", 23000, 230)
    build_inference_notebook("C", 230)
