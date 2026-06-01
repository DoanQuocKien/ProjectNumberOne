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
# HARDCODED FLAGS FOR GPU TRAINING
# ==============================================================
flags = {
    "filter_dead": False,
    "use_events": False,
    "copurchase": True,
    "rich_metadata": False,
    "discount_features": False
}
"""

    streaming_code = """
def build_lgb_dataset_streaming(target_tx, tables, sample_n, flags, reference=None):
    '''OOM-safe Phase B training for large sample_n (100k users).
    Preallocates numpy arrays to completely avoid np.vstack memory duplication.
    '''
    sample_users = (
        target_tx["customer_id"].unique().cast(pl.Int64)
        .sample(n=min(sample_n, target_tx["customer_id"].n_unique()), seed=42).to_list()
    )
    sub_chunk = 2000
    total = (len(sample_users) + sub_chunk - 1) // sub_chunk
    
    X_mat, y_mat = None, None
    all_groups = []
    final_features = None
    current_idx = 0
    
    for i in range(0, len(sample_users), sub_chunk):
        sub = sample_users[i:i + sub_chunk]
        cands = candidates_for_chunk(sub, tables, flags)
        df    = assemble_dataset(cands, target_tx, tables, is_inference=False, flags=flags)
        del cands; gc.collect()
        
        df = df.sort("customer_id")
        
        if final_features is None:
            final_features = [c for c in df.columns if c not in ["customer_id", "item_id", "label", "channel", "rank", "final_rank"]]
            
        chunk_X = df.select(final_features).to_numpy()
        chunk_y = df["label"].to_numpy()
        chunk_groups = df.group_by("customer_id", maintain_order=True).len()["len"].to_numpy()
        
        if X_mat is None:
            # Preallocate with a generous 1050 candidates per user buffer to avoid re-allocation
            max_rows = len(sample_users) * 1050
            X_mat = np.empty((max_rows, chunk_X.shape[1]), dtype=np.float32)
            y_mat = np.empty(max_rows, dtype=np.int32)
            
        rows = chunk_X.shape[0]
        # Dynamically resize if we somehow exceed the buffer (very rare)
        if current_idx + rows > X_mat.shape[0]:
            extra_rows = max(rows, 500000)
            X_mat = np.vstack([X_mat, np.empty((extra_rows, chunk_X.shape[1]), dtype=np.float32)])
            y_mat = np.concatenate([y_mat, np.empty(extra_rows, dtype=np.int32)])
            
        X_mat[current_idx : current_idx + rows] = chunk_X
        y_mat[current_idx : current_idx + rows] = chunk_y
        all_groups.append(chunk_groups)
        
        current_idx += rows
        del df, chunk_X, chunk_y, chunk_groups; gc.collect()
        print(f"  Streamed chunk {i//sub_chunk+1}/{total} (RAM: ~safe)", end="\\r")
        
    print("")
    # Slice off the unused preallocated space (creates a view, ZERO memory copy!)
    X_mat = X_mat[:current_idx]
    y_mat = y_mat[:current_idx]
    group_mat = np.concatenate(all_groups)
    del all_groups; gc.collect()
    
    ds = lgb.Dataset(X_mat, label=y_mat, group=group_mat, reference=reference, free_raw_data=False)
    ds.construct()  # Force construction early to build bin mappers
    
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
import optuna
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

MODEL_OUT_PATH   = "/kaggle/working/final_lgb_model.txt"
FEATURES_OUT_PATH= "/kaggle/working/final_features.pkl"
"""

    cell_data = """
items = load_items(flags)

print("Collecting Nov & Dec history for TRUE temporal validation...")
# Validation Split (Target = Dec, History = Jan-Nov)
hist_dec = scan_tx(cutoff_end=datetime(2025, 12, 1)).collect()
targ_dec = scan_tx(cutoff_start=datetime(2025, 12, 1)).collect()

# Training Split (Target = Nov, History = Jan-Oct)
hist_nov = scan_tx(cutoff_end=datetime(2025, 11, 1)).collect()
targ_nov = scan_tx(cutoff_start=datetime(2025, 11, 1), cutoff_end=datetime(2025, 12, 1)).collect()

all_valid_users = targ_dec["customer_id"].unique().cast(pl.Int64).to_list()
all_train_users = targ_nov["customer_id"].unique().cast(pl.Int64).to_list()

print("Building lookup tables for Dec history (Validation)...")
tables_dec = precompute_lookup_tables(hist_dec, items, all_valid_users, flags)
del hist_dec; gc.collect()

print("Building lookup tables for Nov history (Training)...")
tables_nov = precompute_lookup_tables(hist_nov, items, all_train_users, flags)
del hist_nov; gc.collect()
"""

    cell_optuna = """
print("Sampling subsets (60k from Nov, 20k from Dec) to keep GPU tuning fast and memory-safe...")
np.random.seed(42)
train_users = np.random.choice(all_train_users, size=min(60000, len(all_train_users)), replace=False).tolist()
valid_users = np.random.choice(all_valid_users, size=min(20000, len(all_valid_users)), replace=False).tolist()

train_tx = targ_nov.filter(pl.col("customer_id").is_in(train_users))
valid_tx = targ_dec.filter(pl.col("customer_id").is_in(valid_users))

print("Building streaming training dataset (Nov Target)...")
train_data, FINAL_FEATURES = build_lgb_dataset_streaming(train_tx, tables_nov, sample_n=60000, flags=flags)

print("Building streaming validation dataset (Dec Target)...")
valid_data, _ = build_lgb_dataset_streaming(valid_tx, tables_dec, sample_n=20000, flags=flags, reference=train_data)

del targ_nov, train_tx, valid_tx; gc.collect()

print("Saving FINAL_FEATURES list to disk...")
with open(FEATURES_OUT_PATH, "wb") as f:
    pickle.dump(FINAL_FEATURES, f)

def objective(trial):
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'eval_at': 10,
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 31, 127),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 200),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 0.9),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 0.9),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-3, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-3, 10.0, log=True),
        'device_type': 'gpu',
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    evals_result = {}
    model = lgb.train(
        params,
        train_data,
        valid_sets=[valid_data],
        valid_names=['valid'],
        num_boost_round=300,
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.record_evaluation(evals_result)]
    )
    
    best_ndcg = evals_result['valid']['ndcg@10'][-1]
    del model; gc.collect()
    return best_ndcg

print("Running Optuna Hyperparameter Tuning (10 trials)...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)

print(f"Best trial: {study.best_value}")
print(f"Best params: {study.best_params}")
"""

    cell_train = """
print("Training final model with best params on ALL December data...")
# Now that we found best params using Nov->Dec validation, we retrain on Dec data!
del train_data, valid_data, tables_nov; gc.collect()

print("Rebuilding dataset using Dec Target (100k users)...")
np.random.seed(42)
final_users = np.random.choice(all_valid_users, size=min(100000, len(all_valid_users)), replace=False).tolist()
final_tx = targ_dec.filter(pl.col("customer_id").is_in(final_users))

final_lgb_train, _ = build_lgb_dataset_streaming(final_tx, tables_dec, sample_n=100000, flags=flags)
del final_tx, targ_dec; gc.collect()

best_params = study.best_params
best_params['objective'] = 'lambdarank'
best_params['metric'] = 'ndcg'
best_params['eval_at'] = 10
best_params['boosting_type'] = 'gbdt'
best_params['device_type'] = 'gpu'
best_params['random_state'] = 42
best_params['n_jobs'] = -1
best_params['verbose'] = -1

final_model = lgb.train(
    best_params, 
    final_lgb_train, 
    num_boost_round=150,
)

final_model.save_model(MODEL_OUT_PATH)
print(f"Model saved successfully to {MODEL_OUT_PATH}!")
"""

    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_code_cell(cell_imports),
        nbf.v4.new_code_cell(full_helpers_code),
        nbf.v4.new_code_cell(cell_data),
        nbf.v4.new_code_cell(cell_optuna),
        nbf.v4.new_code_cell(cell_train),
    ]

    with open("01_Train_GPU.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("01_Train_GPU.ipynb generated successfully!")

if __name__ == "__main__":
    run()
