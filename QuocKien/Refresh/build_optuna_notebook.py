import nbformat as nbf
from build_notebooks import CELL_IMPORTS, CELL_HELPERS

OPTUNA_CELL = """
# ==============================================================
# OPTUNA HYPERPARAMETER TUNING
# ==============================================================
import optuna
from sklearn.model_selection import GroupShuffleSplit
import lightgbm as lgb
import numpy as np

print("Building lookup tables for Nov history...")
items = load_items()
hist_nov = scan_tx(cutoff_end=datetime(2025, 12, 1)).collect()
targ_dec = scan_tx(cutoff_start=datetime(2025, 12, 1), cutoff_end=datetime(2026, 1, 1)).collect()

all_dec_users = targ_dec["customer_id"].unique().cast(pl.Int64).to_list()
tables_dec = precompute_lookup_tables(hist_nov, items, all_dec_users)

print("Building training dataset (50k users)...")
nov_df = build_training_dataset(hist_nov, targ_dec, items, sample_n=50000, tables=tables_dec)
del hist_nov, targ_dec, tables_dec; gc.collect()

# Prepare Dataset
users_nov = nov_df["customer_id"].unique().to_list()
np.random.seed(42); np.random.shuffle(users_nov)
train_users = users_nov[:int(0.8*len(users_nov))]
valid_users = users_nov[int(0.8*len(users_nov)):]

nov_train = nov_df.filter(pl.col("customer_id").is_in(train_users)).sort("customer_id")
nov_valid = nov_df.filter(pl.col("customer_id").is_in(valid_users)).sort("customer_id")
del nov_df; gc.collect()

FEATURES = [c for c in nov_train.columns if c not in ["customer_id", "item_id", "label", "final_rank"]]
q_tr = nov_train.group_by("customer_id", maintain_order=True).len()["len"].to_numpy()
q_va = nov_valid.group_by("customer_id", maintain_order=True).len()["len"].to_numpy()

X_train = nov_train[FEATURES].to_numpy()
y_train = nov_train["label"].to_numpy()
X_valid = nov_valid[FEATURES].to_numpy()
y_valid = nov_valid["label"].to_numpy()
del nov_train, nov_valid; gc.collect()

def objective(trial):
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'eval_at': 1,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 16, 128),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 50, 500),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 5),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'random_state': 42,
        'n_jobs': -1,
        'device_type': 'gpu',
        'verbose': -1,
        'feature_pre_filter': False
    }
    
    lgb_tr = lgb.Dataset(X_train, label=y_train, group=q_tr)
    lgb_va = lgb.Dataset(X_valid, label=y_valid, group=q_va, reference=lgb_tr)
    
    model = lgb.train(
        params, 
        lgb_tr, 
        num_boost_round=150,
        valid_sets=[lgb_va], 
        callbacks=[lgb.early_stopping(20, verbose=False)]
    )
    
    # Get the best NDCG@1 score on the validation set
    best_score = model.best_score['valid_0']['ndcg@1']
    return best_score

print("Starting Optuna search...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print("\\nBest Trial:")
print(f"  NDCG@1: {study.best_value:.5f}")
print("  Params: ")
for key, value in study.best_trial.params.items():
    print(f"    '{key}': {value},")
"""

def make_optuna_notebook():
    nb = nbf.v4.new_notebook()
    nb.cells.append(nbf.v4.new_code_cell("!pip install optuna"))
    nb.cells.append(nbf.v4.new_code_cell(CELL_IMPORTS))
    nb.cells.append(nbf.v4.new_code_cell(CELL_HELPERS))
    nb.cells.append(nbf.v4.new_code_cell(OPTUNA_CELL))
    return nb

if __name__ == "__main__":
    nb_optuna = make_optuna_notebook()
    with open("Final_PIR_Optuna.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb_optuna, f)
    print("Final_PIR_Optuna.ipynb created successfully!")
