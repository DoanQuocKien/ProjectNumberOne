import nbformat as nbf
import os

SOURCE_SCRIPT = "d:/CS116/ProjectNumberOne/QuocKien/Refresh/local_ablation_test.py"
OUTPUT_DIR = "d:/CS116/ProjectNumberOne/QuocKien/Refresh/ablation_notebooks"

KAGGLE_PATHS = """
# ──────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────
TRANSACTION_PATH = "/kaggle/input/qkindataset2/transaction_full_2025.parquet"
ITEM_PATH = "/kaggle/input/qkindataset2/items.parquet"
EVENT_PATH = "/kaggle/input/qkindataset2/event_full_2025.parquet"

# Install optuna if not present
!pip install optuna -q
"""

OPTUNA_BLOCK = r"""
    # ==========================================
    # OPTUNA HYPERPARAMETER TUNING
    # ==========================================
    import optuna
    
    # We need to keep nov_valid for custom evaluation
    # (del nov_train; gc.collect() is fine, we will just delete it inside)
    del nov_train; gc.collect()
    
    # Build ground truth for the validation set (targ_nov)
    valid_users = users_nov[int(0.8*len(users_nov)):]
    ground_truth = {}
    valid_truth = targ_nov.filter(pl.col("customer_id").is_in(valid_users)).group_by("customer_id").agg(pl.col("item_id").unique())
    for r in valid_truth.iter_rows(named=True):
        ground_truth[int(r["customer_id"])] = [str(x) for x in r["item_id"]]

    def objective(trial):
        params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'eval_at': 10,
            'device_type': 'gpu',  # Run on GPU
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 31, 255),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 200),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1,
            'feature_pre_filter': False
        }

        # Train model with early stopping on NDCG
        model = lgb.train(
            params,
            lgb_tr,
            num_boost_round=150,
            valid_sets=[lgb_va],
            callbacks=[lgb.early_stopping(15, verbose=False)]
        )

        # Predict on valid set
        preds = model.predict(nov_valid[FEATURES].to_numpy())
        
        # Build predictions dict
        res = nov_valid.select(["customer_id", "item_id"]).with_columns(pl.Series("pred", preds))
        top_k = res.sort(["customer_id", "pred", "item_id"], descending=[False, True, False]).group_by("customer_id").head(10)
        predictions = {}
        for r in top_k.group_by("customer_id").agg(pl.col("item_id")).iter_rows(named=True):
            predictions[int(r["customer_id"])] = [str(x) for x in r["item_id"]][:10]
            
        # Compute custom score
        metrics = compute_metrics(predictions, ground_truth)
        return metrics["Precision@10"]

    print("\nStarting Optuna study on GPU with Precision@10 Custom Score...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=40)

    print(f"\n{'='*60}")
    print(f"BEST PARAMS FOUND:")
    print(f"Best Precision@10: {study.best_value:.5f}")
    print(f"Params: {study.best_params}")
    print(f"{'='*60}\n")

    return study.best_params
"""

def generate():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(SOURCE_SCRIPT, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Extract everything up to lgb_va creation
    main_idx = content.find("if __name__ == \"__main__\":")
    core_code = content[:main_idx]
    
    path_start = core_code.find("# PATHS")
    path_start = core_code.rfind("# ──", 0, path_start)
    path_end = core_code.find("SAMPLE_N")
    
    core_code = core_code[:path_start] + KAGGLE_PATHS + "\n" + core_code[path_end:]
    
    # Find the end of data prep (after lgb_va is created and gc.collect())
    prep_end = core_code.find("del nov_train, nov_valid; gc.collect()")
    if prep_end == -1:
        raise ValueError("Could not find data prep end.")
    
    # We deliberately cut BEFORE the 'del nov_train, nov_valid; gc.collect()' line
    # so that we can keep nov_valid for custom Optuna evaluation.
    tuning_code = core_code[:prep_end] + "\n" + OPTUNA_BLOCK
    
    nb = nbf.v4.new_notebook()
    cell1 = nbf.v4.new_code_cell(tuning_code)
    nb.cells.append(cell1)
    
    flags = {
        "filter_dead": True,
        "use_events": True,
        "copurchase": True,
        "rich_metadata": True,
        "discount_features": True
    }
    flag_str = ",\n    ".join(f'"{k}": {v}' for k, v in flags.items())
    runner_code = f"""
flags = {{
    {flag_str}
}}
print(f"Running Optuna Tuning with All Changes enabled...")
best_params = run_ablation(flags)
"""
    cell2 = nbf.v4.new_code_cell(runner_code.strip())
    nb.cells.append(cell2)
    
    out_path = os.path.join(OUTPUT_DIR, "PIR_Optuna_Tuning.ipynb")
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Generated: {out_path}")

if __name__ == "__main__":
    generate()
