import sys
import os
from build_tuning_notebook import CELL_IMPORTS, CELL_HELPERS

cell_data = '''
items = load_items()
'''
cell_optuna = '''
import optuna
import gc
from datetime import datetime
import polars as pl

print('Building Nov Train Dataset...')
hist_nov = scan_tx(cutoff_end=datetime(2025, 11, 1)).collect()
targ_nov = scan_tx(cutoff_start=datetime(2025, 11, 1), cutoff_end=datetime(2025, 12, 1)).collect()
tables_nov = precompute_lookup_tables(hist_nov, items, targ_nov['customer_id'].unique().cast(pl.Int64).to_list())

dummy_df = build_training_dataset(hist_nov, targ_nov, items, sample_n=50, tables=tables_nov)
FEATURES = [c for c in dummy_df.columns if c not in ['customer_id', 'item_id', 'label', 'final_rank']]
del dummy_df; gc.collect()

print(f'Total features used for tuning: {len(FEATURES)}')

lgb_train = build_lgb_dataset_streaming(targ_nov, tables_nov, FEATURES, sample_n=50000)
del hist_nov, targ_nov, tables_nov; gc.collect()

print('Building Dec Valid Dataset...')
hist_dec = scan_tx(cutoff_end=datetime(2025, 12, 1)).collect()
targ_dec = scan_tx(cutoff_start=datetime(2025, 12, 1)).collect()
tables_dec = precompute_lookup_tables(hist_dec, items, targ_dec['customer_id'].unique().cast(pl.Int64).to_list())

lgb_valid = build_lgb_dataset_streaming(targ_dec, tables_dec, FEATURES, sample_n=20000)
del hist_dec, targ_dec, tables_dec; gc.collect()

valid_y = lgb_valid.get_label()
valid_groups = lgb_valid.get_group()

def compute_hybrid_score(preds, labels, groups):
    import numpy as np
    idx = 0
    mrr_sum = p10_sum = map10_sum = hr10_sum = 0.0
    num_users = len(groups)
    
    for g in groups:
        p = preds[idx:idx+g]
        y = labels[idx:idx+g]
        idx += g
        
        if y.sum() == 0:
            num_users -= 1
            continue
            
        order = np.argsort(-p)
        y_sorted = y[order]
        y_top10 = y_sorted[:10]
        
        hits = y_top10.sum()
        if hits > 0:
            hr10_sum += 1.0
            p10_sum += hits / 10.0
            
            hit_indices = np.where(y_sorted > 0)[0]
            mrr_sum += 1.0 / (hit_indices[0] + 1)
            
            hit_positions = np.where(y_top10 > 0)[0]
            precisions = np.arange(1, len(hit_positions) + 1) / (hit_positions + 1)
            map10_sum += precisions.sum() / min(10.0, y.sum())
            
    if num_users == 0: return 0.0
    return 0.35 * (mrr_sum / num_users) + 0.30 * (p10_sum / num_users) + 0.20 * (map10_sum / num_users) + 0.15 * (hr10_sum / num_users)

import lightgbm as lgb
def objective(trial):
    param = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'eval_at': 10,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 31, 255),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 50, 1000),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-3, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-3, 10.0, log=True),
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1,
        'feature_pre_filter': False
    }

    model = lgb.train(
        param, lgb_train, num_boost_round=100,
        valid_sets=[lgb_valid], valid_names=['valid'],
        callbacks=[lgb.early_stopping(15, verbose=False)]
    )
    
    preds = model.predict(lgb_valid.data)
    score = compute_hybrid_score(preds, valid_y, valid_groups)
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)

print('==============================================')
print('Best trial:')
trial = study.best_trial
print(f'  Value (Hybrid Score): {trial.value}')
print('  Params: ')
for key, value in trial.params.items():
    print(f'    \"{key}\": {value},')
print('==============================================')
'''

out = CELL_IMPORTS.replace('/kaggle/input/datasets/kinonquc/qkindataset2/', 'd:/CS116/ProjectNumberOne/') + '\n' + CELL_HELPERS + '\n' + cell_data + '\n' + cell_optuna

with open('scratch/local_optuna.py', 'w', encoding='utf-8') as f:
    f.write(out)
print('Created scratch/local_optuna.py')
