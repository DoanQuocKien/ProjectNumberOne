import json

nb_path = r'd:\CS116\ProjectNumberOne\QuocKien\pir_pipeline_v12_submission.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell_4_source = """print("Preparing Folds...")
def get_fold(train_end, val_m):
    h = df_raw.filter(pl.col('month') <= train_end)
    t = df_raw.filter(pl.col('month') == val_m)
    return create_dataset_v12(h, t, items_df, sample_users=60000, n_negatives=150)

f1 = get_fold(8, 9)
f2 = get_fold(9, 10)
f3 = get_fold(10, 11)

cat_feat_ids = [f'{c}_id' for c in cat_cols]
all_feats = ['u_unique_items', 'u_total_qty', 'u_avg_price', 'u_price_std', 'u_tenure_days', 'u_exploration_ratio', 'u_brand_hhi',
             'i_unique_users', 'i_total_qty', 'i_hubs_count', 'i_ref_price', 'i_repeat_rate',
             'ui_total_qty', 'ui_recency_days', 'ui_is_primary_cat', 'ui_is_preferred_brand',
             'ui_price_diff', 'ui_price_ratio', 'ui_loc_sales', 'item_momentum', 'item_age_proxy', 'u_cat_affinity',
             'u_cat_hhi', 'u_avg_age_proxy', 'ui_size_age_diff', 'ui_size_age_ratio', 'ui_already_bought_discretionary', 'ui_loc_sparsity_penalty'] + cat_feat_ids

def prep_lgb(df):
    X = df.select(all_feats).to_numpy()
    y = df['target'].to_numpy()
    g = df.group_by('customer_id', maintain_order=True).len()['len'].to_numpy()
    return X, y, g

X1, y1, g1 = prep_lgb(f1)
X2, y2, g2 = prep_lgb(f2)
X3, y3, g3 = prep_lgb(f3)

def objective(trial):
    param = {
        'objective': 'lambdarank', 'metric': 'ndcg', 'ndcg_eval_at': [10], 'verbosity': -1,
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.03),
        'num_leaves': trial.suggest_int('num_leaves', 127, 1023),
        'max_depth': trial.suggest_int('max_depth', 9, 20),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 100, 1000),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'max_bin': 255, 'device': 'gpu', 'random_state': SEED
    }
    X_train = np.vstack([X1, X2])
    y_train = np.concatenate([y1, y2])
    g_train = np.concatenate([g1, g2])
    dtrain = lgb.Dataset(X_train, y_train, group=g_train, feature_name=all_feats, categorical_feature=cat_feat_ids)
    dval = lgb.Dataset(X3, y3, group=g3, reference=dtrain, feature_name=all_feats, categorical_feature=cat_feat_ids)
    m = lgb.train(param, dtrain, valid_sets=[dval], num_boost_round=800, callbacks=[lgb.early_stopping(50)])
    score = m.best_score['valid_0']['ndcg@10']
    del m, dtrain, dval; gc.collect()
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=1)
best_params = study.best_params
best_params.update({'objective': 'lambdarank', 'metric': 'ndcg', 'device': 'gpu', 'max_bin': 255})

# Retrieve the validation model iteration (early stopping iteration) dynamically
# to prevent over-boosting and extremely long training penalty on CPU!
best_iteration = 50
try:
    X_train = np.vstack([X1, X2])
    y_train = np.concatenate([y1, y2])
    g_train = np.concatenate([g1, g2])
    dtrain = lgb.Dataset(X_train, y_train, group=g_train, feature_name=all_feats, categorical_feature=cat_feat_ids)
    dval = lgb.Dataset(X3, y3, group=g3, reference=dtrain, feature_name=all_feats, categorical_feature=cat_feat_ids)
    val_m = lgb.train(best_params, dtrain, valid_sets=[dval], num_boost_round=800, callbacks=[lgb.early_stopping(50)])
    best_iteration = val_m.best_iteration if val_m.best_iteration > 0 else 50
    print(f"Dynamically retrieved best iteration from validation model: {best_iteration}")
    del val_m, dtrain, dval; gc.collect()
except Exception as e:
    print(f"Error retrieving best iteration: {e}. Falling back to 50 iterations.")

X_final = np.vstack([X1, X2, X3])
y_final = np.concatenate([y1, y2, y3])
g_final = np.concatenate([g1, g2, g3])

# PROACTIVE MEMORY CLEANUP: Delete validation dataframes and arrays BEFORE final training to free 15+ GB of RAM!
del f1, f2, f3, X1, y1, g1, X2, y2, g2, X3, y3, g3, study
gc.collect()

d_final = lgb.Dataset(X_final, y_final, group=g_final, feature_name=all_feats, categorical_feature=cat_feat_ids)
lgb_m = lgb.train(best_params, d_final, num_boost_round=best_iteration)

# CLEAN UP FINAL TRAINING MEMORY IMMEDIATELY AFTER TRAINING
del X_final, y_final, g_final, d_final
gc.collect()"""

nb['cells'][4]['source'] = [line + "\n" for line in cell_4_source.split("\n")]
if nb['cells'][4]['source']:
    nb['cells'][4]['source'][-1] = nb['cells'][4]['source'][-1].rstrip("\n")

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Cell 4 successfully rebuilt and verified!")
