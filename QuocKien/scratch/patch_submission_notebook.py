import json

nb_path = r'd:\CS116\ProjectNumberOne\QuocKien\pir_pipeline_v12_submission.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the cell that contains "lgb_m = lgb.train(best_params, d_final, num_boost_round=1200)"
for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "num_boost_round=1200" in source:
            print(f"Found target cell at index {idx}!")
            
            new_source = """study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=1)
best_params = study.best_params
best_params.update({'objective': 'lambdarank', 'metric': 'ndcg', 'device': 'gpu', 'max_bin': 255})

# Retrieve the validation model iteration (early stopping iteration) dynamically
# to prevent over-boosting and extremely long training penalty on CPU!
best_iteration = 50
try:
    # We can get the best_iteration from the validation study or optuna trial, or simple callback
    # Since we only run 1 trial, we can train a validation model with the best parameters to get best_iteration
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
            
            cell['source'] = [line + "\n" for line in new_source.split("\n")]
            # Remove trailing newline from last element
            if cell['source']:
                cell['source'][-1] = cell['source'][-1].rstrip("\n")

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook successfully patched!")
