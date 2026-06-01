import json
from pathlib import Path


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(True)}


pipeline_source = Path("QuocKien/pir_pipeline_v13_direct_upgrade.py").read_text(encoding="utf-8")
pipeline_source = pipeline_source.replace(
    'ROOT = Path(__file__).resolve().parents[1]',
    'ROOT = Path.cwd()',
)
pipeline_source = pipeline_source.replace(
    'if __name__ == "__main__":\n    main()\n',
    '# Notebook embed: CLI entrypoint disabled.\n',
)


cells = [
    md(
        """# PIR Pipeline V13 Direct Upgrade Lab

This notebook is a direct upgrade from `pir_pipeline_v12.ipynb`, not a throwaway heuristic baseline.

What stays from v12:
- Two-stage recommender: retrieval -> feature matrix -> LightGBM LambdaRank -> Top 10.
- Core retrieval: history, replenishment, global, local, SVD, I2I, category top.
- Core features: user, item, user-item, category/brand, location, momentum, size-age proxy, repeat/discretionary penalties.

What v13 adds from `analytical_insights_master.md`:
- Candidate-selection upgrades for top-3 category discovery, local-category heroes, brand-in-category candidates, momentum candidates, stronger replenishment.
- V13.1 retrieval policy: segment-aware source budgets for habituals, active discoverers, and hibernators.
- V13.1 replenishment: user-item gaps backed off to item/category median repeat gaps, so single-purchase staples can still become due candidates.
- Source-aware features: `src_*`, `rank_*`, `retr_score`, and `retr_source_count`.
- Fold-by-fold candidate diagnostics before model training, so we can tune the chokehold directly.

Important clarification: v12 does **not** leak Month 12 into model training. The distinction here is evaluation mode:
- `EVAL_APPEND_MISSING_POSITIVES=True`: v12-comparable candidate set evaluation.
- `EVAL_APPEND_MISSING_POSITIVES=False`: strict end-to-end retrieval evaluation.
"""
    ),
    md("## 0. Embedded Pipeline Code\n\nThis notebook is self-contained for Kaggle. The full v13 pipeline code is written directly into this cell, with no import from `pir_pipeline_v13_direct_upgrade.py`.\n"),
    code(pipeline_source),
    code(
        """try:
    import lightgbm as lgb
except ImportError as exc:
    raise ImportError('Install LightGBM in this environment first: pip install lightgbm') from exc

pl.Config.set_tbl_rows(30)
pl.Config.set_tbl_cols(30)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print('Resolved transaction path:', TRANSACTION_PATH)
print('Resolved event path:', EVENT_PATH)
print('Resolved items path:', ITEMS_PATH)
print('Resolved output dir:', OUTPUT_DIR)
assert TRANSACTION_PATH.exists(), f'Missing transaction parquet: {TRANSACTION_PATH}'
assert ITEMS_PATH.exists(), f'Missing items parquet: {ITEMS_PATH}'
"""
    ),
    md(
        """## 1. Tuning Config

These defaults keep the v12-scale comparison intact. The notebook is optimized by keeping data in Polars until the LightGBM boundary and releasing fold DataFrames after conversion.

`INCLUDE_CF` is off by default because SVD/I2I is the largest candidate-generation RAM risk. The non-CF retrieval channels stay full-size.
"""
    ),
    code(
        """TRAIN_SAMPLE_USERS = DEFAULT_TRAIN_SAMPLE_USERS
EVAL_SAMPLE_USERS = DEFAULT_EVAL_SAMPLE_USERS
N_NEGATIVES = DEFAULT_N_NEGATIVES
INCLUDE_CF = DEFAULT_INCLUDE_CF
USE_EVENTS = DEFAULT_USE_EVENTS
LIGHTGBM_DEVICE = DEFAULT_LIGHTGBM_DEVICE
CF_CHUNK_SIZE = DEFAULT_CF_CHUNK_SIZE
CF_MAX_USERS = DEFAULT_CF_MAX_USERS
CF_MAX_ITEMS = DEFAULT_CF_MAX_ITEMS

# True = v12-comparable. False = strict retrieval-only eval.
EVAL_APPEND_MISSING_POSITIVES = DEFAULT_EVAL_APPEND_MISSING_POSITIVES

LGB_PARAMS = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'ndcg_eval_at': [10],
    'learning_rate': DEFAULT_LEARNING_RATE,
    'num_leaves': DEFAULT_NUM_LEAVES,
    'max_depth': DEFAULT_MAX_DEPTH,
    'min_data_in_leaf': DEFAULT_MIN_DATA_IN_LEAF,
    'lambda_l1': DEFAULT_LAMBDA_L1,
    'lambda_l2': DEFAULT_LAMBDA_L2,
    'verbosity': -1,
    'random_state': SEED,
}
if LIGHTGBM_DEVICE != 'cpu':
    LGB_PARAMS.update({'device': LIGHTGBM_DEVICE, 'max_bin': 255})

NUM_BOOST_ROUND = DEFAULT_NUM_BOOST_ROUND
EARLY_STOPPING_ROUNDS = DEFAULT_EARLY_STOPPING_ROUNDS
"""
    ),
    md("## 2. Load Data\n\nAll feature/candidate processing is Polars. LightGBM receives NumPy arrays only at the final ranker boundary.\n"),
    code(
        """df_raw = load_transactions(TRANSACTION_PATH)
items_df = load_items(ITEMS_PATH)

print('transactions:', df_raw.shape)
print('items:', items_df.shape)
print(df_raw.group_by('month').agg(
    pl.len().alias('rows'),
    pl.col('customer_id').n_unique().alias('users'),
    pl.col('item_id').n_unique().alias('items'),
).sort('month'))
"""
    ),
    md("## 3. Helper: Pretty Candidate Diagnostics\n\nThis is the key v13 tuning panel. If candidate recall or source hit contribution does not improve, the ranker will not save us.\n"),
    code(
        """def print_candidate_report(name, ds, truth_df):
    report = candidate_diagnostics(ds, truth_df, top_k=10)
    compact = {k: v for k, v in report.items() if k != 'source_metrics'}
    print(f'\\n===== {name}: Candidate Diagnostics =====')
    print(json.dumps(compact, indent=2))
    source_df = pl.DataFrame(report['source_metrics']).sort('hit_pairs', descending=True)
    print(source_df)
    return report, source_df

all_reports = {}
"""
    ),
    md(
        """## 4. Build Temporal Folds And Print Candidate Metrics

Fold design is inherited from v12:
- train <= M8 -> label M9
- train <= M9 -> label M10
- train <= M10 -> label M11

Training mode appends missing positives for supervised ranker learning, matching the v12 style. The diagnostics count only rows retrieved by real sources (`retr_source_count > 0`), so candidate recall stays honest even when appended positives are present for learning.
"""
    ),
    code(
        """def build_fold_arrays(train_end, val_month, name):
    print(f'Building {name}: history <= M{train_end}, truth = M{val_month}')
    h = df_raw.filter(pl.col('month') <= train_end)
    t = df_raw.filter(pl.col('month') == val_month)
    ds = create_dataset_v13(
        history_df=h,
        truth_df=t,
        items_df=items_df,
        sample_users=TRAIN_SAMPLE_USERS,
        n_negatives=N_NEGATIVES,
        include_cf=INCLUDE_CF,
        include_events=USE_EVENTS,
        event_path=EVENT_PATH,
        mode='train',
        append_missing_positives=True,
        cf_chunk_size=CF_CHUNK_SIZE,
        cf_max_users=CF_MAX_USERS,
        cf_max_items=CF_MAX_ITEMS,
    )
    report, source_df = print_candidate_report(name, ds, t)
    all_reports[name] = report
    print('dataset shape:', ds.shape)
    print('positive rate:', float(ds['target'].mean()))
    arrays = prep_lgb(ds)
    del ds, h, t
    gc.collect()
    return arrays

(X1, y1, g1) = build_fold_arrays(8, 9, 'fold1_M9')
(X2, y2, g2) = build_fold_arrays(9, 10, 'fold2_M10')
(X3, y3, g3) = build_fold_arrays(10, 11, 'fold3_M11')
"""
    ),
    md("## 5. Train LambdaRank And Show Validation NDCG\n\nThe ranker is still v12-style LightGBM LambdaRank. V13 adds source-aware features, so feature importance becomes more informative.\n"),
    code(
        """X_train = np.vstack([X1, X2])
y_train = np.concatenate([y1, y2])
g_train = np.concatenate([g1, g2])
cat_idx = [ALL_FEATURES.index(c) for c in CAT_FEATURES if c in ALL_FEATURES]

dtrain = lgb.Dataset(X_train, y_train, group=g_train, categorical_feature=cat_idx)
dval = lgb.Dataset(X3, y3, group=g3, categorical_feature=cat_idx, reference=dtrain)

model = lgb.train(
    LGB_PARAMS,
    dtrain,
    valid_sets=[dval],
    num_boost_round=NUM_BOOST_ROUND,
    callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS), lgb.log_evaluation(25)],
)
print('best_iteration:', model.best_iteration)
print('best_score:', model.best_score)
del dtrain, dval, X_train, y_train, g_train
gc.collect()
"""
    ),
    md("## 6. Feature Importance\n\nCheck whether v13 retrieval/source features are being used. If source features are dead, candidate-generation improvements are not reaching the ranker.\n"),
    code(
        """importance = pl.DataFrame({
    'feature': ALL_FEATURES,
    'gain': model.feature_importance(importance_type='gain'),
    'split': model.feature_importance(importance_type='split'),
}).sort('gain', descending=True)
print(importance.head(50))
print('\\nSource feature importance:')
print(importance.filter(
    pl.col('feature').str.starts_with('src_')
    | pl.col('feature').str.starts_with('rank_')
    | pl.col('feature').is_in(['retr_score', 'retr_best_rank', 'retr_source_count'])
).head(80))
"""
    ),
    md("## 7. Retrain Final Model On Folds 1-3\n\nThis mirrors v12: after tuning on temporal folds, refit on all fold data before Month 12 evaluation.\n"),
    code(
        """X_final = np.vstack([X1, X2, X3])
y_final = np.concatenate([y1, y2, y3])
g_final = np.concatenate([g1, g2, g3])
final_rounds = model.best_iteration or NUM_BOOST_ROUND

final_data = lgb.Dataset(X_final, y_final, group=g_final, categorical_feature=cat_idx)
final_model = lgb.train(LGB_PARAMS, final_data, num_boost_round=final_rounds)

model_path = OUTPUT_DIR / 'v13_direct_lgbm_from_notebook.txt'
final_model.save_model(str(model_path))
print('saved:', model_path)

del X1, X2, X3, y1, y2, y3, g1, g2, g3
del X_final, y_final, g_final, final_data
gc.collect()
"""
    ),
    md(
        """## 8. Month 12 Evaluation With Candidate Diagnostics

This is the main comparison cell.

- With `EVAL_APPEND_MISSING_POSITIVES=True`, this is v12-comparable.
- With `False`, this is strict retrieval-only end-to-end evaluation.
"""
    ),
    code(
        """m12_history = df_raw.filter(pl.col('month') <= 11)
m12_truth = df_raw.filter(pl.col('month') == 12)

test_set = create_dataset_v13(
    history_df=m12_history,
    truth_df=m12_truth,
    items_df=items_df,
    sample_users=EVAL_SAMPLE_USERS,
    n_negatives=None,
    include_cf=INCLUDE_CF,
    include_events=USE_EVENTS,
    event_path=EVENT_PATH,
    mode='train' if EVAL_APPEND_MISSING_POSITIVES else 'eval',
    append_missing_positives=EVAL_APPEND_MISSING_POSITIVES,
    cf_chunk_size=CF_CHUNK_SIZE,
    cf_max_users=CF_MAX_USERS,
    cf_max_items=CF_MAX_ITEMS,
)
report, source_df = print_candidate_report('month12_eval', test_set, m12_truth)

X_ts, y_ts, g_ts = prep_lgb(test_set)
test_set = test_set.with_columns(pl.Series('pred', final_model.predict(X_ts)))
metrics = evaluate_scored(test_set, m12_truth, pred_col='pred', top_k=10)
metrics.update({
    'eval_append_missing_positives': EVAL_APPEND_MISSING_POSITIVES,
    'include_cf': INCLUDE_CF,
    'use_events': USE_EVENTS,
    'eval_sample_users': EVAL_SAMPLE_USERS,
})
print(json.dumps(metrics, indent=2))

metrics_path = OUTPUT_DIR / 'v13_direct_notebook_m12_metrics.json'
metrics_path.write_text(json.dumps({'metrics': metrics, 'candidate_report': report}, indent=2), encoding='utf-8')
print('saved metrics:', metrics_path)
"""
    ),
    md("## 9. Optional: Compare Strict vs V12-Comparable Candidate Evaluation\n\nRun this after the main evaluation if you want both views in one notebook. It rebuilds only the Month 12 candidate set, not the model.\n"),
    code(
        """def run_m12_mode(eval_append_missing):
    mode_name = 'v12_comparable' if eval_append_missing else 'strict_retrieval'
    ds = create_dataset_v13(
        history_df=m12_history,
        truth_df=m12_truth,
        items_df=items_df,
        sample_users=EVAL_SAMPLE_USERS,
        n_negatives=None,
        include_cf=INCLUDE_CF,
        include_events=USE_EVENTS,
        event_path=EVENT_PATH,
        mode='train' if eval_append_missing else 'eval',
        append_missing_positives=eval_append_missing,
        cf_chunk_size=CF_CHUNK_SIZE,
        cf_max_users=CF_MAX_USERS,
        cf_max_items=CF_MAX_ITEMS,
    )
    report, _ = print_candidate_report(f'month12_{mode_name}', ds, m12_truth)
    X, _, _ = prep_lgb(ds)
    ds = ds.with_columns(pl.Series('pred', final_model.predict(X)))
    score = evaluate_scored(ds, m12_truth)
    score['mode'] = mode_name
    return score, report

# Uncomment to run both modes.
# strict_score, strict_report = run_m12_mode(False)
# comparable_score, comparable_report = run_m12_mode(True)
# print(pl.DataFrame([strict_score, comparable_score]))
"""
    ),
    md("## 10. Export All Known Transaction Customers\n\nDo this on high-RAM only. The CLI export is recommended for the final artifact.\n"),
    code(
        """print(f'''python -X utf8 QuocKien/pir_pipeline_v13_direct_upgrade.py \\
  --mode export-all \\
  --model-path {model_path} \\
  --confirm-large-export \\
  --export-batch-users {DEFAULT_EXPORT_BATCH_USERS} \\
  --output-name v13_direct_all_users_recommendations.pkl''')
"""
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path("QuocKien/pir_pipeline_v13_direct_upgrade.ipynb")
out.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(out)
