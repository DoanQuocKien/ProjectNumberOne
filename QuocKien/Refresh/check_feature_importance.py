import polars as pl
import lightgbm as lgb
from datetime import datetime
import numpy as np
import new_pir_lgbm_v3 as v3

print("Loading items...")
items = v3.load_items()

print("Extracting small dataset...")
df = v3.extract_dataset(datetime(2025, 11, 1), datetime(2025, 12, 1), 3000, items)

print("Preparing for LightGBM...")
features = v3.extract_feature_names(df)
cat_cols = [c for c in v3.CATEGORICAL if c in features]

X = df.select(features).to_pandas()
y = df["label"].to_pandas()
q = df.group_by("customer_id", maintain_order=True).len()["len"].to_numpy()

lgb_train = lgb.Dataset(X, label=y, group=q, categorical_feature=cat_cols)
params = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'learning_rate': 0.1,
    'num_leaves': 31,
    'verbose': -1,
    'random_state': 42
}
print("Training quick model...")
model = lgb.train(params, lgb_train, num_boost_round=100)

imps = model.feature_importance(importance_type='gain')
feat_imp = sorted(zip(features, imps), key=lambda x: x[1], reverse=True)

print("\n--- TOP 20 FEATURE IMPORTANCES (GAIN) ---")
for i, (f, imp) in enumerate(feat_imp[:20]):
    print(f"{i+1:02d}. {f}: {imp:.2f}")

print("\n--- TARGET FEATURES ---")
targets = ["basket_size_mismatch", "weekend_shopper_match", "replenishment_overdue_days"]
for f, imp in feat_imp:
    for t in targets:
        if t in f:
            rank = [x[0] for x in feat_imp].index(f) + 1
            print(f"Rank {rank}: {f} (Gain: {imp:.2f})")
