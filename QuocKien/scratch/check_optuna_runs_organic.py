import polars as pl
import numpy as np
import lightgbm as lgb
import warnings
import gc
import re

pl.enable_string_cache()
warnings.filterwarnings('ignore')

SEED = 42
T_PATH = r'd:\CS116\ProjectNumberOne\transaction_full_2025.parquet'
I_PATH = r'd:\CS116\ProjectNumberOne\items.parquet'

print("Loading Data...")
items_raw = pl.read_parquet(I_PATH)
item_mapping = items_raw.select('item_id').unique().with_row_index("item_int_id")

items_df = items_raw.join(item_mapping, on='item_id', how='left').drop('item_id').rename({'item_int_id': 'item_id'}).select([
    pl.col('item_id').cast(pl.Int32),
    pl.col('category').cast(pl.Utf8),
    pl.col('category_l1').cast(pl.Utf8),
    pl.col('category_l2').cast(pl.Utf8),
    pl.col('category_l3').cast(pl.Utf8),
    pl.col('brand').cast(pl.Utf8),
    pl.col('size').cast(pl.Utf8)
])

df_raw = pl.read_parquet(T_PATH).join(item_mapping, on='item_id', how='inner').drop('item_id').rename({'item_int_id': 'item_id'}).select([
    pl.col('customer_id').cast(pl.Int64),
    pl.col('item_id').cast(pl.Int32),
    pl.col('quantity').cast(pl.Int32),
    pl.col('price').cast(pl.Float32),
    pl.col('location').cast(pl.Int16),
    pl.col('updated_date').cast(pl.Datetime).alias('event_ts')
]).drop_nulls(subset=['item_id', 'customer_id']).with_columns([
    pl.col('event_ts').dt.month().alias('month').cast(pl.Int8)
])

cat_cols = ['category', 'category_l1', 'category_l2', 'category_l3', 'brand']
for c in cat_cols:
    items_df = items_df.with_columns(pl.col(c).fill_null('Unknown'))
    top_vals = items_df[c].value_counts().sort('count', descending=True).head(254)[c].to_list()
    items_df = items_df.with_columns(
        pl.when(pl.col(c).is_in(top_vals)).then(pl.col(c)).otherwise(pl.lit('Other')).alias(c)
    )
    items_df = items_df.with_columns(pl.col(c).cast(pl.Categorical).to_physical().cast(pl.Int32).alias(f"{c}_id"))

def standardize_age(text):
    raw_text = str(text).strip()
    clean_text = raw_text.lower()
    if re.search(r'(\\*|x\\d|cm)', clean_text): return 0.5
    if re.search(r'\\bb\\d{2}\\b', clean_text): return 18.0
    if 's17' in clean_text: return 1.0
    if '110' in clean_text: return 5.0
    if "không xác định" in clean_text or not clean_text: return -1.0
    diaper_map = {
        r'\\bnb\\b': 0.0, r'\\bss\\b': 0.0, r'\\bsơ sinh\\b': 0.0,
        r'\\bs\\b': 0.25, r'\\bm\\b': 0.6, r'\\bl\\b': 1.2,
        r'\\bxl\\b': 2.0, r'\\bxxl\\b': 3.5
    }
    for pattern, val in diaper_map.items():
        if re.search(pattern, clean_text): return val
    range_match = re.search(r'(\\d+\\.?\\d*)\\s*-\\s*(\\d+\\.?\\d*)', clean_text)
    if range_match:
        s, e = float(range_match.group(1)), float(range_match.group(2))
        avg = (s + e) / 2
        if any(x in clean_text for x in ['m', 'tháng']): return round(avg / 12, 3)
        return avg
    m_match = re.search(r'(\\d+\\.?\\d*)\\s*(m|tháng)', clean_text)
    if m_match: return round(float(m_match.group(1)) / 12, 3)
    y_match = re.search(r'(\\d+\\.?\\d*)\\s*(y|t|tuổi)', clean_text)
    if y_match: return float(y_match.group(1))
    pure_num = re.search(r'^(\\d+)$', clean_text)
    if pure_num:
        val = float(pure_num.group(1))
        return round(val/12, 3) if val > 6 else val
    return -1.0

size_map = {row[0]: standardize_age(row[1]) for row in items_df.select(['item_id', 'size']).iter_rows()}
items_df = items_df.with_columns(pl.col('item_id').replace(size_map, default=-1.0).cast(pl.Float32).alias('item_age_proxy'))

# Standalone imports to run clean
from test_fold_creation_int import create_dataset_v12

def check_fold_details(train_end, val_m):
    all_users = df_raw.filter(pl.col('month') <= train_end)['customer_id'].unique()
    np.random.seed(SEED)
    sampled_users = list(np.random.choice(all_users.to_list(), min(15000, len(all_users)), replace=False))
    
    h = df_raw.filter((pl.col('month') <= train_end) & pl.col('customer_id').is_in(sampled_users))
    t = df_raw.filter((pl.col('month') == val_m) & pl.col('customer_id').is_in(sampled_users))
    
    f = create_dataset_v12(h, t, items_df, sample_users=2000, n_negatives=150)
    return f

print("Generating validation datasets...")
f1 = check_fold_details(8, 9)
f2 = check_fold_details(9, 10)
f3 = check_fold_details(10, 11)

cat_feat_ids = [f'{c}_id' for c in cat_cols]
all_feats = [c for c in f1.columns if c not in ['customer_id', 'item_id', 'target']]

def prep_lgb(df):
    X = df.select(all_feats).to_numpy()
    y = df['target'].to_numpy()
    g = df.group_by('customer_id', maintain_order=True).len()['len'].to_numpy()
    return X, y, g

X1, y1, g1 = prep_lgb(f1)
X2, y2, g2 = prep_lgb(f2)
X3, y3, g3 = prep_lgb(f3)

print("Preparing real LightGBM evaluation simulation using user's parameters...")
param = {
    'objective': 'lambdarank', 'metric': 'ndcg', 'ndcg_eval_at': [10], 'verbosity': -1,
    'learning_rate': 0.01760271383272208,
    'num_leaves': 999,
    'max_depth': 20,
    'min_data_in_leaf': 744,
    'lambda_l1': 6.378936681108256e-05,
    'lambda_l2': 1.5713381755837366,
    'max_bin': 255, 'random_state': SEED
}
X_train = np.vstack([X1, X2])
y_train = np.concatenate([y1, y2])
g_train = np.concatenate([g1, g2])
dtrain = lgb.Dataset(X_train, y_train, group=g_train, feature_name=all_feats, categorical_feature=cat_feat_ids)
dval = lgb.Dataset(X3, y3, group=g3, reference=dtrain, feature_name=all_feats, categorical_feature=cat_feat_ids)

# Train a real model
m = lgb.train(param, dtrain, valid_sets=[dval], num_boost_round=100, callbacks=[lgb.early_stopping(15)])
score = m.best_score['valid_0']['ndcg@10']
print(f"\n--- [TRAINING RUN COMPLETE] Metric: {score} ---")
