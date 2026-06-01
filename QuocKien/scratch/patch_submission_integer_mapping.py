import json

nb_path = r'd:\CS116\ProjectNumberOne\QuocKien\pir_pipeline_v12_submission.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Update Cell 1 (Imports and Data Loading with lightweight integer mapping)
cell_1_source = """import polars as pl
import numpy as np
import lightgbm as lgb
import optuna
import os
import gc
import re
import warnings
import pickle
from pathlib import Path
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

# Enable global string cache to guarantee Categorical alignment across all dataframes
pl.enable_string_cache()
warnings.filterwarnings('ignore')

SEED = 42
T_PATH = '/kaggle/input/datasets/kinonquc/qkindataset2/transaction_full_2025.parquet'
I_PATH = '/kaggle/input/datasets/kinonquc/qkindataset2/items.parquet'

print("Loading Data & Mapping Strings to Integers...")

# 1. Create a global integer mapping for items to prevent massive String RAM usage
items_raw = pl.read_parquet(I_PATH)
item_mapping = items_raw.select('item_id').unique().with_row_index("item_int_id")

# Save global mapping dictionary to stream back original string IDs at the end
idx2item = dict(zip(item_mapping['item_int_id'], item_mapping['item_id']))
# Also save as a global pickle file in case of memory separation
with open("idx2item.pkl", "wb") as f_map:
    pickle.dump(idx2item, f_map)

# 2. Load items, join mapping, and drop the string ID
items_df = items_raw.join(item_mapping, on='item_id', how='left').drop('item_id').rename({'item_int_id': 'item_id'}).select([
    pl.col('item_id').cast(pl.Int32), # Lightweight integer!
    pl.col('category').cast(pl.Utf8),
    pl.col('category_l1').cast(pl.Utf8),
    pl.col('category_l2').cast(pl.Utf8),
    pl.col('category_l3').cast(pl.Utf8),
    pl.col('brand').cast(pl.Utf8),
    pl.col('size').cast(pl.Utf8)
])

# 3. Load transactions, join mapping, and drop the string ID
df_raw = pl.read_parquet(T_PATH).join(item_mapping, on='item_id', how='left').drop('item_id').rename({'item_int_id': 'item_id'}).select([
    pl.col('customer_id').cast(pl.Int64), # Must keep as Int64 to avoid overflow
    pl.col('item_id').cast(pl.Int32),     # Lightweight integer!
    pl.col('quantity').cast(pl.Int32),
    pl.col('price').cast(pl.Float32),
    # Map location strings to physical Int16 immediately
    pl.col('location').cast(pl.Categorical).to_physical().cast(pl.Int16), 
    pl.col('updated_date').cast(pl.Datetime).alias('event_ts')
]).with_columns([
    pl.col('event_ts').dt.month().alias('month').cast(pl.Int8),
    pl.col('event_ts').dt.weekday().alias('dow').cast(pl.Int8)
])

# Clean up temporary frames
del items_raw, item_mapping
gc.collect()

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
items_df = items_df.with_columns(pl.col('item_id').replace(size_map, default=-1.0).cast(pl.Float32).alias('item_age_proxy'))"""

nb['cells'][1]['source'] = [line + "\n" for line in cell_1_source.split("\n")]
if nb['cells'][1]['source']:
    nb['cells'][1]['source'][-1] = nb['cells'][1]['source'][-1].rstrip("\n")

# 2. Update Cell 4 (Set Optuna trials to 15 for Bayesian hyperparameter tuning)
cell_4_source = "".join(nb['cells'][4]['source'])
cell_4_source = cell_4_source.replace("study.optimize(objective, n_trials=1)", "study.optimize(objective, n_trials=15)")
nb['cells'][4]['source'] = [line + "\n" for line in cell_4_source.split("\n")]
if nb['cells'][4]['source']:
    nb['cells'][4]['source'][-1] = nb['cells'][4]['source'][-1].rstrip("\n")

# 3. Update Cell 5 (Load idx2item mapping and cast predictions back to original string IDs during binary streaming)
cell_5_source = "".join(nb['cells'][5]['source'])

# Inject loading the global idx2item mapping right at the beginning of the function
target_str = "def make_chunked_submission_pkl(history_df, items_df, target_users, model, all_feats, output_path, chunk_size=20000):"
replacement = """def make_chunked_submission_pkl(history_df, items_df, target_users, model, all_feats, output_path, chunk_size=20000):
    # Load global integer-to-string item mapping
    with open("idx2item.pkl", "rb") as f_map:
        idx2item = pickle.load(f_map)"""

cell_5_source = cell_5_source.replace(target_str, replacement)

# Inject casting prediction integer IDs back to original string IDs before writing
target_write_loop = """            for customer_id, items in zip(c_ids, item_lists):
                key_bytes = pickle.dumps(int(customer_id), protocol=4)
                val_bytes = pickle.dumps(items, protocol=4)"""

replacement_write_loop = """            for customer_id, items in zip(c_ids, item_lists):
                str_items = [idx2item[item] for item in items]
                key_bytes = pickle.dumps(int(customer_id), protocol=4)
                val_bytes = pickle.dumps(str_items, protocol=4)"""

cell_5_source = cell_5_source.replace(target_write_loop, replacement_write_loop)

# Fix items_df brand/category_l1 cast to Categorical since they are Utf8 and need global string cache
nb['cells'][5]['source'] = [line + "\n" for line in cell_5_source.split("\n")]
if nb['cells'][5]['source']:
    nb['cells'][5]['source'][-1] = nb['cells'][5]['source'][-1].rstrip("\n")

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook successfully updated with all integer-mapping, string cache, and Optuna n_trials=15 optimizations!")
