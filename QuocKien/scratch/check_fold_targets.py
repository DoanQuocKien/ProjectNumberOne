import polars as pl
import numpy as np
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

def check_fold(train_end, val_m):
    # Sample 10000 users from history to prevent TruncatedSVD memory error on small-RAM machines
    all_users = df_raw.filter(pl.col('month') <= train_end)['customer_id'].unique()
    np.random.seed(SEED)
    sampled_users = list(np.random.choice(all_users.to_list(), min(15000, len(all_users)), replace=False))
    
    h = df_raw.filter((pl.col('month') <= train_end) & pl.col('customer_id').is_in(sampled_users))
    t = df_raw.filter((pl.col('month') == val_m) & pl.col('customer_id').is_in(sampled_users))
    
    print(f"\n--- Checking Fold: Train End {train_end}, Val Month {val_m} ---")
    print(f"Sampled History size: {h.height}, Unique Users: {h['customer_id'].n_unique()}")
    print(f"Sampled Val Truth size: {t.height}, Unique Users: {t['customer_id'].n_unique()}")
    
    # Generate fold for a slice of 1000 users
    f = create_dataset_v12(h, t, items_df, sample_users=1000, n_negatives=150)
    print(f"Generated fold dataset size: {f.height}")
    
    # Label distribution
    lbls = f['target'].to_list()
    unique_lbls, counts = np.unique(lbls, return_counts=True)
    print(f"Label distribution: {dict(zip(unique_lbls.tolist(), counts.tolist()))}")
    
    # Group sizes
    g = f.group_by('customer_id').len()
    print(f"Number of groups: {len(g)}")
    if len(g) > 0:
        print(f"Max group size: {int(g['len'].max())}, Min group size: {int(g['len'].min())}, Mean group size: {float(g['len'].mean())}")
        
        pos_per_group = f.group_by('customer_id').agg(pl.col('target').sum().alias('pos_count'))
        pos_counts = pos_per_group['pos_count'].to_list()
        pos_uniq, pos_cnts = np.unique(pos_counts, return_counts=True)
        print(f"Positive label distribution per group: {dict(zip(pos_uniq.tolist(), pos_cnts.tolist()))}")
        
        # Verify contiguous groups alignment
        first_group_id = f['customer_id'][0]
        first_group_len = int(g.filter(pl.col('customer_id') == first_group_id)['len'][0])
        first_group_rows = f.head(first_group_len)
        print(f"First parsed group customer: {first_group_id}")
        print(f"First group target array: {first_group_rows['target'].to_list()}")
        print(f"Are all rows in the first contiguous segment belonging to this user? {bool(all(first_group_rows['customer_id'] == first_group_id))}")
    else:
        print("No groups generated!")

check_fold(8, 9)
