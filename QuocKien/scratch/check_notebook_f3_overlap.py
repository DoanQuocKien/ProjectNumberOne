import polars as pl
import numpy as np

SEED = 42
T_PATH = r'd:\CS116\ProjectNumberOne\transaction_full_2025.parquet'
I_PATH = r'd:\CS116\ProjectNumberOne\items.parquet'

print("Loading Data...")
items_raw = pl.read_parquet(I_PATH)
item_mapping = items_raw.select('item_id').unique().with_row_index("item_int_id")

df_raw = pl.read_parquet(T_PATH).join(item_mapping, on='item_id', how='inner').drop('item_id').rename({'item_int_id': 'item_id'}).select([
    pl.col('customer_id').cast(pl.Int64),
    pl.col('item_id').cast(pl.Int32),
    pl.col('updated_date').cast(pl.Datetime).alias('event_ts')
]).drop_nulls(subset=['item_id', 'customer_id']).with_columns([
    pl.col('event_ts').dt.month().alias('month').cast(pl.Int8)
])

# Month 10 vs Month 11
h = df_raw.filter(pl.col('month') <= 10)
t = df_raw.filter(pl.col('month') == 11)

print(f"History Users: {h['customer_id'].n_unique()}")
print(f"Validation Truth Users: {t['customer_id'].n_unique()}")

# Sample 60000 users from history exactly as in the notebook
valid_u = h['customer_id'].unique().shuffle(seed=SEED).head(60000).to_list()

# Check overlap
truth_users = t['customer_id'].unique().to_list()
overlap = set(valid_u).intersection(set(truth_users))

print(f"Overlap size: {len(overlap)}")
print(f"Overlap ratio: {len(overlap) / 60000 * 100:.4f}%")
