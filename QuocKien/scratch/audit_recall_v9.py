import polars as pl
import numpy as np

T_PATH = '../../transaction_full_2025.parquet'
I_PATH = '../../items.parquet'

df_raw = pl.read_parquet(T_PATH).select([
    pl.col('customer_id').cast(pl.Int64),
    pl.col('item_id').cast(pl.Utf8),
    pl.col('updated_date').cast(pl.Datetime).alias('event_ts')
]).with_columns(pl.col('event_ts').dt.month().alias('month'))

# 1. Get Truth for December
truth = df_raw.filter(pl.col('month') == 12).select(['customer_id', 'item_id']).unique()
test_users = truth['customer_id'].unique().head(20000).to_list()
truth = truth.filter(pl.col('customer_id').is_in(test_users))

# 2. Simulate V8-style Simple Retriever (History + Global Hot)
hist = df_raw.filter(pl.col('month') <= 11).filter(pl.col('customer_id').is_in(test_users))
rep_cands = hist.select(['customer_id', 'item_id']).unique()
hot_cands = df_raw.filter((pl.col('month') == 11)).group_by('item_id').len().sort('len', descending=True).head(50).select('item_id')
v8_cands = pl.concat([rep_cands, pl.DataFrame({'customer_id': test_users}).join(hot_cands.with_columns(pl.lit(1).alias('_k')), how='cross').drop('_k')]).unique()

# 3. Check V8 Recall
v8_hits = v8_cands.join(truth, on=['customer_id', 'item_id'], how='inner').height
v8_recall = v8_hits / truth.height

print(f"V8-style Recall (History + Hot): {v8_recall:.4f}")
print(f"Total Users: {len(test_users)}, Total Truth Rows: {truth.height}")
