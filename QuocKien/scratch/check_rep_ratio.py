import polars as pl

T_PATH = r'd:\CS116\ProjectNumberOne\transaction_full_2025.parquet'
df_raw = pl.read_parquet(T_PATH).select([
    pl.col('customer_id').cast(pl.Int64),
    pl.col('item_id').cast(pl.Utf8),
    pl.col('updated_date').cast(pl.Datetime).alias('event_ts')
]).with_columns(pl.col('event_ts').dt.month().alias('month'))

# Train/Test split
hist = df_raw.filter(pl.col('month') <= 11)
test = df_raw.filter(pl.col('month') == 12)

# Users who made purchases in December
test_users = test['customer_id'].unique().to_list()
print(f"Total active users in Dec: {len(test_users)}")

# Truth pairs in December
truth = test.select(['customer_id', 'item_id']).unique()
print(f"Total truth pairs in Dec: {truth.height}")

# Seen pairs in history (months <= 11)
seen = hist.filter(pl.col('customer_id').is_in(test_users)).select(['customer_id', 'item_id']).unique()
print(f"Total seen pairs in history for Dec active users: {seen.height}")

# Hits (purchases in Dec of items already purchased before)
hits = truth.join(seen, on=['customer_id', 'item_id'], how='inner')
print(f"Repeat purchase hits in Dec: {hits.height}")
print(f"Percentage of Dec purchases that are repeats: {hits.height / truth.height:.2%}")
