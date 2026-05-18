import polars as pl

T_PATH = r'd:\CS116\ProjectNumberOne\transaction_full_2025.parquet'

print("Loading data...")
df_raw = pl.read_parquet(T_PATH).select([
    pl.col('customer_id').cast(pl.Int64),
    pl.col('item_id').cast(pl.Utf8),
    pl.col('updated_date').cast(pl.Datetime).alias('event_ts')
]).with_columns(pl.col('event_ts').dt.month().alias('month'))

# Train is months <= 11, Test is month 12
hist = df_raw.filter(pl.col('month') <= 11)
test = df_raw.filter(pl.col('month') == 12)

# Sample 30,000 active users in month 12
test_users = test['customer_id'].unique().shuffle(seed=42).head(30000).to_list()
truth_df = test.filter(pl.col('customer_id').is_in(test_users))
truth_dict = {row[0]: set(row[1]) for row in truth_df.group_by('customer_id').agg(pl.col('item_id')).iter_rows()}

# Top global popularity items in Month 11 (November)
nov_pop = df_raw.filter(pl.col('month') == 11).group_by('item_id').len().sort('len', descending=True).head(10)['item_id'].to_list()
print("Top 10 Nov items:", nov_pop)

hits = 0
total_users = 0

for uid in test_users:
    if uid not in truth_dict:
        continue
    total_users += 1
    truth = truth_dict[uid]
    
    # Recommend top global popularity items
    recs = nov_pop
    hits += sum(1 for r in recs if r in truth)

prec = hits / (total_users * 10)
print(f"November Popularity Baseline on Dec: Precision@10 = {prec:.6f}, Hits = {hits}, Users = {total_users}")
