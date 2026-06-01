import polars as pl

T_PATH = r'd:\CS116\ProjectNumberOne\transaction_full_2025.parquet'

print("Loading data...")
df_raw = pl.read_parquet(T_PATH).select([
    pl.col('customer_id').cast(pl.Int64),
    pl.col('item_id').cast(pl.Utf8),
    pl.col('updated_date').cast(pl.Datetime).alias('event_ts')
]).with_columns(pl.col('event_ts').dt.month().alias('month'))

# Train/Test split
hist = df_raw.filter(pl.col('month') <= 11)
test = df_raw.filter(pl.col('month') == 12)

# Sample 30,000 active users in month 12
test_users = test['customer_id'].unique().shuffle(seed=42).head(30000).to_list()
truth_df = test.filter(pl.col('customer_id').is_in(test_users))
truth_dict = {row[0]: set(row[1]) for row in truth_df.group_by('customer_id').agg(pl.col('item_id')).iter_rows()}

# Top global popularity items in months <= 11 as fallback
global_pop = hist.group_by('item_id').len().sort('len', descending=True).head(10)['item_id'].to_list()

# Repurchase profile: for each user, count their purchases per item
user_items = hist.filter(pl.col('customer_id').is_in(test_users)).group_by(['customer_id', 'item_id']).len().sort(['customer_id', 'len'], descending=[False, True])
user_recs = user_items.group_by('customer_id').agg(pl.col('item_id'))

recs_dict = {row[0]: list(row[1]) for row in user_recs.iter_rows()}

hits = 0
total_users = 0

for uid in test_users:
    if uid not in truth_dict:
        continue
    total_users += 1
    truth = truth_dict[uid]
    
    # Recommend top purchased items, pad with global popularity if < 10
    recs = recs_dict.get(uid, [])[:10]
    if len(recs) < 10:
        for item in global_pop:
            if item not in recs:
                recs.append(item)
            if len(recs) == 10:
                break
                
    hits += sum(1 for r in recs if r in truth)

prec = hits / (total_users * 10)
print(f"Pure Repurchase Baseline: Precision@10 = {prec:.6f}, Hits = {hits}, Users = {total_users}")
