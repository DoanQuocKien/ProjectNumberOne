import polars as pl
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

T_PATH = r'd:\CS116\ProjectNumberOne\transaction_full_2025.parquet'

print("Loading data...")
df_raw = pl.read_parquet(T_PATH).select([
    pl.col('customer_id').cast(pl.Int64),
    pl.col('item_id').cast(pl.Utf8),
    pl.col('updated_date').cast(pl.Datetime).alias('event_ts')
]).with_columns(pl.col('event_ts').dt.month().alias('month'))

# Train is months <= 11
df_train = df_raw.filter(pl.col('month') <= 11)
# Test is month 12
df_test = df_raw.filter(pl.col('month') == 12)

# Sample 30,000 active users in month 12
test_users = df_test['customer_id'].unique().shuffle(seed=42).head(30000).to_list()
truth_df = df_test.filter(pl.col('customer_id').is_in(test_users))
truth_dict = {row[0]: set(row[1]) for row in truth_df.group_by('customer_id').agg(pl.col('item_id')).iter_rows()}

print("Filtering train data for fast execution...")
# We only train SVD on these 30,000 users!
df_train_sub = df_train.filter(pl.col('customer_id').is_in(test_users))

hist_users = df_train_sub['customer_id'].unique().to_list()
hist_items = df_train_sub['item_id'].unique().to_list()

u2idx = {uid: idx for idx, uid in enumerate(hist_users)}
i2idx = {iid: idx for idx, iid in enumerate(hist_items)}

rows = [u2idx[uid] for uid in df_train_sub['customer_id'].to_list()]
cols = [i2idx[iid] for iid in df_train_sub['item_id'].to_list()]
data = np.ones(len(rows))

mtx = csr_matrix((data, (rows, cols)), shape=(len(hist_users), len(hist_items)))

print(f"Matrix shape: {mtx.shape}")

print("Running TruncatedSVD...")
svd = TruncatedSVD(n_components=64, random_state=42)
u_emb = svd.fit_transform(mtx)
i_emb = svd.components_.T

print("Building seen map...")
seen_dict = {row[0]: set(row[1]) for row in df_train_sub.group_by('customer_id').agg(pl.col('item_id')).iter_rows()}

def evaluate_svd(filter_seen):
    hits = 0
    total_users = 0
    
    # Pre-calculate fallback popularity
    item_popularity = np.asarray(mtx.sum(axis=0)).ravel()
    fallback_indices = np.argsort(-item_popularity)[:10]
    fallback_items = [hist_items[i] for i in fallback_indices]
    
    for uid in test_users:
        if uid not in truth_dict:
            continue
        total_users += 1
        truth = truth_dict[uid]
        seen = seen_dict.get(uid, set())
        
        if uid not in u2idx:
            recs = fallback_items
        else:
            u_idx = u2idx[uid]
            scores = u_emb[u_idx] @ i_emb.T
            if filter_seen:
                for iid in seen:
                    if iid in i2idx:
                        scores[i2idx[iid]] = -np.inf
            top10 = np.argsort(-scores)[:10]
            recs = [hist_items[i] for i in top10]
            
        hits += sum(1 for r in recs if r in truth)
        
    prec = hits / (total_users * 10)
    print(f"SVD (Filter Seen = {filter_seen}): Precision@10 = {prec:.6f}, Hits = {hits}, Users = {total_users}")

evaluate_svd(filter_seen=True)
evaluate_svd(filter_seen=False)
