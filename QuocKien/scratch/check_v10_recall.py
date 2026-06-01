import polars as pl
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

T_PATH = r'd:\CS116\ProjectNumberOne\transaction_full_2025.parquet'
I_PATH = r'd:\CS116\ProjectNumberOne\items.parquet'
SEED = 42

print("Loading data...")
df_raw = pl.read_parquet(T_PATH).select([
    pl.col('customer_id').cast(pl.Int64),
    pl.col('item_id').cast(pl.Utf8),
    pl.col('updated_date').cast(pl.Datetime).alias('event_ts')
]).with_columns(pl.col('event_ts').dt.month().alias('month'))

items_df = pl.read_parquet(I_PATH).select(['item_id', 'category_l1'])

hist_df = df_raw.filter(pl.col('month') <= 11)
truth_df = df_raw.filter(pl.col('month') == 12)

# Sample 10,000 active users in December
target_users = truth_df['customer_id'].unique().shuffle(seed=SEED).head(10000).to_list()
truth = truth_df.filter(pl.col('customer_id').is_in(target_users)).select(['customer_id', 'item_id']).unique()

print("Initializing V10Retriever...")
max_ts = hist_df['event_ts'].max()

global_top = hist_df.filter(pl.col('event_ts') >= max_ts - pl.duration(days=14))\
    .group_by('item_id').len().sort('len', descending=True).head(150).select('item_id')

print("Building CF sources (filtered to target users to prevent RAM crash)...")
hist_cf = hist_df.filter(pl.col('customer_id').is_in(target_users))
u_map = hist_cf['customer_id'].unique()
i_map = hist_df['item_id'].unique()  # Keep all items in catalog

u_df = pl.DataFrame({'customer_id': u_map, 'u_idx': np.arange(len(u_map), dtype=np.int64)})
i_df = pl.DataFrame({'item_id': i_map, 'i_idx': np.arange(len(i_map), dtype=np.int32)})

hist_indexed = hist_cf.join(u_df, on='customer_id', how='inner').join(i_df, on='item_id', how='inner')
rows = hist_indexed['u_idx'].to_numpy()
cols = hist_indexed['i_idx'].to_numpy()
data = np.ones(len(rows))

mtx = csr_matrix((data, (rows, cols)), shape=(len(u_map), len(i_map)))
u2idx = dict(zip(u_df['customer_id'], u_df['u_idx']))
idx2i = i_map.to_list()

svd = TruncatedSVD(n_components=100, random_state=SEED)
u_emb = svd.fit_transform(mtx)
i_emb = svd.components_.T

norm_m = normalize(mtx, norm='l2', axis=0)
i2i_sim = (norm_m.T.dot(norm_m)).astype(np.float32)
i2i_sim.setdiag(0)

print("Extracting candidates...")
cands = {}
hist_s = hist_df.filter(pl.col('customer_id').is_in(target_users))

# Source 1: Hist
cands['hist'] = hist_s.select(['customer_id', 'item_id']).unique()

# Source 2: Global
cands['global'] = pl.DataFrame({'customer_id': target_users}).join(global_top.with_columns(pl.lit(1).alias('_k')), how='cross').drop('_k')

# Source 3: SVD & I2I
u_idx = [u2idx[u] for u in target_users if u in u2idx]
t_u = [u for u in target_users if u in u2idx]
i_arr = np.array(idx2i)

c_svd, c_i2i = [], []
if u_idx:
    chunk = 4000
    for i in range(0, len(u_idx), chunk):
        idx_chunk = u_idx[i:i+chunk]
        u_b = np.array(t_u[i:i+chunk])
        
        # SVD
        scores_svd = u_emb[idx_chunk] @ i_emb.T
        t60 = np.argsort(-scores_svd, axis=1)[:, :60]
        c_svd.append(pl.DataFrame({
            'customer_id': pl.Series(np.repeat(u_b, 60), dtype=pl.Int64),
            'item_id': i_arr[t60.flatten()]
        }))
        
        # I2I
        scores_i2i = mtx[idx_chunk].dot(i2i_sim).toarray()
        t80 = np.argsort(-scores_i2i, axis=1)[:, :80]
        mask = np.take_along_axis(scores_i2i, t80, axis=1) > 0
        c_i2i.append(pl.DataFrame({
            'customer_id': pl.Series(np.repeat(u_b, 80)[mask.flatten()], dtype=pl.Int64),
            'item_id': i_arr[t80.flatten()][mask.flatten()]
        }))
        
cands['svd'] = pl.concat(c_svd).unique() if c_svd else pl.DataFrame(schema={'customer_id': pl.Int64, 'item_id': pl.Utf8})
cands['i2i'] = pl.concat(c_i2i).unique() if c_i2i else pl.DataFrame(schema={'customer_id': pl.Int64, 'item_id': pl.Utf8})

# Source 4: Category Top
u_cat_top = hist_s.join(items_df.select(['item_id', 'category_l1']), on='item_id')\
    .group_by(['customer_id', 'category_l1']).len().sort('len', descending=True).group_by('customer_id').head(1)

cat_global_top = hist_df.filter(pl.col('event_ts') >= max_ts - pl.duration(days=30))\
    .join(items_df.select(['item_id', 'category_l1']), on='item_id')\
    .group_by(['category_l1', 'item_id']).len().sort('len', descending=True).group_by('category_l1').head(10)

cands['cat_top'] = u_cat_top.join(cat_global_top, on='category_l1').select(['customer_id', 'item_id'])

# Check recall for each source and combined
print("\n--- Recall Audits ---")
for name, df in cands.items():
    if df is not None and df.height > 0:
        h = df.join(truth, on=['customer_id', 'item_id'], how='inner').height
        print(f"  {name} Recall: {h / truth.height:.4f} ({h} hits)")

combined = pl.concat([df for df in cands.values() if df is not None and df.height > 0]).unique()
c_hits = combined.join(truth, on=['customer_id', 'item_id'], how='inner').height
print(f"\n  Combined Retriever Recall: {c_hits / truth.height:.4f} ({c_hits}/{truth.height})")
print(f"  Average candidates per user: {combined.height / len(target_users):.1f}")
