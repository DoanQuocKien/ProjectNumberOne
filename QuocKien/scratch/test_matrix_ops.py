import polars as pl
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import sys

T_PATH = r'd:\CS116\ProjectNumberOne\transaction_full_2025.parquet'
SEED = 42

print("Loading a tiny sample...")
df_raw = pl.read_parquet(T_PATH).select([
    pl.col('customer_id').cast(pl.Int64),
    pl.col('item_id').cast(pl.Utf8),
    pl.col('updated_date').cast(pl.Datetime).alias('event_ts')
]).with_columns(pl.col('event_ts').dt.month().alias('month'))

hist_df = df_raw.filter(pl.col('month') <= 11)
target_users = hist_df['customer_id'].unique().shuffle(seed=SEED).head(2000).to_list()

print("Filtering history...")
hist = hist_df.filter(pl.col('customer_id').is_in(target_users))

u_map = hist['customer_id'].unique()
i_map = hist_df['item_id'].unique()

print(f"Users: {len(u_map)}, Items: {len(i_map)}")

u_df = pl.DataFrame({'customer_id': u_map, 'u_idx': np.arange(len(u_map), dtype=np.int64)})
i_df = pl.DataFrame({'item_id': i_map, 'i_idx': np.arange(len(i_map), dtype=np.int32)})

hist_indexed = hist.join(u_df, on='customer_id', how='inner').join(i_df, on='item_id', how='inner')
rows = hist_indexed['u_idx'].to_numpy()
cols = hist_indexed['i_idx'].to_numpy()
data = np.ones(len(rows))

mtx = csr_matrix((data, (rows, cols)), shape=(len(u_map), len(i_map)))

print("Testing SVD fit...")
svd = TruncatedSVD(n_components=64, random_state=SEED)
u_emb = svd.fit_transform(mtx)
i_emb = svd.components_.T
print("SVD successful!")

print("Testing I2I normalization...")
norm_m = normalize(mtx, norm='l2', axis=0)
i2i_sim = (norm_m.T.dot(norm_m)).astype(np.float32)
i2i_sim.setdiag(0)
print("I2I normalization successful!")

print("Testing SVD inference...")
scores_svd = u_emb[:100] @ i_emb.T
print("SVD inference successful!")

print("Testing I2I inference...")
# Slice mtx and multiply
sub_mtx = mtx[:100]
scores_i2i = sub_mtx.dot(i2i_sim).toarray()
print("I2I inference successful!")

print("\nAll sub-steps completed successfully!")
