import time
import polars as pl
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

SEED = 42
T_PATH = r'd:\CS116\ProjectNumberOne\transaction_full_2025.parquet'
I_PATH = r'd:\CS116\ProjectNumberOne\items.parquet'

print("Loading data...")
df_raw = pl.read_parquet(T_PATH).select([
    pl.col('customer_id').cast(pl.Int64),
    pl.col('item_id').cast(pl.Utf8),
    pl.col('quantity').cast(pl.Int32),
    pl.col('price').cast(pl.Float32),
    pl.col('location').cast(pl.Utf8),
    pl.col('updated_date').cast(pl.Datetime).alias('event_ts')
]).with_columns([
    pl.col('event_ts').dt.month().alias('month')
])

items_df = pl.read_parquet(I_PATH).select([
    pl.col('item_id').cast(pl.Utf8),
    pl.col('category').cast(pl.Utf8),
    pl.col('category_l1').cast(pl.Utf8),
    pl.col('category_l2').cast(pl.Utf8),
    pl.col('category_l3').cast(pl.Utf8),
    pl.col('brand').cast(pl.Utf8),
    pl.col('size').cast(pl.Utf8)
])

cat_cols = ['category', 'category_l1', 'category_l2', 'category_l3', 'brand']
for c in cat_cols:
    items_df = items_df.with_columns(pl.col(c).fill_null('Unknown'))
    top_vals = items_df[c].value_counts().sort('count', descending=True).head(254)[c].to_list()
    items_df = items_df.with_columns(
        pl.when(pl.col(c).is_in(top_vals)).then(pl.col(c)).otherwise(pl.lit('Other')).alias(c)
    )
    items_df = items_df.with_columns(pl.col(c).cast(pl.Categorical).to_physical().cast(pl.Int32).alias(f"{c}_id"))

def standardize_age(val):
    try:
        if 'M' in val: return float(val.replace('M',''))/12.0
        if 'Y' in val: return float(val.replace('Y',''))
        return -1.0
    except: return -1.0

size_map = {row[0]: standardize_age(row[1]) for row in items_df.select(['item_id', 'size']).iter_rows()}
items_df = items_df.with_columns(pl.col('item_id').replace(size_map, default=-1.0).cast(pl.Float32).alias('item_age_proxy'))

print("Initializing Retriever...")
# Use months <= 11 as history, Month 12 unique users as targets
history_df = df_raw.filter(pl.col('month') <= 11)
target_users = df_raw.filter(pl.col('month') == 12)['customer_id'].unique().shuffle(seed=SEED).head(10000).to_list()

start = time.time()
class SpeedRetriever:
    def __init__(self, history_df, items_df):
        self.history_df = history_df
        self.items_df = items_df
        self.max_ts = history_df['event_ts'].max()
        
        self.global_top = history_df.filter(pl.col('event_ts') >= self.max_ts - pl.duration(days=14))\
            .group_by('item_id').len().sort('len', descending=True).head(150).select('item_id')
            
        self.local_heroes = history_df.filter(pl.col('event_ts') >= self.max_ts - pl.duration(days=60))\
            .group_by(['location', 'item_id']).len()\
            .sort(['location', 'len'], descending=[False, True])\
            .group_by('location').head(80)
            
        self.replenish = history_df.group_by(['customer_id', 'item_id']).agg([
            pl.col('event_ts').count().alias('buy_count'),
            pl.col('event_ts').min().alias('first_buy'),
            pl.col('event_ts').max().alias('last_buy')
        ]).filter(pl.col('buy_count') > 1)\
          .with_columns(((pl.col('last_buy') - pl.col('first_buy')).dt.total_days() / (pl.col('buy_count') - 1)).alias('avg_gap'))
        
        self._build_cf()
        
    def _build_cf(self):
        print("   Building CF sparse matrix...")
        s_t = time.time()
        hist = self.history_df.filter(pl.col('event_ts') >= self.max_ts - pl.duration(days=180)).select(['customer_id', 'item_id'])
        u_map = hist['customer_id'].unique()
        i_map = hist['item_id'].unique()
        
        u_df = pl.DataFrame({'customer_id': u_map, 'u_idx': np.arange(len(u_map), dtype=np.int64)})
        i_df = pl.DataFrame({'item_id': i_map, 'i_idx': np.arange(len(i_map), dtype=np.int32)})
        
        hist_indexed = hist.join(u_df, on='customer_id', how='inner').join(i_df, on='item_id', how='inner')
        rows = hist_indexed['u_idx'].to_numpy()
        cols = hist_indexed['i_idx'].to_numpy()
        data = np.ones(len(rows))
        
        self.mtx = csr_matrix((data, (rows, cols)), shape=(len(u_map), len(i_map)))
        self.u2idx = dict(zip(u_df['customer_id'], u_df['u_idx']))
        self.idx2i = i_map.to_list()
        print(f"   Sparse matrix built in {time.time() - s_t:.2f}s")
        
        print("   Running TruncatedSVD...")
        s_t = time.time()
        self.svd = TruncatedSVD(n_components=100, random_state=SEED)
        self.u_emb = self.svd.fit_transform(self.mtx)
        self.i_emb = self.svd.components_.T
        print(f"   SVD completed in {time.time() - s_t:.2f}s")
        
        print("   Building I2I similarity matrix...")
        s_t = time.time()
        norm_m = normalize(self.mtx, norm='l2', axis=0)
        self.i2i_sim = (norm_m.T.dot(norm_m)).astype(np.float32)
        self.i2i_sim.setdiag(0)
        print(f"   Similarity matrix computed in {time.time() - s_t:.2f}s")

    def get_candidates(self, target_users):
        cands = {}
        
        s_t = time.time()
        hist_s = self.history_df.filter(pl.col('customer_id').is_in(target_users))
        cands['hist'] = hist_s.select(['customer_id', 'item_id']).unique()
        print(f"      Hist candidates: {time.time() - s_t:.2f}s")
        
        s_t = time.time()
        due = self.replenish.filter(pl.col('customer_id').is_in(target_users))\
            .with_columns((self.max_ts - pl.col('last_buy')).dt.total_days().alias('days_since'))\
            .filter(pl.col('days_since') >= pl.col('avg_gap') * 0.8)\
            .select(['customer_id', 'item_id'])
        cands['repl'] = due
        print(f"      Repl candidates: {time.time() - s_t:.2f}s")
        
        s_t = time.time()
        cands['global'] = pl.DataFrame({'customer_id': target_users}).join(self.global_top.with_columns(pl.lit(1).alias('_k')), how='cross').drop('_k')
        print(f"      Global candidates: {time.time() - s_t:.2f}s")
        
        s_t = time.time()
        user_loc = hist_s.group_by('customer_id').agg(pl.col('location').mode().first().alias('location'))
        cands['local'] = user_loc.join(self.local_heroes, on='location').select(['customer_id', 'item_id']).unique()
        print(f"      Local candidates: {time.time() - s_t:.2f}s")
        
        s_t = time.time()
        u_idx = [self.u2idx[u] for u in target_users if u in self.u2idx]
        t_u = [u for u in target_users if u in self.u2idx]
        i_arr = np.array(self.idx2i)
        if u_idx:
            chunk = 4000
            c_svd, c_i2i = [], []
            for i in range(0, len(u_idx), chunk):
                idx_chunk = u_idx[i:i+chunk]
                u_b = np.array(t_u[i:i+chunk])
                
                scores_svd = self.u_emb[idx_chunk] @ self.i_emb.T
                t60 = np.argsort(-scores_svd, axis=1)[:, :60]
                c_svd.append(pl.DataFrame({
                    'customer_id': pl.Series(np.repeat(u_b, 60), dtype=pl.Int64),
                    'item_id': i_arr[t60.flatten()]
                }))
                
                scores_i2i = self.mtx[idx_chunk].dot(self.i2i_sim).toarray()
                t80 = np.argsort(-scores_i2i, axis=1)[:, :80]
                mask = np.take_along_axis(scores_i2i, t80, axis=1) > 0
                c_i2i.append(pl.DataFrame({
                    'customer_id': pl.Series(np.repeat(u_b, 80)[mask.flatten()], dtype=pl.Int64),
                    'item_id': i_arr[t80.flatten()][mask.flatten()]
                }))
            cands['svd'] = pl.concat(c_svd).unique() if c_svd else pl.DataFrame(schema={'customer_id': pl.Int64, 'item_id': pl.Utf8})
            cands['i2i'] = pl.concat(c_i2i).unique() if c_i2i else pl.DataFrame(schema={'customer_id': pl.Int64, 'item_id': pl.Utf8})
        print(f"      CF candidates: {time.time() - s_t:.2f}s")
        
        s_t = time.time()
        u_cat_top = self.history_df.filter(pl.col('customer_id').is_in(target_users))\
            .join(self.items_df.select(['item_id', 'category_l1']), on='item_id')\
            .group_by(['customer_id', 'category_l1']).len().sort('len', descending=True).group_by('customer_id').head(1)
        
        cat_global_top = self.history_df.filter(pl.col('event_ts') >= self.max_ts - pl.duration(days=30))\
            .join(self.items_df.select(['item_id', 'category_l1']), on='item_id')\
            .group_by(['category_l1', 'item_id']).len().sort('len', descending=True).group_by('category_l1').head(10)
            
        cands['cat_top'] = u_cat_top.join(cat_global_top, on='category_l1').select(['customer_id', 'item_id'])
        print(f"      Cat Top candidates: {time.time() - s_t:.2f}s")

        s_t = time.time()
        all_c = pl.concat([df for df in cands.values() if df is not None and df.height > 0]).unique()
        print(f"      Deduplication and concat: {time.time() - s_t:.2f}s")
        return all_c

ret = SpeedRetriever(history_df, items_df)
print("Benchmarking get_candidates on 10,000 target users...")
cands = ret.get_candidates(target_users)
print(f"Total candidates: {cands.height}")
