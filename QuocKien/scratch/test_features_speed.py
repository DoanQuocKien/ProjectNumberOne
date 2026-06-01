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

history_df = df_raw.filter(pl.col('month') <= 11)
target_users = df_raw.filter(pl.col('month') == 12)['customer_id'].unique().shuffle(seed=SEED).head(10000).to_list()

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
        
        self.svd = TruncatedSVD(n_components=100, random_state=SEED)
        self.u_emb = self.svd.fit_transform(self.mtx)
        self.i_emb = self.svd.components_.T
        
        norm_m = normalize(self.mtx, norm='l2', axis=0)
        self.i2i_sim = (norm_m.T.dot(norm_m)).astype(np.float32)
        self.i2i_sim.setdiag(0)

    def get_candidates(self, target_users):
        cands = {}
        hist_s = self.history_df.filter(pl.col('customer_id').is_in(target_users))
        cands['hist'] = hist_s.select(['customer_id', 'item_id']).unique()
        
        due = self.replenish.filter(pl.col('customer_id').is_in(target_users))\
            .with_columns((self.max_ts - pl.col('last_buy')).dt.total_days().alias('days_since'))\
            .filter(pl.col('days_since') >= pl.col('avg_gap') * 0.8)\
            .select(['customer_id', 'item_id'])
        cands['repl'] = due
        
        cands['global'] = pl.DataFrame({'customer_id': target_users}).join(self.global_top.with_columns(pl.lit(1).alias('_k')), how='cross').drop('_k')
        
        user_loc = hist_s.group_by('customer_id').agg(pl.col('location').mode().first().alias('location'))
        cands['local'] = user_loc.join(self.local_heroes, on='location').select(['customer_id', 'item_id']).unique()
        
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
        
        u_cat_top = self.history_df.filter(pl.col('customer_id').is_in(target_users))\
            .join(self.items_df.select(['item_id', 'category_l1']), on='item_id')\
            .group_by(['customer_id', 'category_l1']).len().sort('len', descending=True).group_by('customer_id').head(1)
        
        cat_global_top = self.history_df.filter(pl.col('event_ts') >= self.max_ts - pl.duration(days=30))\
            .join(self.items_df.select(['item_id', 'category_l1']), on='item_id')\
            .group_by(['category_l1', 'item_id']).len().sort('len', descending=True).group_by('category_l1').head(10)
            
        cands['cat_top'] = u_cat_top.join(cat_global_top, on='category_l1').select(['customer_id', 'item_id'])

        all_c = pl.concat([df for df in cands.values() if df is not None and df.height > 0]).unique()
        return all_c

ret = SpeedRetriever(history_df, items_df)
print("Generating candidates for 10k users...")
ds = ret.get_candidates(target_users)
print(f"Candidates generated: {ds.height} rows.")

print("Pre-computing Global Profiles...")
max_ts = history_df['event_ts'].max()
u_brand_counts = history_df.select(['customer_id', 'item_id']).join(items_df.select(['item_id', 'brand']), on='item_id')\
    .group_by(['customer_id', 'brand']).len().rename({'len': 'brand_count'})
u_brand_hhi = u_brand_counts.with_columns(
    (pl.col('brand_count') / pl.col('brand_count').sum().over('customer_id')).alias('brand_share')
).with_columns(
    (pl.col('brand_share') * pl.col('brand_share')).alias('brand_share_sq')
).group_by('customer_id').agg(pl.col('brand_share_sq').sum().alias('u_brand_hhi'))

u_cat_counts = history_df.select(['customer_id', 'item_id']).join(items_df.select(['item_id', 'category_l1']), on='item_id')\
    .group_by(['customer_id', 'category_l1']).len().rename({'len': 'cat_count'})
u_cat_hhi = u_cat_counts.with_columns(
    (pl.col('cat_count') / pl.col('cat_count').sum().over('customer_id')).alias('cat_share')
).with_columns(
    (pl.col('cat_share') * pl.col('cat_share')).alias('cat_share_sq')
).group_by('customer_id').agg(pl.col('cat_share_sq').sum().alias('u_cat_hhi'))

global_avg_age = items_df.filter(pl.col('item_age_proxy') >= 0)['item_age_proxy'].mean()
if global_avg_age is None: global_avg_age = 1.0
u_avg_age = history_df.select(['customer_id', 'item_id']).join(items_df.select(['item_id', 'item_age_proxy']), on='item_id')\
    .filter(pl.col('item_age_proxy') >= 0)\
    .group_by('customer_id').agg(pl.col('item_age_proxy').mean().alias('u_avg_age_proxy'))

u_prof = history_df.group_by('customer_id').agg([
    pl.col('item_id').n_unique().alias('u_unique_items'),
    pl.col('quantity').sum().alias('u_total_qty'),
    pl.col('price').mean().alias('u_avg_price'),
    pl.col('price').std().alias('u_price_std'),
    (max_ts - pl.col('event_ts').min()).dt.total_days().alias('u_tenure_days'),
    (pl.col('item_id').n_unique() / pl.col('quantity').sum().clip(1)).alias('u_exploration_ratio')
]).join(u_brand_hhi, on='customer_id', how='left')\
  .join(u_cat_hhi, on='customer_id', how='left')\
  .join(u_avg_age, on='customer_id', how='left')\
  .with_columns(pl.col('u_avg_age_proxy').fill_null(global_avg_age))

i_repeats = history_df.group_by(['item_id', 'customer_id']).len().filter(pl.col('len') > 1)\
    .group_by('item_id').len().rename({'len': 'repeat_buyers'})

i_prof = history_df.group_by('item_id').agg([
    pl.col('customer_id').n_unique().alias('i_unique_users'),
    pl.col('quantity').sum().alias('i_total_qty'),
    pl.col('location').n_unique().alias('i_hubs_count'),
    pl.col('price').median().alias('i_ref_price')
]).join(i_repeats, on='item_id', how='left')\
  .with_columns((pl.col('repeat_buyers').fill_null(0) / pl.col('i_unique_users')).alias('i_repeat_rate'))\
  .drop('repeat_buyers')

ui_hist = history_df.group_by(['customer_id', 'item_id']).agg([
    pl.col('quantity').sum().alias('ui_total_qty'),
    (max_ts - pl.col('event_ts').max()).dt.total_days().alias('ui_recency_days')
])

u_pref_cat = history_df.join(items_df.select(['item_id', 'category_l1']), on='item_id')\
    .group_by(['customer_id', 'category_l1']).len().sort('len', descending=True)\
    .group_by('customer_id').head(1).select(['customer_id', 'category_l1']).rename({'category_l1': 'pref_cat_l1'})
    
u_pref_brand = history_df.join(items_df.select(['item_id', 'category_l1', 'brand']), on='item_id')\
    .group_by(['customer_id', 'category_l1', 'brand']).len().sort('len', descending=True)\
    .group_by(['customer_id', 'category_l1']).head(1).select(['customer_id', 'category_l1', 'brand']).rename({'brand': 'pref_brand'})

vol_7d = history_df.filter(pl.col('event_ts') >= max_ts - pl.duration(days=7)).group_by('item_id').len().rename({'len': 'v7'})
vol_21d = history_df.filter(pl.col('event_ts') >= max_ts - pl.duration(days=21)).group_by('item_id').len().rename({'len': 'v21'})
momentum = vol_7d.join(vol_21d, on='item_id', how='left').with_columns((pl.col('v7') / (pl.col('v21') / 3.0 + 1)).alias('item_momentum'))

u_cat = history_df.join(items_df.select(['item_id', 'category_l1']), on='item_id')\
    .group_by(['customer_id', 'category_l1']).len()\
    .with_columns((pl.col('len') / pl.col('len').sum().over('customer_id')).alias('u_cat_affinity'))

u_loc = history_df.group_by('customer_id').agg(pl.col('location').mode().first().alias('location'))

chunk_u_df = pl.DataFrame({'customer_id': target_users}, schema={'customer_id': pl.Int64})

print("Benchmarking Joins...")
s_t = time.time()
ds = ds.join(u_prof.join(chunk_u_df, on='customer_id', how='inner'), on='customer_id', how='left')
print(f"   Join u_prof: {time.time() - s_t:.2f}s")

s_t = time.time()
ds = ds.join(i_prof, on='item_id', how='left')
print(f"   Join i_prof: {time.time() - s_t:.2f}s")

s_t = time.time()
ds = ds.join(ui_hist.join(chunk_u_df, on='customer_id', how='inner'), on=['customer_id', 'item_id'], how='left')
print(f"   Join ui_hist: {time.time() - s_t:.2f}s")

s_t = time.time()
ds = ds.join(items_df.select(['item_id', 'item_age_proxy'] + [f'{c}_id' for c in cat_cols] + ['category_l1', 'brand']), on='item_id', how='left')
print(f"   Join items_df: {time.time() - s_t:.2f}s")

s_t = time.time()
ds = ds.join(momentum.select(['item_id', 'item_momentum']), on='item_id', how='left')
print(f"   Join momentum: {time.time() - s_t:.2f}s")

s_t = time.time()
ds = ds.join(u_cat.join(chunk_u_df, on='customer_id', how='inner').select(['customer_id', 'category_l1', 'u_cat_affinity']), on=['customer_id', 'category_l1'], how='left')
print(f"   Join u_cat: {time.time() - s_t:.2f}s")

s_t = time.time()
ds = ds.join(u_pref_cat.join(chunk_u_df, on='customer_id', how='inner'), on='customer_id', how='left')
print(f"   Join u_pref_cat: {time.time() - s_t:.2f}s")

s_t = time.time()
ds = ds.join(u_pref_brand.join(chunk_u_df, on='customer_id', how='inner'), on=['customer_id', 'category_l1'], how='left')
print(f"   Join u_pref_brand: {time.time() - s_t:.2f}s")

s_t = time.time()
ds = ds.join(u_loc.join(chunk_u_df, on='customer_id', how='inner'), on='customer_id', how='left')
print(f"   Join u_loc: {time.time() - s_t:.2f}s")

print("Done benchmarking features!")
