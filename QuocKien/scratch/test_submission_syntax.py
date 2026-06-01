import polars as pl
import numpy as np
import os
import gc
import re
import warnings
import pickle
from pathlib import Path
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

# Enable global string cache to guarantee Categorical alignment across all dataframes
pl.enable_string_cache()
warnings.filterwarnings('ignore')

SEED = 42
T_PATH = r'd:\CS116\ProjectNumberOne\transaction_full_2025.parquet'
I_PATH = r'd:\CS116\ProjectNumberOne\items.parquet'

print("Loading Data slices for rigorous type-safety validation...")
# Load a small sample (10k rows is enough to test all schema and type safety!)
df_raw = pl.read_parquet(T_PATH, n_rows=10000).select([
    pl.col('customer_id').cast(pl.Int64),
    pl.col('item_id').cast(pl.Utf8),
    pl.col('quantity').cast(pl.Int32),
    pl.col('price').cast(pl.Float32),
    pl.col('location').cast(pl.Utf8),
    pl.col('updated_date').cast(pl.Datetime).alias('event_ts')
]).with_columns([
    pl.col('event_ts').dt.month().alias('month'),
    pl.col('event_ts').dt.weekday().alias('dow')
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

# Item preparation
cat_cols = ['category', 'category_l1', 'category_l2', 'category_l3', 'brand']
for c in cat_cols:
    items_df = items_df.with_columns(pl.col(c).fill_null('Unknown'))
    top_vals = items_df[c].value_counts().sort('count', descending=True).head(254)[c].to_list()
    items_df = items_df.with_columns(
        pl.when(pl.col(c).is_in(top_vals)).then(pl.col(c)).otherwise(pl.lit('Other')).alias(c)
    )
    items_df = items_df.with_columns(pl.col(c).cast(pl.Categorical).to_physical().cast(pl.Int32).alias(f"{c}_id"))

def standardize_age(text):
    raw_text = str(text).strip()
    clean_text = raw_text.lower()
    if re.search(r'(\*|x\\d|cm)', clean_text): return 0.5
    if re.search(r'\\bb\\d{2}\\b', clean_text): return 18.0
    if 's17' in clean_text: return 1.0
    if '110' in clean_text: return 5.0
    if "không xác định" in clean_text or not clean_text: return -1.0
    diaper_map = {
        r'\\bnb\\b': 0.0, r'\\bss\\b': 0.0, r'\\bsơ sinh\\b': 0.0,
        r'\\bs\\b': 0.25, r'\\bm\\b': 0.6, r'\\bl\\b': 1.2,
        r'\\bxl\\b': 2.0, r'\\bxxl\\b': 3.5
    }
    for pattern, val in diaper_map.items():
        if re.search(pattern, clean_text): return val
    range_match = re.search(r'(\\d+\\.?\\d*)\\s*-\\s*(\\d+\\.?\\d*)', clean_text)
    if range_match:
        s, e = float(range_match.group(1)), float(range_match.group(2))
        avg = (s + e) / 2
        if any(x in clean_text for x in ['m', 'tháng']): return round(avg / 12, 3)
        return avg
    m_match = re.search(r'(\\d+\\.?\\d*)\\s*(m|tháng)', clean_text)
    if m_match: return round(float(m_match.group(1)) / 12, 3)
    y_match = re.search(r'(\\d+\\.?\\d*)\\s*(y|t|tuổi)', clean_text)
    if y_match: return float(y_match.group(1))
    pure_num = re.search(r'^(\\d+)$', clean_text)
    if pure_num:
        val = float(pure_num.group(1))
        return round(val/12, 3) if val > 6 else val
    return -1.0

size_map = {row[0]: standardize_age(row[1]) for row in items_df.select(['item_id', 'size']).iter_rows()}
items_df = items_df.with_columns(pl.col('item_id').replace(size_map, default=-1.0).cast(pl.Float32).alias('item_age_proxy'))

class V12Retriever:
    def __init__(self, history_df, items_df):
        self.history_df = history_df
        self.items_df = items_df
        self.max_ts = history_df['event_ts'].max()
        
        self.item_locs = history_df.group_by('item_id').agg(pl.col('location').unique().alias('item_hubs'))
        self.item_prcs = history_df.group_by('item_id').agg(pl.col('price').median().alias('item_p'))
        
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
        
        u_df = pl.DataFrame({
            'customer_id': u_map,
            'u_idx': np.arange(len(u_map), dtype=np.int64)
        })
        i_df = pl.DataFrame({
            'item_id': i_map,
            'i_idx': np.arange(len(i_map), dtype=pl.Int32)
        })
        
        hist_indexed = hist.join(u_df, on='customer_id', how='inner').join(i_df, on='item_id', how='inner')
        
        rows = hist_indexed['u_idx'].to_numpy()
        cols = hist_indexed['i_idx'].to_numpy()
        data = np.ones(len(rows))
        
        self.mtx = csr_matrix((data, (rows, cols)), shape=(len(u_map), len(i_map)))
        
        self.u2idx = dict(zip(u_df['customer_id'], u_df['u_idx']))
        self.i2idx = dict(zip(i_df['item_id'], i_df['i_idx']))
        self.idx2i = i_map.to_list()
        
        n_comp = min(100, len(i_map) - 1)
        self.svd = TruncatedSVD(n_components=n_comp, random_state=SEED)
        self.u_emb = self.svd.fit_transform(self.mtx)
        self.i_emb = self.svd.components_.T
        
        norm_m = normalize(self.mtx, norm='l2', axis=0)
        sim = (norm_m.T.dot(norm_m)).astype(np.float32)
        sim.setdiag(0)
        
        indptr = sim.indptr
        indices = sim.indices
        data = sim.data
        
        p_rows, p_cols, p_vals = [], [], []
        for i in range(sim.shape[0]):
            start, end = indptr[i], indptr[i+1]
            if end > start:
                idx = indices[start:end]
                d = data[start:end]
                if len(d) > 150:
                    top_k = np.argpartition(-d, 150)[:150]
                    idx = idx[top_k]
                    d = d[top_k]
                p_rows.extend([i] * len(idx))
                p_cols.extend(idx)
                p_vals.extend(d)
                
        self.i2i_sim = csr_matrix((p_vals, (p_rows, p_cols)), shape=sim.shape)
        del norm_m, sim; gc.collect()

    def get_candidates(self, target_users):
        target_users_df = pl.DataFrame({'customer_id': target_users}, schema={'customer_id': pl.Int64})
        cands = {}
        
        hist_s = self.history_df.join(target_users_df, on='customer_id', how='inner')
        cands['hist'] = hist_s.select(['customer_id', 'item_id']).unique()
        
        due = self.replenish.join(target_users_df, on='customer_id', how='inner')\
            .with_columns((self.max_ts - pl.col('last_buy')).dt.total_days().alias('days_since'))\
            .filter(pl.col('days_since') >= pl.col('avg_gap') * 0.8)\
            .select(['customer_id', 'item_id'])
        cands['repl'] = due
        
        cands['global'] = target_users_df.join(self.global_top.with_columns(pl.lit(1).alias('_k')), how='cross').drop('_k')
        
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
                t60_idx = np.argpartition(-scores_svd, 60, axis=1)[:, :60]
                t60_scores = np.take_along_axis(-scores_svd, t60_idx, axis=1)
                t60_sort = np.argsort(t60_scores, axis=1)
                t60 = np.take_along_axis(t60_idx, t60_sort, axis=1)
                
                c_svd.append(pl.DataFrame({
                    'customer_id': pl.Series(np.repeat(u_b, 60), dtype=pl.Int64),
                    'item_id': i_arr[t60.flatten()]
                }))
                
                scores_i2i = self.mtx[idx_chunk].dot(self.i2i_sim).toarray()
                t80_idx = np.argpartition(-scores_i2i, 80, axis=1)[:, :80]
                t80_scores = np.take_along_axis(-scores_i2i, t80_idx, axis=1)
                t80_sort = np.argsort(t80_scores, axis=1)
                t80 = np.take_along_axis(t80_idx, t80_sort, axis=1)
                
                mask = np.take_along_axis(scores_i2i, t80, axis=1) > 0
                c_i2i.append(pl.DataFrame({
                    'customer_id': pl.Series(np.repeat(u_b, 80)[mask.flatten()], dtype=pl.Int64),
                    'item_id': i_arr[t80.flatten()][mask.flatten()]
                }))
            cands['svd'] = pl.concat(c_svd).unique() if c_svd else pl.DataFrame(schema={'customer_id': pl.Int64, 'item_id': pl.Utf8})
            cands['i2i'] = pl.concat(c_i2i).unique() if c_i2i else pl.DataFrame(schema={'customer_id': pl.Int64, 'item_id': pl.Utf8})
            
        u_cat_top = self.history_df.join(target_users_df, on='customer_id', how='inner')\
            .join(self.items_df.select(['item_id', 'category_l1']), on='item_id')\
            .group_by(['customer_id', 'category_l1']).len().sort('len', descending=True).group_by('customer_id').head(1)
        
        cat_global_top = self.history_df.filter(pl.col('event_ts') >= self.max_ts - pl.duration(days=30))\
            .join(self.items_df.select(['item_id', 'category_l1']), on='item_id')\
            .group_by(['category_l1', 'item_id']).len().sort('len', descending=True).group_by('category_l1').head(10)
            
        cands['cat_top'] = u_cat_top.join(cat_global_top, on='category_l1').select(['customer_id', 'item_id'])

        all_c = pl.concat([df for df in cands.values() if df is not None and df.height > 0]).unique()
        
        uh = user_loc.rename({'location': 'loc'})
        up = hist_s.group_by('customer_id').agg(pl.col('price').mean().alias('avg_p'))
        f = all_c.join(up, on='customer_id', how='left')\
                 .join(self.item_prcs, on='item_id', how='left')\
                 .filter((pl.col('item_p') <= pl.col('avg_p') * 6) | (pl.col('avg_p').is_null()))\
                 .select(['customer_id', 'item_id'])
        item_loc_flat = self.item_locs.explode('item_hubs').rename({'item_hubs': 'loc'})
        filtered_cands = f.join(uh, on='customer_id', how='left')\
                          .join(item_loc_flat, on=['item_id', 'loc'], how='inner')\
                          .select(['customer_id', 'item_id'])
                          
        return filtered_cands

# Instantiate retriever on slice data
print("Building Retriever...")
retriever = V12Retriever(df_raw, items_df)

# Setup dummy targets and features
all_feats = ['u_unique_items', 'u_total_qty', 'u_avg_price', 'u_price_std', 'u_tenure_days', 'u_exploration_ratio', 'u_brand_hhi',
             'i_unique_users', 'i_total_qty', 'i_hubs_count', 'i_ref_price', 'i_repeat_rate',
             'ui_total_qty', 'ui_recency_days', 'ui_is_primary_cat', 'ui_is_preferred_brand',
             'ui_price_diff', 'ui_price_ratio', 'ui_loc_sales', 'item_momentum', 'item_age_proxy', 'u_cat_affinity',
             'u_cat_hhi', 'u_avg_age_proxy', 'ui_size_age_diff', 'ui_size_age_ratio', 'ui_already_bought_discretionary', 'ui_loc_sparsity_penalty'] + [f'{c}_id' for c in cat_cols]

# Build a dummy model to test prediction step
class DummyModel:
    def predict(self, X):
        return np.random.rand(len(X))

model = DummyModel()
target_users = df_raw['customer_id'].unique().head(100).to_list()

# Run the optimized function with full type safety checks
def test_submission_pkl(history_df, items_df, target_users, model, all_feats, output_path, chunk_size=20):
    max_ts = history_df['event_ts'].max()
    
    print("Pre-computing Global Profiles...")
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
      
    i_repeats = history_df.select(['customer_id', 'item_id']).group_by(['item_id', 'customer_id']).len().filter(pl.col('len') > 1)\
        .group_by('item_id').len().rename({'len': 'repeat_buyers'})
    i_prof = history_df.group_by('item_id').agg([
        pl.col('customer_id').n_unique().alias('i_unique_users'),
        pl.col('quantity').sum().alias('i_total_qty'),
        pl.col('location').n_unique().alias('i_hubs_count'),
        pl.col('price').median().alias('i_ref_price')
    ]).join(i_repeats, on='item_id', how='left')\
      .with_columns((pl.col('repeat_buyers').fill_null(0) / pl.col('i_unique_users')).alias('i_repeat_rate'))\
      .drop('repeat_buyers')

    vol_7d = history_df.filter(pl.col('event_ts') >= max_ts - pl.duration(days=7)).group_by('item_id').len().rename({'len': 'v7'})
    vol_21d = history_df.filter(pl.col('event_ts') >= max_ts - pl.duration(days=21)).group_by('item_id').len().rename({'len': 'v21'})
    momentum = vol_7d.join(vol_21d, on='item_id', how='left').with_columns((pl.col('v7') / (pl.col('v21') / 3.0 + 1)).alias('item_momentum'))
    
    u_cat = history_df.select(['customer_id', 'item_id']).join(items_df.select(['item_id', 'category_l1']), on='item_id')\
        .group_by(['customer_id', 'category_l1']).len()\
        .with_columns((pl.col('len') / pl.col('len').sum().over('customer_id')).alias('u_cat_affinity'))

    u_loc = history_df.group_by('customer_id').agg(pl.col('location').mode().first().alias('location'))
    loc_item_pop = history_df.group_by(['location', 'item_id']).len().rename({'len': 'ui_loc_sales'})

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

    # --- THE DOWNCASTING HELPER THAT EXCLUDES CRITICAL JOIN KEYS TO AVOID COMPUTE ERROR TYPE MISMATCHES ---
    def safe_downcast(df, exclude=['customer_id', 'item_id']):
        cols = []
        for col, dtype in zip(df.columns, df.dtypes):
            if col in exclude:
                continue
            if dtype == pl.Float64:
                cols.append(pl.col(col).cast(pl.Float32))
            elif dtype == pl.Int64:
                cols.append(pl.col(col).cast(pl.Int32))
        return df.with_columns(cols) if cols else df

    print("Downcasting global tables...")
    u_prof = safe_downcast(u_prof)
    i_prof = safe_downcast(i_prof)
    ui_hist = safe_downcast(ui_hist)
    momentum = safe_downcast(momentum)
    u_cat = safe_downcast(u_cat)
    loc_item_pop = safe_downcast(loc_item_pop)
    u_loc = safe_downcast(u_loc)

    # Cast to Categorical on active cache
    items_df = items_df.with_columns([
        pl.col('brand').cast(pl.Categorical),
        pl.col('category_l1').cast(pl.Categorical)
    ])
    u_pref_cat = u_pref_cat.with_columns(pl.col('pref_cat_l1').cast(pl.Categorical))
    u_pref_brand = u_pref_brand.with_columns([
        pl.col('pref_brand').cast(pl.Categorical),
        pl.col('category_l1').cast(pl.Categorical)
    ])
    u_cat = u_cat.with_columns(pl.col('category_l1').cast(pl.Categorical))
    u_loc = u_loc.with_columns(pl.col('location').cast(pl.Categorical))
    loc_item_pop = loc_item_pop.with_columns(pl.col('location').cast(pl.Categorical))

    # Lazy frames setup
    u_prof_lazy = u_prof.lazy()
    i_prof_lazy = i_prof.lazy()
    ui_hist_lazy = ui_hist.lazy()
    momentum_lazy = momentum.lazy()
    u_cat_lazy = u_cat.lazy()
    u_loc_lazy = u_loc.lazy()
    loc_item_pop_lazy = loc_item_pop.lazy()
    u_pref_cat_lazy = u_pref_cat.lazy()
    u_pref_brand_lazy = u_pref_brand.lazy()
    
    items_feat_lazy = items_df.select(
        ['item_id', 'item_age_proxy', 'brand', 'category_l1'] + [f'{c}_id' for c in cat_cols]
    ).lazy()

    print("Executing loop lazy collection validation...")
    for idx in range(0, len(target_users), chunk_size):
        chunk_u = target_users[idx:idx+chunk_size]
        ds = retriever.get_candidates(chunk_u)
        if ds.is_empty():
            continue
            
        ds_lazy = ds.lazy()
        ds_lazy = (
            ds_lazy
            .join(u_prof_lazy, on='customer_id', how='left')
            .join(i_prof_lazy, on='item_id', how='left')
            .join(ui_hist_lazy, on=['customer_id', 'item_id'], how='left')
            .join(items_feat_lazy, on='item_id', how='left')
            .join(momentum_lazy.select(['item_id', 'item_momentum']), on='item_id', how='left')
            .join(u_cat_lazy.select(['customer_id', 'category_l1', 'u_cat_affinity']), on=['customer_id', 'category_l1'], how='left')
            .join(u_pref_cat_lazy, on='customer_id', how='left')
            .join(u_pref_brand_lazy, on=['customer_id', 'category_l1'], how='left')
            .join(u_loc_lazy, on='customer_id', how='left')
            .join(loc_item_pop_lazy, on=['location', 'item_id'], how='left')
        )
        
        ds_lazy = ds_lazy.with_columns([
            pl.when(pl.col('category_l1') == pl.col('pref_cat_l1')).then(1).otherwise(0).alias('ui_is_primary_cat'),
            pl.when(pl.col('brand') == pl.col('pref_brand')).then(1).otherwise(0).alias('ui_is_preferred_brand'),
            (pl.col('i_ref_price') - pl.col('u_avg_price')).abs().alias('ui_price_diff'),
            (pl.col('i_ref_price') / (pl.col('u_avg_price') + 1e-5)).alias('ui_price_ratio'),
            (pl.col('item_age_proxy') - pl.col('u_avg_age_proxy')).alias('ui_size_age_diff'),
            (pl.col('item_age_proxy') / (pl.col('u_avg_age_proxy') + 1e-5)).alias('ui_size_age_ratio'),
            
            pl.when(pl.col('category_l1').is_in(['Thời trang', 'Đồ chơi & Sách', 'Phụ kiện']) & pl.col('ui_total_qty').is_not_null())
              .then(1).otherwise(0).alias('ui_already_bought_discretionary'),
              
            pl.when(pl.col('category_l1').is_in(['Thời trang', 'Đồ chơi & Sách', 'Phụ kiện']) & (pl.col('ui_loc_sales').fill_null(0) == 0))
              .then(1).otherwise(0).alias('ui_loc_sparsity_penalty')
        ]).drop(['pref_cat_l1', 'pref_brand', 'brand', 'location', 'category_l1'])
        
        # Trigger actual lazy DAG collection
        ds_collected = ds_lazy.collect()
        
        num_cols = [c for c in ds_collected.columns if c not in ['customer_id', 'item_id']]
        ds_collected = ds_collected.with_columns(pl.col(num_cols).fill_null(0))
        
        X_sub = ds_collected.select(all_feats).to_numpy()
        ds_collected = ds_collected.with_columns(pl.Series(name='pred', values=model.predict(X_sub)))
        
        top10 = ds_collected.group_by('customer_id').agg(
            pl.col('item_id').sort_by('pred', descending=True).head(10)
        )
        
        c_ids = top10['customer_id'].to_numpy()
        item_lists = top10['item_id'].to_list()
        
        for customer_id, items in zip(c_ids, item_lists):
            key_bytes = pickle.dumps(int(customer_id), protocol=4)
            val_bytes = pickle.dumps(items, protocol=4)
            
        print("Chunk processed successfully without any lazy or mismatch error!")
        break

test_submission_pkl(df_raw, items_df, target_users, model, all_feats, Path('test_sub.pkl'))
print("RIGOROUS SYNTAX AND SCHEMA INTERSECTION TEST COMPLETED SUCCESSFULLY!")
