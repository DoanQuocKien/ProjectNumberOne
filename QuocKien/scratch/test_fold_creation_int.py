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

print("Loading Data & Mapping Strings to Integers...")

items_raw = pl.read_parquet(I_PATH)
item_mapping = items_raw.select('item_id').unique().with_row_index("item_int_id")
idx2item = dict(zip(item_mapping['item_int_id'], item_mapping['item_id']))

items_df = items_raw.join(item_mapping, on='item_id', how='left').drop('item_id').rename({'item_int_id': 'item_id'}).select([
    pl.col('item_id').cast(pl.Int32), # Lightweight integer!
    pl.col('category').cast(pl.Utf8),
    pl.col('category_l1').cast(pl.Utf8),
    pl.col('category_l2').cast(pl.Utf8),
    pl.col('category_l3').cast(pl.Utf8),
    pl.col('brand').cast(pl.Utf8),
    pl.col('size').cast(pl.Utf8)
])

df_raw = pl.read_parquet(T_PATH, n_rows=200000).join(item_mapping, on='item_id', how='inner').drop('item_id').rename({'item_int_id': 'item_id'}).select([
    pl.col('customer_id').cast(pl.Int64), # Keep as Int64 to avoid overflow
    pl.col('item_id').cast(pl.Int32),     # Lightweight integer!
    pl.col('quantity').cast(pl.Int32),
    pl.col('price').cast(pl.Float32),
    # Directly cast location from Int32 to Int16
    pl.col('location').cast(pl.Int16), 
    pl.col('updated_date').cast(pl.Datetime).alias('event_ts')
]).drop_nulls(subset=['item_id', 'customer_id']).with_columns([
    pl.col('event_ts').dt.month().alias('month').cast(pl.Int8),
    pl.col('event_ts').dt.weekday().alias('dow').cast(pl.Int8)
])

# Clean up temporary frames
del items_raw, item_mapping
gc.collect()

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
    if re.search(r'(\\*|x\\d|cm)', clean_text): return 0.5
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
        i_arr = np.array(self.idx2i, dtype=np.int32)
        if u_idx:
            chunk = 4000
            c_svd, c_i2i = [], []
            for i in range(0, len(u_idx), chunk):
                idx_chunk = u_idx[i:i+chunk]
                u_b = np.array(t_u[i:i+chunk], dtype=np.int64)
                
                scores_svd = self.u_emb[idx_chunk] @ self.i_emb.T
                t60_idx = np.argpartition(-scores_svd, 60, axis=1)[:, :60]
                t60_scores = np.take_along_axis(-scores_svd, t60_idx, axis=1)
                t60_sort = np.argsort(t60_scores, axis=1)
                t60 = np.take_along_axis(t60_idx, t60_sort, axis=1)
                
                c_svd.append(pl.DataFrame({
                    'customer_id': pl.Series(np.repeat(u_b, 60), dtype=pl.Int64),
                    'item_id': pl.Series(i_arr[t60.flatten()], dtype=pl.Int32)
                }))
                
                scores_i2i = self.mtx[idx_chunk].dot(self.i2i_sim).toarray()
                t80_idx = np.argpartition(-scores_i2i, 80, axis=1)[:, :80]
                t80_scores = np.take_along_axis(-scores_i2i, t80_idx, axis=1)
                t80_sort = np.argsort(t80_scores, axis=1)
                t80 = np.take_along_axis(t80_idx, t80_sort, axis=1)
                
                mask = np.take_along_axis(scores_i2i, t80, axis=1) > 0
                c_i2i.append(pl.DataFrame({
                    'customer_id': pl.Series(np.repeat(u_b, 80)[mask.flatten()], dtype=pl.Int64),
                    'item_id': pl.Series(i_arr[t80.flatten()][mask.flatten()], dtype=pl.Int32)
                }))
            cands['svd'] = pl.concat(c_svd).unique() if c_svd else pl.DataFrame(schema={'customer_id': pl.Int64, 'item_id': pl.Int32})
            cands['i2i'] = pl.concat(c_i2i).unique() if c_i2i else pl.DataFrame(schema={'customer_id': pl.Int64, 'item_id': pl.Int32})
            
        u_cat_top = self.history_df.join(target_users_df, on='customer_id', how='inner')\
            .join(self.items_df.select(['item_id', 'category_l1']), on='item_id')\
            .group_by(['customer_id', 'category_l1']).len().sort('len', descending=True).group_by('customer_id').head(1)
        
        cat_global_top = self.history_df.filter(pl.col('event_ts') >= self.max_ts - pl.duration(days=30))\
            .join(self.items_df.select(['item_id', 'category_l1']), on='item_id')\
            .group_by(['category_l1', 'item_id']).len().sort('len', descending=True).group_by('category_l1').head(10)
            
        cands['cat_top'] = u_cat_top.join(cat_global_top, on='category_l1').select(['customer_id', 'item_id'])

        cands_list = [df for df in cands.values() if df is not None and df.height > 0]
        if not cands_list:
            return pl.DataFrame(schema={'customer_id': pl.Int64, 'item_id': pl.Int32})
            
        all_c = pl.concat(cands_list).unique()
        
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

# Cell 3 Function
def create_dataset_v12(history_df, target_df, items_df, sample_users=2000, n_negatives=150):
    retriever = V12Retriever(history_df, items_df)
    max_ts = history_df['event_ts'].max()
    
    valid_users = target_df['customer_id'].unique().to_list()
    if len(valid_users) > sample_users:
        np.random.seed(SEED)
        valid_users = list(np.random.choice(valid_users, sample_users, replace=False))
        
    print(f"Retrieving candidates for {len(valid_users)} users...")
    ds = retriever.get_candidates(valid_users)
    
    if ds.is_empty():
        return pl.DataFrame(schema={
            'customer_id': pl.Int64, 'item_id': pl.Int32, 'target': pl.Int8,
            'u_unique_items': pl.Int32, 'u_total_qty': pl.Int32, 'u_avg_price': pl.Float32, 'u_price_std': pl.Float32,
            'u_tenure_days': pl.Int32, 'u_exploration_ratio': pl.Float32, 'u_brand_hhi': pl.Float32,
            'u_cat_hhi': pl.Float32, 'u_avg_age_proxy': pl.Float32, 'i_unique_users': pl.Int32,
            'i_total_qty': pl.Int32, 'i_hubs_count': pl.Int32, 'i_ref_price': pl.Float32, 'i_repeat_rate': pl.Float32,
            'ui_total_qty': pl.Int32, 'ui_recency_days': pl.Int32, 'item_momentum': pl.Float32,
            'item_age_proxy': pl.Float32, 'u_cat_affinity': pl.Float32, 'ui_is_primary_cat': pl.Int8,
            'ui_is_preferred_brand': pl.Int8, 'ui_price_diff': pl.Float32, 'ui_price_ratio': pl.Float32,
            'ui_size_age_diff': pl.Float32, 'ui_size_age_ratio': pl.Float32, 'ui_already_bought_discretionary': pl.Int8,
            'ui_loc_sparsity_penalty': pl.Int8, 'ui_loc_sales': pl.Int32
        })
        
    positives = target_df.filter(pl.col('customer_id').is_in(valid_users)).select(['customer_id', 'item_id']).unique().with_columns(pl.lit(1).alias('target'))
    ds = ds.join(positives, on=['customer_id', 'item_id'], how='left')
    
    # Negative sampling cap
    negatives = ds.filter(pl.col('target').is_null())
    pos_only = ds.filter(pl.col('target') == 1)
    
    neg_sampled = negatives.with_columns(pl.lit(np.random.rand(len(negatives))).alias('_r'))\
                           .sort(['customer_id', '_r'])\
                           .group_by('customer_id').head(n_negatives)\
                           .drop('_r')
                           
    ds = pl.concat([pos_only, neg_sampled]).with_columns(pl.col('target').fill_null(0).cast(pl.Int8))
    
    print("Extracting features...")
    # Precompute features
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

    # Join features onto ds
    ds = ds.join(u_prof, on='customer_id', how='left')
    ds = ds.join(i_prof, on='item_id', how='left')
    ds = ds.join(ui_hist, on=['customer_id', 'item_id'], how='left')
    ds = ds.join(items_df.select(['item_id', 'item_age_proxy', 'brand', 'category_l1'] + [f'{c}_id' for c in cat_cols]), on='item_id', how='left')
    ds = ds.join(momentum.select(['item_id', 'item_momentum']), on='item_id', how='left')
    ds = ds.join(u_cat.select(['customer_id', 'category_l1', 'u_cat_affinity']), on=['customer_id', 'category_l1'], how='left')
    ds = ds.join(u_pref_cat, on='customer_id', how='left')
    ds = ds.join(u_pref_brand, on=['customer_id', 'category_l1'], how='left')
    
    ds = ds.with_columns([
        pl.when(pl.col('category_l1') == pl.col('pref_cat_l1')).then(1).otherwise(0).alias('ui_is_primary_cat'),
        pl.when(pl.col('brand') == pl.col('pref_brand')).then(1).otherwise(0).alias('ui_is_preferred_brand'),
        (pl.col('i_ref_price') - pl.col('u_avg_price')).abs().alias('ui_price_diff'),
        (pl.col('i_ref_price') / (pl.col('u_avg_price') + 1e-5)).alias('ui_price_ratio')
    ]).drop(['pref_cat_l1', 'pref_brand', 'brand'])
    
    ds = ds.join(u_loc, on='customer_id', how='left')
    ds = ds.join(loc_item_pop, on=['location', 'item_id'], how='left').drop('location')
    
    ds = ds.with_columns([
        (pl.col('item_age_proxy') - pl.col('u_avg_age_proxy')).alias('ui_size_age_diff'),
        (pl.col('item_age_proxy') / (pl.col('u_avg_age_proxy') + 1e-5)).alias('ui_size_age_ratio'),
        pl.when(pl.col('category_l1').is_in(['Thời trang', 'Đồ chơi & Sách', 'Phụ kiện']) & pl.col('ui_total_qty').is_not_null())\
          .then(1).otherwise(0).alias('ui_already_bought_discretionary'),
        pl.when(pl.col('category_l1').is_in(['Thời trang', 'Đồ chơi & Sách', 'Phụ kiện']) & (pl.col('ui_loc_sales').fill_null(0) == 0))\
          .then(1).otherwise(0).alias('ui_loc_sparsity_penalty')
    ])

    num_cols = [c for c in ds.columns if c not in ['customer_id', 'item_id', 'category_l1']]
    ds = ds.with_columns([
        pl.col(num_cols).fill_null(0)
    ]).drop('category_l1')
    
    return ds.sort('customer_id')

# Running real fold generation slices!
print("Creating temporal fold 1 (month 3 and 4) using valid target users...")
h1 = df_raw.filter(pl.col('month') <= 3) # month 3 slice
t1 = df_raw.filter(pl.col('month') == 4) # month 4 slice
f1 = create_dataset_v12(h1, t1, items_df, sample_users=2000, n_negatives=150)
print(f"Temporal fold 1 generated successfully! Shape: {f1.shape}")

print("RIGOROUS FOLD CREATION TEST COMPLETED SUCCESSFULLY!")
