import polars as pl
import numpy as np
import pandas as pd
import lightgbm as lgb
import os
import gc
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import sys

SEED = 42
T_PATH = r'd:\CS116\ProjectNumberOne\transaction_full_2025.parquet'
I_PATH = r'd:\CS116\ProjectNumberOne\items.parquet'

print("1. Loading Data...")
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

class debug_Retriever:
    def __init__(self, history_df, items_df, target_users):
        self.history_df = history_df
        self.items_df = items_df
        self.max_ts = history_df['event_ts'].max()
        self.target_users = target_users
        
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
        
        print("   Initializing CF...")
        self._build_cf()
        
    def _build_cf(self):
        hist = self.history_df.filter(pl.col('customer_id').is_in(self.target_users))
        u_map = hist['customer_id'].unique()
        i_map = self.history_df['item_id'].unique()
        
        u_df = pl.DataFrame({'customer_id': u_map, 'u_idx': np.arange(len(u_map), dtype=np.int64)})
        i_df = pl.DataFrame({'item_id': i_map, 'i_idx': np.arange(len(i_map), dtype=np.int32)})
        
        hist_indexed = hist.join(u_df, on='customer_id', how='inner').join(i_df, on='item_id', how='inner')
        rows = hist_indexed['u_idx'].to_numpy()
        cols = hist_indexed['i_idx'].to_numpy()
        data = np.ones(len(rows))
        
        self.mtx = csr_matrix((data, (rows, cols)), shape=(len(u_map), len(i_map)))
        self.u2idx = dict(zip(u_df['customer_id'], u_df['u_idx']))
        self.idx2i = i_map.to_list()
        
        self.svd = TruncatedSVD(n_components=64, random_state=SEED)
        self.u_emb = self.svd.fit_transform(self.mtx)
        self.i_emb = self.svd.components_.T
        
        norm_m = normalize(self.mtx, norm='l2', axis=0)
        self.i2i_sim = (norm_m.T.dot(norm_m)).astype(np.float32)
        self.i2i_sim.setdiag(0)

    def get_candidates(self):
        cands = {}
        hist_s = self.history_df.filter(pl.col('customer_id').is_in(self.target_users))
        
        cands['hist'] = hist_s.select(['customer_id', 'item_id']).unique()
        
        due = self.replenish.filter(pl.col('customer_id').is_in(self.target_users))\
            .with_columns((self.max_ts - pl.col('last_buy')).dt.total_days().alias('days_since'))\
            .filter(pl.col('days_since') >= pl.col('avg_gap') * 0.8)\
            .select(['customer_id', 'item_id'])
        cands['repl'] = due
        
        cands['global'] = pl.DataFrame({'customer_id': self.target_users}).join(self.global_top.with_columns(pl.lit(1).alias('_k')), how='cross').drop('_k')
        
        user_loc = hist_s.group_by('customer_id').agg(pl.col('location').mode().first().alias('location'))
        cands['local'] = user_loc.join(self.local_heroes, on='location').select(['customer_id', 'item_id']).unique()
        
        print("   Running CF inference...")
        u_idx = [self.u2idx[u] for u in self.target_users if u in self.u2idx]
        t_u = [u for u in self.target_users if u in self.u2idx]
        i_arr = np.array(self.idx2i)
        if u_idx:
            chunk = 4000
            c_svd, c_i2i = [], []
            for i in range(0, len(u_idx), chunk):
                idx_chunk = u_idx[i:i+chunk]
                u_b = np.array(t_u[i:i+chunk])
                
                scores_svd = self.u_emb[idx_chunk] @ self.i_emb.T
                t60 = np.argsort(-scores_svd, axis=1)[:, :60]
                c_svd.append(pl.DataFrame({'customer_id': pl.Series(np.repeat(u_b, 60), dtype=pl.Int64), 'item_id': i_arr[t60.flatten()]}))
                
                scores_i2i = self.mtx[idx_chunk].dot(self.i2i_sim).toarray()
                t80 = np.argsort(-scores_i2i, axis=1)[:, :80]
                mask = np.take_along_axis(scores_i2i, t80, axis=1) > 0
                c_i2i.append(pl.DataFrame({'customer_id': pl.Series(np.repeat(u_b, 80)[mask.flatten()], dtype=pl.Int64), 
                                          'item_id': i_arr[t80.flatten()][mask.flatten()]}))
            cands['svd'] = pl.concat(c_svd).unique() if c_svd else pl.DataFrame(schema={'customer_id': pl.Int64, 'item_id': pl.Utf8})
            cands['i2i'] = pl.concat(c_i2i).unique() if c_i2i else pl.DataFrame(schema={'customer_id': pl.Int64, 'item_id': pl.Utf8})
            
        u_cat_top = hist_s.join(self.items_df.select(['item_id', 'category_l1']), on='item_id')\
            .group_by(['customer_id', 'category_l1']).len().sort('len', descending=True).group_by('customer_id').head(1)
        
        cat_global_top = self.history_df.filter(pl.col('event_ts') >= self.max_ts - pl.duration(days=30))\
            .join(self.items_df.select(['item_id', 'category_l1']), on='item_id')\
            .group_by(['category_l1', 'item_id']).len().sort('len', descending=True).group_by('category_l1').head(10)
            
        cands['cat_top'] = u_cat_top.join(cat_global_top, on='category_l1').select(['customer_id', 'item_id'])

        all_c = pl.concat([df for df in cands.values() if df is not None and df.height > 0]).unique()
        return all_c

def test_pipeline():
    train_m = df_raw.filter(pl.col('month') <= 10)
    val_m = df_raw.filter(pl.col('month') == 11)
    
    valid_u = train_m['customer_id'].unique().shuffle(seed=SEED).head(5000).to_list()
    
    print("2. Running Retriever...")
    retriever = debug_Retriever(train_m, items_df, valid_u)
    ds = retriever.get_candidates()
    print(f"   Retrieved {ds.height} candidate rows.")
    
    print("3. Joining target labels...")
    truth = val_m.filter(pl.col('customer_id').is_in(valid_u)).select(['customer_id', 'item_id']).unique()
    ds = ds.join(truth.with_columns(pl.lit(1).cast(pl.Int8).alias('target')), on=['customer_id', 'item_id'], how='left').fill_null(0)
    missed = truth.join(ds, on=['customer_id', 'item_id'], how='anti').with_columns(pl.lit(1).cast(pl.Int8).alias('target'))
    ds = pl.concat([ds, missed]).unique(subset=['customer_id', 'item_id'])
    print(f"   Labels joined. Total rows: {ds.height}.")
    
    print("4. Doing negative sampling...")
    pos = ds.filter(pl.col('target') == 1)
    neg = ds.filter(pl.col('target') == 0).sample(fraction=1.0, shuffle=True, seed=SEED).group_by('customer_id').head(150)
    ds = pl.concat([pos, neg])
    ds = ds.sort(['customer_id', 'target'], descending=[False, True])
    print(f"   Sampling done. Pos: {pos.height}, Neg: {neg.height}, Total: {ds.height}.")
    
    print("5. Feature Engineering...")
    max_ts = train_m['event_ts'].max()
    
    u_prof = train_m.group_by('customer_id').agg([
        pl.col('item_id').n_unique().alias('u_unique_items'),
        pl.col('quantity').sum().alias('u_total_qty'),
        pl.col('price').mean().alias('u_avg_price'),
        pl.col('price').std().alias('u_price_std'),
        (max_ts - pl.col('event_ts').min()).dt.total_days().alias('u_tenure_days'),
        (pl.col('item_id').n_unique() / pl.col('quantity').sum().clip(1)).alias('u_exploration_ratio')
    ])
    
    i_prof = train_m.group_by('item_id').agg([
        pl.col('customer_id').n_unique().alias('i_unique_users'),
        pl.col('quantity').sum().alias('i_total_qty'),
        pl.col('location').n_unique().alias('i_hubs_count'),
        pl.col('price').median().alias('i_ref_price')
    ])
    
    ui_hist = train_m.filter(pl.col('customer_id').is_in(valid_u)).group_by(['customer_id', 'item_id']).agg([
        pl.col('quantity').sum().alias('ui_total_qty'),
        (max_ts - pl.col('event_ts').max()).dt.total_days().alias('ui_recency_days')
    ])
    
    vol_7d = train_m.filter(pl.col('event_ts') >= max_ts - pl.duration(days=7)).group_by('item_id').len().rename({'len': 'v7'})
    vol_21d = train_m.filter(pl.col('event_ts') >= max_ts - pl.duration(days=21)).group_by('item_id').len().rename({'len': 'v21'})
    momentum = vol_7d.join(vol_21d, on='item_id', how='left').with_columns((pl.col('v7') / (pl.col('v21') / 3.0 + 1)).alias('item_momentum'))
    
    u_cat = train_m.join(items_df.select(['item_id', 'category_l1']), on='item_id')\
        .group_by(['customer_id', 'category_l1']).len()\
        .with_columns((pl.col('len') / pl.col('len').sum().over('customer_id')).alias('u_cat_affinity'))
        
    print("   Joining features...")
    ds = ds.join(u_prof, on='customer_id', how='left')
    ds = ds.join(i_prof, on='item_id', how='left')
    ds = ds.join(ui_hist, on=['customer_id', 'item_id'], how='left')
    ds = ds.join(items_df.select(['item_id', 'item_age_proxy'] + [f'{c}_id' for c in cat_cols] + ['category_l1']), on='item_id', how='left')
    ds = ds.join(momentum.select(['item_id', 'item_momentum']), on='item_id', how='left')
    ds = ds.join(u_cat.select(['customer_id', 'category_l1', 'u_cat_affinity']), on=['customer_id', 'category_l1'], how='left')
    
    # Safe Fill by Type to prevent Categorical fill_null crash
    num_cols = [c for c in ds.columns if c not in ['customer_id', 'item_id', 'category_l1']]
    ds = ds.with_columns([
        pl.col(num_cols).fill_null(0)
    ]).drop('category_l1')
    
    print("   Features joined successfully!")
    
    print("6. Preparing LightGBM group sizes...")
    cat_feat_ids = [f'{c}_id' for c in cat_cols]
    all_feats = ['u_unique_items', 'u_total_qty', 'u_avg_price', 'u_price_std', 'u_tenure_days', 'u_exploration_ratio', 
                 'i_unique_users', 'i_total_qty', 'i_hubs_count', 'i_ref_price', 
                 'ui_total_qty', 'ui_recency_days', 'item_momentum', 'item_age_proxy', 'u_cat_affinity'] + cat_feat_ids
                 
    p = ds.to_pandas()
    X = p[all_feats]
    y = p['target']
    g = p.groupby('customer_id').size().values
    
    print("7. Training LightGBM...")
    dtrain = lgb.Dataset(X, y, group=g, categorical_feature=cat_feat_ids)
    params = {
        'objective': 'lambdarank', 'metric': 'ndcg', 'ndcg_eval_at': [10], 'verbosity': -1,
        'learning_rate': 0.05, 'num_leaves': 63, 'max_depth': 8, 'device': 'cpu', 'random_state': SEED
    }
    model = lgb.train(params, dtrain, num_boost_round=50)
    print("   Training successful!")

test_pipeline()
