import polars as pl
import numpy as np
import lightgbm as lgb
import os
import gc
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import warnings
warnings.filterwarnings('ignore')

SEED = 42
T_PATH = r'd:\CS116\ProjectNumberOne\transaction_full_2025.parquet'
I_PATH = r'd:\CS116\ProjectNumberOne\items.parquet'

print("Loading Data...")
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

class fast_Retriever:
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
            
        # Mathematically optimized replenishment
        self.replenish = history_df.group_by(['customer_id', 'item_id']).agg([
            pl.col('event_ts').count().alias('buy_count'),
            pl.col('event_ts').min().alias('first_buy'),
            pl.col('event_ts').max().alias('last_buy')
        ]).filter(pl.col('buy_count') > 1)\
          .with_columns(((pl.col('last_buy') - pl.col('first_buy')).dt.total_days() / (pl.col('buy_count') - 1)).alias('avg_gap'))
        
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

def create_dataset_fast(history_df, truth_df, items_df, sample_users=20000, n_negatives=150):
    if sample_users:
        valid_u = history_df['customer_id'].unique().shuffle(seed=SEED).head(sample_users).to_list()
    else:
        valid_u = history_df['customer_id'].unique().to_list()
    
    retriever = fast_Retriever(history_df, items_df, valid_u)
    ds = retriever.get_candidates()
    
    if truth_df is not None:
        truth = truth_df.filter(pl.col('customer_id').is_in(valid_u)).select(['customer_id', 'item_id']).unique()
        ds = ds.join(truth.with_columns(pl.lit(1).cast(pl.Int8).alias('target')), on=['customer_id', 'item_id'], how='left').fill_null(0)
        missed = truth.join(ds, on=['customer_id', 'item_id'], how='anti').with_columns(pl.lit(1).cast(pl.Int8).alias('target'))
        ds = pl.concat([ds, missed]).unique(subset=['customer_id', 'item_id'])
        if n_negatives:
            pos = ds.filter(pl.col('target') == 1)
            neg = ds.filter(pl.col('target') == 0).sample(fraction=1.0, shuffle=True, seed=SEED).group_by('customer_id').head(n_negatives)
            ds = pl.concat([pos, neg])
        ds = ds.sort(['customer_id', 'target'], descending=[False, True])
    else:
        ds = ds.sort('customer_id')
    
    max_ts = history_df['event_ts'].max()
    
    u_prof = history_df.group_by('customer_id').agg([
        pl.col('item_id').n_unique().alias('u_unique_items'),
        pl.col('quantity').sum().alias('u_total_qty'),
        pl.col('price').mean().alias('u_avg_price'),
        pl.col('price').std().alias('u_price_std'),
        (max_ts - pl.col('event_ts').min()).dt.total_days().alias('u_tenure_days'),
        (pl.col('item_id').n_unique() / pl.col('quantity').sum().clip(1)).alias('u_exploration_ratio')
    ])
    
    i_prof = history_df.group_by('item_id').agg([
        pl.col('customer_id').n_unique().alias('i_unique_users'),
        pl.col('quantity').sum().alias('i_total_qty'),
        pl.col('location').n_unique().alias('i_hubs_count'),
        pl.col('price').median().alias('i_ref_price')
    ])
    
    ui_hist = history_df.filter(pl.col('customer_id').is_in(valid_u)).group_by(['customer_id', 'item_id']).agg([
        pl.col('quantity').sum().alias('ui_total_qty'),
        (max_ts - pl.col('event_ts').max()).dt.total_days().alias('ui_recency_days')
    ])
    
    vol_7d = history_df.filter(pl.col('event_ts') >= max_ts - pl.duration(days=7)).group_by('item_id').len().rename({'len': 'v7'})
    vol_21d = history_df.filter(pl.col('event_ts') >= max_ts - pl.duration(days=21)).group_by('item_id').len().rename({'len': 'v21'})
    momentum = vol_7d.join(vol_21d, on='item_id', how='left').with_columns((pl.col('v7') / (pl.col('v21') / 3.0 + 1)).alias('item_momentum'))
    
    u_cat = history_df.join(items_df.select(['item_id', 'category_l1']), on='item_id')\
        .group_by(['customer_id', 'category_l1']).len()\
        .with_columns((pl.col('len') / pl.col('len').sum().over('customer_id')).alias('u_cat_affinity'))
    
    ds = ds.join(u_prof, on='customer_id', how='left')
    ds = ds.join(i_prof, on='item_id', how='left')
    ds = ds.join(ui_hist, on=['customer_id', 'item_id'], how='left')
    ds = ds.join(items_df.select(['item_id', 'item_age_proxy'] + [f'{c}_id' for c in cat_cols] + ['category_l1']), on='item_id', how='left')
    ds = ds.join(momentum.select(['item_id', 'item_momentum']), on='item_id', how='left')
    ds = ds.join(u_cat.select(['customer_id', 'category_l1', 'u_cat_affinity']), on=['customer_id', 'category_l1'], how='left')
    
    num_cols = [c for c in ds.columns if c not in ['customer_id', 'item_id', 'category_l1']]
    ds = ds.with_columns([
        pl.col(num_cols).fill_null(0)
    ]).drop('category_l1')
    
    return ds

print("Preparing small training fold...")
train_m = df_raw.filter(pl.col('month') <= 10)
val_m = df_raw.filter(pl.col('month') == 11)

train_ds = create_dataset_fast(train_m, val_m, items_df, sample_users=25000, n_negatives=150)
cat_feat_ids = [f'{c}_id' for c in cat_cols]
all_feats = ['u_unique_items', 'u_total_qty', 'u_avg_price', 'u_price_std', 'u_tenure_days', 'u_exploration_ratio', 
             'i_unique_users', 'i_total_qty', 'i_hubs_count', 'i_ref_price', 
             'ui_total_qty', 'ui_recency_days', 'item_momentum', 'item_age_proxy', 'u_cat_affinity'] + cat_feat_ids

def prep_lgb(ds):
    X = ds.select(all_feats).to_numpy().astype(np.float32)
    y = ds['target'].to_numpy().astype(np.int8)
    g = ds.group_by('customer_id', maintain_order=True).len()['len'].to_numpy()
    return X, y, g

X_tr, y_tr, g_tr = prep_lgb(train_ds)

params = {
    'objective': 'lambdarank', 'metric': 'ndcg', 'ndcg_eval_at': [10], 'verbosity': -1,
    'learning_rate': 0.05, 'num_leaves': 127, 'max_depth': 10, 'min_data_in_leaf': 150,
    'device': 'cpu', 'random_state': SEED
}
dtrain = lgb.Dataset(X_tr, y_tr, group=g_tr, categorical_feature=cat_feat_ids)
model = lgb.train(params, dtrain, num_boost_round=150)

print("\nFinal Evaluation (Month 12)...")
test_ds = create_dataset_fast(df_raw.filter(pl.col('month') <= 11), df_raw.filter(pl.col('month') == 12), items_df, sample_users=20000, n_negatives=None)
X_ts, y_ts, _ = prep_lgb(test_ds)
test_ds = test_ds.with_columns(pl.Series(name='pred', values=model.predict(X_ts)))

def evaluate(df, col):
    top10 = df.sort(['customer_id', col], descending=[False, True]).group_by('customer_id', maintain_order=True).head(10)
    truth_map = df_raw.filter(pl.col('month') == 12).filter(pl.col('customer_id').is_in(top10['customer_id'].unique().to_list())).group_by('customer_id').agg(pl.col('item_id'))
    truth_dict = {row[0]: set(row[1]) for row in truth_map.iter_rows()}
    pred_dict = {row[0]: list(row[1]) for row in top10.group_by('customer_id', maintain_order=True).agg(pl.col('item_id')).iter_rows()}
    h, m, p = 0, 0.0, 0.0
    for uid, truth in truth_dict.items():
        preds = pred_dict.get(uid, [])
        hits = [pr for pr in preds if pr in truth]
        h += len(hits); p += len(hits)/10.0
        for i, pr in enumerate(preds):
            if pr in truth: m += 1.0/(i+1); break
    n = max(1, len(truth_dict))
    return {'Hits': h, 'Precision@10': p/n, 'MRR': m/n, 'Users': n}

baseline_score = evaluate(test_ds, 'pred')
print(f"Baseline Score: {baseline_score}")
