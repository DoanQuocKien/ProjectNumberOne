import polars as pl
import numpy as np
import os
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

# Mock paths (standard for this workspace)
T_PATH = '../transaction_full_2025.parquet'
I_PATH = '../items.parquet'

def check_recall():
    print("=== Analyzing Candidate Recall (V6 Logic) ===")
    df_raw = pl.read_parquet(T_PATH).select([
        pl.col('customer_id').cast(pl.Int64),
        pl.col('item_id').cast(pl.Utf8),
        pl.col('updated_date').cast(pl.Datetime).alias('event_ts'),
        pl.col('location').cast(pl.Utf8),
        pl.col('quantity').cast(pl.Float32)
    ]).with_columns(pl.col('event_ts').dt.month().alias('month'))

    items_df = pl.read_parquet(I_PATH).select(['item_id', 'category_l1', 'brand'])
    items_df = items_df.with_columns(pl.col('item_id').cast(pl.Utf8))
    
    # Prep maps
    cat_map = items_df.select(['item_id', 'category_l1']).unique()
    
    # Train/Val split
    df_train = df_raw.filter(pl.col('month') <= 10)
    df_val_truth = df_raw.filter(pl.col('month') == 11)
    
    target_users = df_val_truth['customer_id'].unique().head(10000).to_list()
    truth = df_val_truth.filter(pl.col('customer_id').is_in(target_users)).select(['customer_id', 'item_id']).unique()
    
    print(f"Target Users: {len(target_users)}, Truth Pairs: {truth.height}")

    # Retriever Logic V6
    hist_sampled = df_train.filter(pl.col('customer_id').is_in(target_users))
    ints = hist_sampled.group_by(['customer_id', 'item_id']).agg(pl.col('quantity').sum().alias('w'))
    ints = ints.with_columns([pl.col('customer_id').rank('dense').cast(pl.Int64).alias('u_idx')-1, pl.col('item_id').rank('dense').cast(pl.Int32).alias('i_idx')-1])
    u_map = ints.select(['customer_id', 'u_idx']).unique()
    i_map = ints.select(['item_id', 'i_idx']).unique()
    u2idx = dict(zip(u_map['customer_id'], u_map['u_idx']))
    items_list = i_map.sort('i_idx')['item_id'].to_list()
    matrix = csr_matrix((ints['w'].to_numpy(), (ints['u_idx'].to_numpy(), ints['i_idx'].to_numpy())), shape=(u_map.height, i_map.height), dtype=np.float32)
    
    # I2I
    bin_m = matrix.copy()
    bin_m.data = np.ones_like(bin_m.data)
    i2i = bin_m.T.dot(bin_m).astype(np.float32)
    i2i.setdiag(0)
    
    # Candidates
    df_rep = hist_sampled.select(['customer_id', 'item_id']).unique()
    
    # I2I Top 40
    c_i2i = []
    i_arr = np.array(items_list)
    chunk_size = 1000
    t_idx = [u2idx[u] for u in target_users if u in u2idx]
    t_u = [u for u in target_users if u in u2idx]
    
    for i in range(0, len(t_idx), chunk_size):
        idx = t_idx[i:i+chunk_size]
        u = np.array(t_u[i:i+chunk_size])
        s_i = matrix[idx].dot(i2i).toarray()
        t40 = np.argsort(-s_i, axis=1)[:, :40]
        m = np.take_along_axis(s_i, t40, axis=1) > 0
        c_i2i.append(pl.DataFrame({
            'customer_id': pl.Series(np.repeat(u, 40)[m.flatten()], dtype=pl.Int64),
            'item_id': i_arr[t40.flatten()][m.flatten()]
        }))
    df_i2i = pl.concat(c_i2i).unique()
    
    # Total candidates before filtering
    all_cands = pl.concat([df_rep, df_i2i]).unique()
    
    # Recall Check
    hits = all_cands.join(truth, on=['customer_id', 'item_id'], how='inner')
    recall = hits.height / truth.height
    print(f"Recall (Unfiltered): {recall:.4f} ({hits.height}/{truth.height})")

    # Filtering Logic (Location)
    item_locations = df_train.group_by('item_id').agg(pl.col('location').unique().alias('hubs'))
    user_hubs = df_train.filter(pl.col('customer_id').is_in(target_users)).group_by('customer_id').agg(pl.col('location').mode().first().alias('loc'))
    
    item_loc_flat = item_locations.explode('hubs').rename({'hubs': 'loc'})
    filtered_cands = (all_cands.join(user_hubs, on='customer_id', how='left')
                     .join(item_loc_flat, on=['item_id', 'loc'], how='inner'))
    
    hits_f = filtered_cands.join(truth, on=['customer_id', 'item_id'], how='inner')
    recall_f = hits_f.height / truth.height
    print(f"Recall (Location Filtered): {recall_f:.4f} ({hits_f.height}/{truth.height})")
    
    # Source I2I Recall
    h = df_i2i.join(truth, on=['customer_id', 'item_id'], how='inner').height
    print(f"  Source I2I Recall: {h/truth.height:.4f}")

    # NEW: Trending Recall Check
    max_ts = df_train['event_ts'].max()
    vol_30d = df_train.filter(pl.col('event_ts') >= max_ts - pl.duration(days=30)).group_by('item_id').len()
    top_50_trending = vol_30d.sort('len', descending=True).head(50).select('item_id')
    
    df_trend = pl.DataFrame({'customer_id': target_users}).join(top_50_trending.with_columns(pl.lit(1).alias('_k')), how='cross').drop('_k')
    h_trend = df_trend.join(truth, on=['customer_id', 'item_id'], how='inner').height
    print(f"  Source Top 50 Trending Recall: {h_trend/truth.height:.4f}")

    # NEW: Multi-Category Popularity (Top 3 Cats)
    user_top_cats = df_train.filter(pl.col('customer_id').is_in(target_users)).join(cat_map, on='item_id').group_by(['customer_id', 'category_l1']).len().sort(['customer_id', 'len'], descending=[False, True]).group_by('customer_id').head(3)
    
    cat_pop = df_train.filter(pl.col('event_ts') >= max_ts - pl.duration(days=60)).join(cat_map, on='item_id').group_by(['category_l1', 'item_id']).len().sort(['category_l1', 'len'], descending=[False, True]).group_by('category_l1').head(20)
    
    df_multi_cat = user_top_cats.join(cat_pop, on='category_l1').select(['customer_id', 'item_id']).unique()
    h_mcat = df_multi_cat.join(truth, on=['customer_id', 'item_id'], how='inner').height
    print(f"  Source Top-3 Cat Popularity Recall: {h_mcat/truth.height:.4f}")

    # Combined Recall
    combined = pl.concat([all_cands, df_trend, df_multi_cat]).unique()
    print(f"Combined Potential Recall (V7 Prototype): {combined.join(truth, on=['customer_id', 'item_id'], how='inner').height / truth.height:.4f}")


if __name__ == "__main__":
    check_recall()
