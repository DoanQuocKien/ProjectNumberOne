import json

nb_path = r'd:\CS116\ProjectNumberOne\QuocKien\pir_pipeline_v12_submission.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell_2_source = """class V12Retriever:
    def __init__(self, history_df, items_df):
        self.history_df = history_df
        self.items_df = items_df
        self.max_ts = history_df['event_ts'].max()
        
        # Pre-compute item stock locations and price reference
        self.item_locs = history_df.group_by('item_id').agg(pl.col('location').unique().alias('item_hubs'))
        self.item_prcs = history_df.group_by('item_id').agg(pl.col('price').median().alias('item_p'))
        
        # Source 1: Global/Local Hot (Capped at 150/80)
        self.global_top = history_df.filter(pl.col('event_ts') >= self.max_ts - pl.duration(days=14))\\
            .group_by('item_id').len().sort('len', descending=True).head(150).select('item_id')
            
        self.local_heroes = history_df.filter(pl.col('event_ts') >= self.max_ts - pl.duration(days=60))\\
            .group_by(['location', 'item_id']).len()\\
            .sort(['location', 'len'], descending=[False, True])\\
            .group_by('location').head(80)
            
        # Source 2: Replenishment Cycle (Mathematically proven fast formula)
        self.replenish = history_df.group_by(['customer_id', 'item_id']).agg([
            pl.col('event_ts').count().alias('buy_count'),
            pl.col('event_ts').min().alias('first_buy'),
            pl.col('event_ts').max().alias('last_buy')
        ]).filter(pl.col('buy_count') > 1)\\
          .with_columns(((pl.col('last_buy') - pl.col('first_buy')).dt.total_days() / (pl.col('buy_count') - 1)).alias('avg_gap'))
        
        # Source 3: CF (SVD + I2I)
        self._build_cf()
        
    def _build_cf(self):
        # Build Sparse interaction matrix using past 180 days
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
        
        # Dynamic components
        n_comp = min(100, len(i_map) - 1)
        self.svd = TruncatedSVD(n_components=n_comp, random_state=SEED)
        self.u_emb = self.svd.fit_transform(self.mtx)
        self.i_emb = self.svd.components_.T
        
        # Dynamic Item-Item Cosine Similarity Matrix
        norm_m = normalize(self.mtx, norm='l2', axis=0)
        sim = (norm_m.T.dot(norm_m)).astype(np.float32)
        sim.setdiag(0)
        
        # Dense Row-wise partitioning (Retain only top 150 similarity neighbours)
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
        
        # Candidate Source 1: Purchase History
        hist_s = self.history_df.join(target_users_df, on='customer_id', how='inner')
        cands['hist'] = hist_s.select(['customer_id', 'item_id']).unique()
        
        # Candidate Source 2: Replenishment items
        due = self.replenish.join(target_users_df, on='customer_id', how='inner')\\
            .with_columns((self.max_ts - pl.col('last_buy')).dt.total_days().alias('days_since'))\\
            .filter(pl.col('days_since') >= pl.col('avg_gap') * 0.8)\\
            .select(['customer_id', 'item_id'])
        cands['repl'] = due
        
        # Candidate Source 3: Global Hot Items (Restored to 150)
        cands['global'] = target_users_df.join(self.global_top.with_columns(pl.lit(1).alias('_k')), how='cross').drop('_k')
        
        # Candidate Source 4: Local Location Hot Items (Restored to 80)
        user_loc = hist_s.group_by('customer_id').agg(pl.col('location').mode().first().alias('location'))
        cands['local'] = user_loc.join(self.local_heroes, on='location').select(['customer_id', 'item_id']).unique()
        
        # Candidate Source 5: SVD + I2I CF recommendations (Batched processing to fit in 13 GB RAM)
        u_idx = [self.u2idx[u] for u in target_users if u in self.u2idx]
        t_u = [u for u in target_users if u in self.u2idx]
        i_arr = np.array(self.idx2i)
        if u_idx:
            chunk = 4000
            c_svd, c_i2i = [], []
            for i in range(0, len(u_idx), chunk):
                idx_chunk = u_idx[i:i+chunk]
                u_b = np.array(t_u[i:i+chunk])
                
                # SVD top 60 candidates
                scores_svd = self.u_emb[idx_chunk] @ self.i_emb.T
                t60_idx = np.argpartition(-scores_svd, 60, axis=1)[:, :60]
                t60_scores = np.take_along_axis(-scores_svd, t60_idx, axis=1)
                t60_sort = np.argsort(t60_scores, axis=1)
                t60 = np.take_along_axis(t60_idx, t60_sort, axis=1)
                
                c_svd.append(pl.DataFrame({
                    'customer_id': pl.Series(np.repeat(u_b, 60), dtype=pl.Int64),
                    'item_id': pl.Series(i_arr[t60.flatten()], dtype=pl.Int32)
                }))
                
                # I2I top 80 candidates
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
            
        # Candidate Source 6: Target Category Popular Items
        u_cat_top = self.history_df.join(target_users_df, on='customer_id', how='inner')\\
            .join(self.items_df.select(['item_id', 'category_l1']), on='item_id')\\
            .group_by(['customer_id', 'category_l1']).len().sort('len', descending=True).group_by('customer_id').head(1)
        
        cat_global_top = self.history_df.filter(pl.col('event_ts') >= self.max_ts - pl.duration(days=30))\\
            .join(self.items_df.select(['item_id', 'category_l1']), on='item_id')\\
            .group_by(['category_l1', 'item_id']).len().sort('len', descending=True).group_by('category_l1').head(10)
            
        cands['cat_top'] = u_cat_top.join(cat_global_top, on='category_l1').select(['customer_id', 'item_id'])

        # Combine all candidate sources
        all_c = pl.concat([df for df in cands.values() if df is not None and df.height > 0]).unique()
        
        # Apply strict location-stock and average-price filter
        uh = user_loc.rename({'location': 'loc'})
        up = hist_s.group_by('customer_id').agg(pl.col('price').mean().alias('avg_p'))
        f = all_c.join(up, on='customer_id', how='left')\\
                 .join(self.item_prcs, on='item_id', how='left')\\
                 .filter((pl.col('item_p') <= pl.col('avg_p') * 6) | (pl.col('avg_p').is_null()))\\
                 .select(['customer_id', 'item_id'])
        item_loc_flat = self.item_locs.explode('item_hubs').rename({'item_hubs': 'loc'})
        filtered_cands = f.join(uh, on='customer_id', how='left')\\
                          .join(item_loc_flat, on=['item_id', 'loc'], how='inner')\\
                          .select(['customer_id', 'item_id'])
                          
        return filtered_cands"""

nb['cells'][2]['source'] = [line + "\n" for line in cell_2_source.split("\n")]
if nb['cells'][2]['source']:
    nb['cells'][2]['source'][-1] = nb['cells'][2]['source'][-1].rstrip("\n")

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Cell 2 (V12Retriever) successfully updated with type-safe explicit Int32 casting!")
