import json

nb_path = r'd:\CS116\ProjectNumberOne\QuocKien\pir_pipeline_v12_submission.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find and update V12Retriever (Cell 2)
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "class V12Retriever:" in source:
            print("Found V12Retriever cell!")
            # Replace global_top and local_heroes limits
            source = source.replace("head(30).select('item_id')", "head(150).select('item_id')")
            source = source.replace("head(30)", "head(80)")
            cell['source'] = [line + "\n" for line in source.split("\n")]
            if cell['source']:
                cell['source'][-1] = cell['source'][-1].rstrip("\n")

# Find and update Optuna hyperparameters (Cell 3 / objective function)
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "learning_rate': trial.suggest_float('learning_rate'" in source:
            print("Found Optuna cell!")
            # Revert/optimize hyperparameters for robust longer learning to avoid early overfitting
            source = source.replace(
                "trial.suggest_float('learning_rate', 0.01, 0.08)",
                "trial.suggest_float('learning_rate', 0.005, 0.03)"
            )
            source = source.replace(
                "trial.suggest_int('num_leaves', 63, 511)",
                "trial.suggest_int('num_leaves', 127, 1023)"
            )
            source = source.replace(
                "trial.suggest_int('max_depth', 7, 15)",
                "trial.suggest_int('max_depth', 9, 20)"
            )
            source = source.replace(
                "trial.suggest_int('min_data_in_leaf', 50, 400)",
                "trial.suggest_int('min_data_in_leaf', 100, 1000)"
            )
            cell['source'] = [line + "\n" for line in source.split("\n")]
            if cell['source']:
                cell['source'][-1] = cell['source'][-1].rstrip("\n")

# Find and update make_chunked_submission_pkl (Cell 5)
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "def make_chunked_submission_pkl" in source:
            print("Found make_chunked_submission_pkl cell!")
            
            new_submission_code = """print("Executing Chunked OOM-Safe Submission Generation (Streaming to pkl)...")
h_full = df_raw.filter(pl.col('month') <= 12)

# Free raw loaded data to clear 1.5 GB memory!
del df_raw
gc.collect()

target_users = h_full['customer_id'].unique().to_list()

def make_chunked_submission_pkl(history_df, items_df, target_users, model, all_feats, output_path, chunk_size=20000):
    retriever = V12Retriever(history_df, items_df)
    
    max_ts = history_df['event_ts'].max()
    
    print("Pre-computing Global Profiles...")
    
    # 1. User Profiles & HHIs (Pre-computed once globally!)
    u_brand_counts = history_df.select(['customer_id', 'item_id']).join(items_df.select(['item_id', 'brand']), on='item_id')\\
        .group_by(['customer_id', 'brand']).len().rename({'len': 'brand_count'})
    u_brand_hhi = u_brand_counts.with_columns(
        (pl.col('brand_count') / pl.col('brand_count').sum().over('customer_id')).alias('brand_share')
    ).with_columns(
        (pl.col('brand_share') * pl.col('brand_share')).alias('brand_share_sq')
    ).group_by('customer_id').agg(pl.col('brand_share_sq').sum().alias('u_brand_hhi'))

    u_cat_counts = history_df.select(['customer_id', 'item_id']).join(items_df.select(['item_id', 'category_l1']), on='item_id')\\
        .group_by(['customer_id', 'category_l1']).len().rename({'len': 'cat_count'})
    u_cat_hhi = u_cat_counts.with_columns(
        (pl.col('cat_count') / pl.col('cat_count').sum().over('customer_id')).alias('cat_share')
    ).with_columns(
        (pl.col('cat_share') * pl.col('cat_share')).alias('cat_share_sq')
    ).group_by('customer_id').agg(pl.col('cat_share_sq').sum().alias('u_cat_hhi'))

    global_avg_age = items_df.filter(pl.col('item_age_proxy') >= 0)['item_age_proxy'].mean()
    if global_avg_age is None: global_avg_age = 1.0
    u_avg_age = history_df.select(['customer_id', 'item_id']).join(items_df.select(['item_id', 'item_age_proxy']), on='item_id')\\
        .filter(pl.col('item_age_proxy') >= 0)\\
        .group_by('customer_id').agg(pl.col('item_age_proxy').mean().alias('u_avg_age_proxy'))

    u_prof = history_df.group_by('customer_id').agg([
        pl.col('item_id').n_unique().alias('u_unique_items'),
        pl.col('quantity').sum().alias('u_total_qty'),
        pl.col('price').mean().alias('u_avg_price'),
        pl.col('price').std().alias('u_price_std'),
        (max_ts - pl.col('event_ts').min()).dt.total_days().alias('u_tenure_days'),
        (pl.col('item_id').n_unique() / pl.col('quantity').sum().clip(1)).alias('u_exploration_ratio')
    ]).join(u_brand_hhi, on='customer_id', how='left')\\
      .join(u_cat_hhi, on='customer_id', how='left')\\
      .join(u_avg_age, on='customer_id', how='left')\\
      .with_columns(pl.col('u_avg_age_proxy').fill_null(global_avg_age))
      
    # 2. Item Profiles & repeat propensities
    i_repeats = history_df.select(['customer_id', 'item_id']).group_by(['item_id', 'customer_id']).len().filter(pl.col('len') > 1)\\
        .group_by('item_id').len().rename({'len': 'repeat_buyers'})
    i_prof = history_df.group_by('item_id').agg([
        pl.col('customer_id').n_unique().alias('i_unique_users'),
        pl.col('quantity').sum().alias('i_total_qty'),
        pl.col('location').n_unique().alias('i_hubs_count'),
        pl.col('price').median().alias('i_ref_price')
    ]).join(i_repeats, on='item_id', how='left')\\
      .with_columns((pl.col('repeat_buyers').fill_null(0) / pl.col('i_unique_users')).alias('i_repeat_rate'))\\
      .drop('repeat_buyers')

    # 3. Momentum
    vol_7d = history_df.filter(pl.col('event_ts') >= max_ts - pl.duration(days=7)).group_by('item_id').len().rename({'len': 'v7'})
    vol_21d = history_df.filter(pl.col('event_ts') >= max_ts - pl.duration(days=21)).group_by('item_id').len().rename({'len': 'v21'})
    momentum = vol_7d.join(vol_21d, on='item_id', how='left').with_columns((pl.col('v7') / (pl.col('v21') / 3.0 + 1)).alias('item_momentum'))
    
    # 4. User Categories & Affinities
    u_cat = history_df.select(['customer_id', 'item_id']).join(items_df.select(['item_id', 'category_l1']), on='item_id')\\
        .group_by(['customer_id', 'category_l1']).len()\\
        .with_columns((pl.col('len') / pl.col('len').sum().over('customer_id')).alias('u_cat_affinity'))

    # 5. Locations & Local sales
    u_loc = history_df.group_by('customer_id').agg(pl.col('location').mode().first().alias('location'))
    loc_item_pop = history_df.group_by(['location', 'item_id']).len().rename({'len': 'ui_loc_sales'})

    # 6. GLOBAL PRE-COMPUTATION OF PREFERENCES & HISTORIES
    print("Pre-computing Global User-Item Histories and Preferences...")
    ui_hist = history_df.group_by(['customer_id', 'item_id']).agg([
        pl.col('quantity').sum().alias('ui_total_qty'),
        (max_ts - pl.col('event_ts').max()).dt.total_days().alias('ui_recency_days')
    ])
    
    u_pref_cat = history_df.join(items_df.select(['item_id', 'category_l1']), on='item_id')\\
        .group_by(['customer_id', 'category_l1']).len().sort('len', descending=True)\\
        .group_by('customer_id').head(1).select(['customer_id', 'category_l1']).rename({'category_l1': 'pref_cat_l1'})
        
    u_pref_brand = history_df.join(items_df.select(['item_id', 'category_l1', 'brand']), on='item_id')\\
        .group_by(['customer_id', 'category_l1', 'brand']).len().sort('len', descending=True)\\
        .group_by(['customer_id', 'category_l1']).head(1).select(['customer_id', 'category_l1', 'brand']).rename({'brand': 'pref_brand'})

    # --- DOWNCAST AND OPTIMIZE TYPES FOR 32-BIT TO SAVE 50% RAM AND COMPLETELY ELIMINATE SWAP Bottleneck ---
    print("Downcasting global tables to 32-bit and casting strings to Categoricals...")
    u_prof = u_prof.cast({pl.Float64: pl.Float32, pl.Int64: pl.Int32})
    i_prof = i_prof.cast({pl.Float64: pl.Float32, pl.Int64: pl.Int32})
    ui_hist = ui_hist.cast({pl.Float64: pl.Float32, pl.Int64: pl.Int32})
    momentum = momentum.cast({pl.Float64: pl.Float32, pl.Int64: pl.Int32})
    u_cat = u_cat.cast({pl.Float64: pl.Float32, pl.Int64: pl.Int32})
    loc_item_pop = loc_item_pop.cast({pl.Float64: pl.Float32, pl.Int64: pl.Int32})
    u_loc = u_loc.cast({pl.Int64: pl.Int32})
    
    items_df = items_df.with_columns([
        pl.col('brand').cast(pl.Categorical),
        pl.col('category_l1').cast(pl.Categorical)
    ])
    u_pref_cat = u_pref_cat.with_columns(pl.col('pref_cat_l1').cast(pl.Categorical))
    u_pref_brand = u_pref_brand.with_columns([
        pl.col('pref_brand').cast(pl.Categorical),
        pl.col('category_l1').cast(pl.Categorical)
    ])
    u_loc = u_loc.with_columns(pl.col('location').cast(pl.Categorical))
    loc_item_pop = loc_item_pop.with_columns(pl.col('location').cast(pl.Categorical))

    # Convert global reference tables to LazyFrames to optimize join DAG execution within the loop
    u_prof_lazy = u_prof.lazy()
    i_prof_lazy = i_prof.lazy()
    ui_hist_lazy = ui_hist.lazy()
    momentum_lazy = momentum.lazy()
    u_cat_lazy = u_cat.lazy()
    u_loc_lazy = u_loc.lazy()
    loc_item_pop_lazy = loc_item_pop.lazy()
    u_pref_cat_lazy = u_pref_cat.lazy()
    u_pref_brand_lazy = u_pref_brand.lazy()
    
    # Pre-select and lazyify items metadata to avoid work in loop
    items_feat_lazy = items_df.select(
        ['item_id', 'item_age_proxy', 'brand', 'category_l1'] + [f'{c}_id' for c in cat_cols]
    ).lazy()

    print(f"Opening binary stream for: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    total_users_written = 0
    with output_path.open("wb") as f:
        f.write(b"\\x80\\x04}")  # PROTO 4 + EMPTY_DICT
        
        for idx in range(0, len(target_users), chunk_size):
            chunk_u = target_users[idx:idx+chunk_size]
            print(f"Processing chunk {idx // chunk_size + 1}: users {idx} to {idx + len(chunk_u)}...")
            
            ds = retriever.get_candidates(chunk_u)
            if ds.is_empty():
                continue
            
            # Start Lazy context
            ds_lazy = ds.lazy()
            
            # Chain Lazy Joins (Driver table ds_lazy acts as driver filter for fast hash tables)
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
            
            # Combine all column transformations to prevent intermediate allocations
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
            
            # Trigger execution DAG
            ds_collected = ds_lazy.collect()
            
            # Fast numeric null filling
            num_cols = [c for c in ds_collected.columns if c not in ['customer_id', 'item_id']]
            ds_collected = ds_collected.with_columns(pl.col(num_cols).fill_null(0))
            
            # Prediction
            X_sub = ds_collected.select(all_feats).to_numpy()
            ds_collected = ds_collected.with_columns(pl.Series(name='pred', values=model.predict(X_sub)))
            
            # Optimized local group sort_by
            top10 = ds_collected.group_by('customer_id').agg(
                pl.col('item_id').sort_by('pred', descending=True).head(10)
            )
            
            # Vectorized python iteration (10x faster than iter_rows!)
            c_ids = top10['customer_id'].to_numpy()
            item_lists = top10['item_id'].to_list()
            
            for customer_id, items in zip(c_ids, item_lists):
                key_bytes = pickle.dumps(int(customer_id), protocol=4)
                val_bytes = pickle.dumps(items, protocol=4)
                f.write(key_bytes[2:-1])  # strip PROTO + STOP
                f.write(val_bytes[2:-1])
                f.write(b"s")  # SETITEM
                total_users_written += 1
                
            del ds, ds_lazy, ds_collected, X_sub, top10, c_ids, item_lists
            gc.collect()
            
        f.write(b".")  # STOP
        
    print(f"Successfully streamed {total_users_written} customers to pickle dictionary!")

output_file = Path('submission.pkl')
make_chunked_submission_pkl(h_full, items_df, target_users, lgb_m, all_feats, output_file, chunk_size=20000)"""
            
            cell['source'] = [line + "\n" for line in new_submission_code.split("\n")]
            if cell['source']:
                cell['source'][-1] = cell['source'][-1].rstrip("\n")

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook successfully patched with Lazy API, Restored limits, and Optimized Optuna Search Space!")
