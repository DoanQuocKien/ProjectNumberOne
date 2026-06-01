import json

nb_path = r'd:\CS116\ProjectNumberOne\QuocKien\pir_pipeline_v12_submission.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Update Cell 2 (V12Retriever with empty candidate check)
cell_2_source = "".join(nb['cells'][2]['source'])

target_concat = """        # Combine all candidate sources
        all_c = pl.concat([df for df in cands.values() if df is not None and df.height > 0]).unique()"""

replacement_concat = """        # Combine all candidate sources safely to prevent ValueErrors
        cands_list = [df for df in cands.values() if df is not None and df.height > 0]
        if not cands_list:
            return pl.DataFrame(schema={'customer_id': pl.Int64, 'item_id': pl.Int32})
        all_c = pl.concat(cands_list).unique()"""

cell_2_source = cell_2_source.replace(target_concat, replacement_concat)
nb['cells'][2]['source'] = [line + "\n" for line in cell_2_source.split("\n")]
if nb['cells'][2]['source']:
    nb['cells'][2]['source'][-1] = nb['cells'][2]['source'][-1].rstrip("\n")


# 2. Update Cell 3 (create_dataset_v12 with empty ds check)
cell_3_source = "".join(nb['cells'][3]['source'])

target_ds_check = """    print(f"Retrieving candidates for {len(valid_users)} users...")
    ds = retriever.get_candidates(valid_users)
    
    positives = target_df.filter(pl.col('customer_id').is_in(valid_users)).select(['customer_id', 'item_id']).unique().with_columns(pl.lit(1).alias('target'))"""

replacement_ds_check = """    print(f"Retrieving candidates for {len(valid_users)} users...")
    ds = retriever.get_candidates(valid_users)
    
    if ds.is_empty():
        # Safeguard schema if candidates are completely empty
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
        
    positives = target_df.filter(pl.col('customer_id').is_in(valid_users)).select(['customer_id', 'item_id']).unique().with_columns(pl.lit(1).alias('target'))"""

cell_3_source = cell_3_source.replace(target_ds_check, replacement_ds_check)
nb['cells'][3]['source'] = [line + "\n" for line in cell_3_source.split("\n")]
if nb['cells'][3]['source']:
    nb['cells'][3]['source'][-1] = nb['cells'][3]['source'][-1].rstrip("\n")

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook updated successfully with empty list concat and candidate schemas safeguards!")
