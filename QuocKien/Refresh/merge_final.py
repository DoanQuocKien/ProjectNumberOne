import os

def run():
    with open('local_ablation_test.py', 'r', encoding='utf-8') as f:
        local_code = f.read()
    with open('build_notebooks.py', 'r', encoding='utf-8') as f:
        build_code = f.read()

    # 1. Extract CELL_HELPERS from local_ablation_test.py
    # From "def standardize_age(text):" up to "def run_ablation"
    start_idx = local_code.find("def standardize_age(text):")
    end_idx = local_code.find("def run_ablation", start_idx)
    helpers_code = local_code[start_idx:end_idx]
    
    # We need to prepend the flags block
    flags_block = """
# HARDCODED FLAGS FOR FINAL JAN SUBMISSION
flags = {
    "filter_dead": True,
    "use_events": True,
    "copurchase": True,
    "rich_metadata": True,
    "discount_features": True
}
"""
    helpers_code = flags_block + "\n" + helpers_code

    # 2. Extract build_lgb_dataset_streaming and infer_chunk from build_notebooks.py
    # They are part of CELL_HELPERS in build_notebooks.py
    b_start = build_code.find("def build_lgb_dataset_streaming")
    b_end = build_code.find('"""', b_start)
    streaming_code = build_code[b_start:b_end]

    helpers_code += "\n" + streaming_code

    # Replace double triple-quotes inside helpers_code to avoid breaking the raw string
    helpers_code = helpers_code.replace('"""', "'''")

    # 3. Create the new CELL_HELPERS string literal for build_notebooks_updated.py
    # We will replace the old CELL_HELPERS block in build_notebooks.py
    old_helpers_start = build_code.find('CELL_HELPERS = r"""')
    old_helpers_end = build_code.find('"""', old_helpers_start + 20) + 3

    new_build_code = build_code[:old_helpers_start] + 'CELL_HELPERS = r"""\n' + helpers_code + '"""' + build_code[old_helpers_end:]

    # 4. Update EVENT_PATH in CELL_IMPORTS
    imports_start = new_build_code.find('CELL_IMPORTS = """')
    imports_end = new_build_code.find('"""', imports_start + 20)
    imports_code = new_build_code[imports_start:imports_end]
    if "EVENT_PATH" not in imports_code:
        imports_code = imports_code.replace('ITEM_PATH        = "/kaggle/input/datasets/kinonquc/qkindataset2/items.parquet"', 
                                            'ITEM_PATH        = "/kaggle/input/datasets/kinonquc/qkindataset2/items.parquet"\nEVENT_PATH       = "/kaggle/input/datasets/kinonquc/qkindataset2/event_full_2025.parquet"')
    new_build_code = imports_code + new_build_code[imports_end:]

    # 5. Update BEST_PARAMS
    old_params_start = new_build_code.find("BEST_PARAMS =")
    old_params_end = new_build_code.find('"""', old_params_start + 20) + 3
    new_params = '''BEST_PARAMS = """
best_params = {
    'objective': 'lambdarank', 'metric': 'ndcg', 'eval_at': 10,
    'learning_rate': 0.05234522483024746, 'num_leaves': 155, 'min_data_in_leaf': 156,
    'feature_fraction': 0.48585196246455786, 'bagging_fraction': 0.6681482272360668, 'bagging_freq': 5,
    'lambda_l1': 2.88737161982848, 'lambda_l2': 0.1863820285933072,
    'device_type': 'gpu',
    'random_state': 42, 'n_jobs': -1, 'verbose': -1, 'feature_pre_filter': False
}
"""'''
    new_build_code = new_build_code[:old_params_start] + new_params + new_build_code[old_params_end:]
    
    # 6. Update Phase B and Inference to use the new variables/flags
    # In Phase B and Inference, they call precompute_lookup_tables. We need to pass flags, events, hist_tx_full.
    # We will replace `precompute_lookup_tables(hist_dec, items, all_dec_users)`
    # with `precompute_lookup_tables(hist_dec, items, all_dec_users, flags, hist_tx_full=scan_tx_full(cutoff_end=datetime(2025, 12, 1)).collect(), events=scan_events(cutoff_end=datetime(2025, 12, 1)).collect())`
    
    # Phase A
    new_build_code = new_build_code.replace(
        'precompute_lookup_tables(hist_nov, items, all_nov_users)',
        'precompute_lookup_tables(hist_nov, items, all_nov_users, flags, hist_tx_full=scan_tx_full(cutoff_end=datetime(2025, 11, 1)).collect(), events=scan_events(cutoff_end=datetime(2025, 11, 1)).collect())'
    )
    
    # Phase B
    new_build_code = new_build_code.replace(
        'precompute_lookup_tables(hist_dec, items, all_dec_users)',
        'precompute_lookup_tables(hist_dec, items, all_dec_users, flags, hist_tx_full=scan_tx_full(cutoff_end=datetime(2025, 12, 1)).collect(), events=scan_events(cutoff_end=datetime(2025, 12, 1)).collect())'
    )
    
    # Inference
    new_build_code = new_build_code.replace(
        'precompute_lookup_tables(hist_inf, items, all_users)',
        'precompute_lookup_tables(hist_inf, items, all_users, flags, hist_tx_full=scan_tx_full().collect(), events=scan_events().collect())'
    )

    with open('build_notebooks.py', 'w', encoding='utf-8') as f:
        f.write(new_build_code)
    
    print("build_notebooks.py updated successfully!")

if __name__ == "__main__":
    run()
