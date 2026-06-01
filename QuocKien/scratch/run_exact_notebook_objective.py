import polars as pl
import numpy as np
import warnings
import gc
import re
import json

pl.enable_string_cache()
warnings.filterwarnings('ignore')

# Read notebook cells and extract the code
nb_path = r'd:\CS116\ProjectNumberOne\QuocKien\pir_pipeline_v12_submission.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Extract code from cells
code_blocks = []
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        # Replace Kaggle paths with local forward-slash paths
        source = source.replace('/kaggle/input/datasets/kinonquc/qkindataset2/transaction_full_2025.parquet', 'd:/CS116/ProjectNumberOne/transaction_full_2025.parquet')
        source = source.replace('/kaggle/input/datasets/kinonquc/qkindataset2/items.parquet', 'd:/CS116/ProjectNumberOne/items.parquet')
        # Remove import optuna
        source = source.replace('import optuna', '# import optuna')
        # Exclude final submission or optuna loop
        if 'optuna.create_study' in source or 'make_chunked_submission_pkl' in source:
            continue
        code_blocks.append(source)

# Merge and execute the pipeline setup code in this script's scope
full_code = "\n".join(code_blocks)

# We will run this and print details of f3
print("Executing notebook code blocks...")
local_scope = {}
exec(full_code, globals(), local_scope)

# Access the generated variables in the local scope
f3 = local_scope.get('f3')
if f3 is not None:
    print("\n--- f3 target stats ---")
    lbls = f3['target'].to_list()
    u, counts = np.unique(lbls, return_counts=True)
    print(dict(zip(u.tolist(), counts.tolist())))
    
    g3 = local_scope.get('g3')
    print(f"g3 length: {len(g3)}")
    print(f"y3 length: {len(f3)}")
    print(f"g3 sum: {g3.sum()}")
    
    # Check if they are contiguous
    cust_ids = f3['customer_id'].to_numpy()
    idx = 0
    homogeneous_count = 0
    scrambled = False
    for sz in g3:
        g_cust = cust_ids[idx : idx + sz]
        if not np.all(g_cust == g_cust[0]):
            scrambled = True
            break
        idx += sz
    print(f"Is f3 contiguous by customer_id? {'No' if scrambled else 'Yes'}")
else:
    print("f3 is None!")
