import json
from pathlib import Path

p = Path('pir_pipeline_v12_submission.ipynb')
if not p.exists():
    print("File not found.")
    exit(1)

with open(p, 'r', encoding='utf-8') as f:
    nb = json.load(f)

patched = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        new_source = []
        for line in source:
            # 1. Remove target sort inside create_dataset_v12
            if "ds = ds.sort(['customer_id', 'target'], descending=[False, True])" in line:
                line = line.replace("ds = ds.sort(['customer_id', 'target'], descending=[False, True])", "# ds = ds.sort(['customer_id', 'target'], descending=[False, True])")
                patched = True
            
            # 2. Shield target from num_cols fill_null sweep
            if "num_cols = [c for c in ds.columns if c not in ['customer_id', 'item_id', 'category_l1']]" in line:
                line = line.replace("['customer_id', 'item_id', 'category_l1']", "['customer_id', 'item_id', 'category_l1', 'target']")
                patched = True
            
            # 3. Apply shuffle-then-sort return statement at the end of create_dataset_v12
            if "return ds.sort('customer_id')" in line:
                line = line.replace("return ds.sort('customer_id')", "return ds.sample(fraction=1.0, shuffle=True, seed=SEED).sort('customer_id')")
                patched = True
                
            new_source.append(line)
        cell['source'] = new_source

if patched:
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Successfully patched tiebreaker and target shielding in pir_pipeline_v12_submission.ipynb!")
else:
    print("No patch was applied.")
