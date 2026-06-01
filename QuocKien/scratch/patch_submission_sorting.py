import json

nb_path = r'd:\CS116\ProjectNumberOne\QuocKien\pir_pipeline_v12_submission.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell_code = "".join(nb['cells'][2]['source'])

target_return = """    num_cols = [c for c in ds.columns if c not in ['customer_id', 'item_id', 'category_l1']]
    ds = ds.with_columns([
        pl.col(num_cols).fill_null(0)
    ]).drop('category_l1')
    
    return ds"""

replacement_return = """    num_cols = [c for c in ds.columns if c not in ['customer_id', 'item_id', 'category_l1']]
    ds = ds.with_columns([
        pl.col(num_cols).fill_null(0)
    ]).drop('category_l1')
    
    return ds.sort('customer_id')"""

if target_return in cell_code:
    cell_code = cell_code.replace(target_return, replacement_return)
    print("Notebook Cell 2/3 return statement successfully updated to sort by customer_id!")
else:
    # Try with backslashes/newlines
    target_return_alt = "    return ds"
    replacement_return_alt = "    return ds.sort('customer_id')"
    cell_code = cell_code.replace(target_return_alt, replacement_return_alt)
    print("Notebook Cell 2/3 return statement successfully updated via fallback!")

nb['cells'][2]['source'] = [line + "\n" for line in cell_code.split("\n")]
if nb['cells'][2]['source']:
    nb['cells'][2]['source'][-1] = nb['cells'][2]['source'][-1].rstrip("\n")

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
