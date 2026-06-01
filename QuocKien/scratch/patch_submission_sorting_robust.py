import json
import re

def robust_patch_notebook(path):
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    patched = False
    for idx, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source_lines = cell['source']
            code = "".join(source_lines)
            
            # Check if this cell defines create_dataset_v12
            if 'def create_dataset_v12' in code:
                print(f"Found create_dataset_v12 in {path} at Cell {idx}!")
                
                # We need to find the final 'return ds' of this function
                # Since the function is indented, the return statement is indented
                # Let's search from the bottom for the first occurrence of return ds
                for line_idx in range(len(source_lines) - 1, -1, -1):
                    line = source_lines[line_idx]
                    if re.search(r'^\s+return\s+ds\b', line):
                        # Replace 'return ds' with 'return ds.sort(\'customer_id\')'
                        old_line = line
                        new_line = re.sub(r'\breturn\s+ds\b', "return ds.sort('customer_id')", line)
                        source_lines[line_idx] = new_line
                        patched = True
                        print(f"Successfully replaced line {line_idx} in Cell {idx}:")
                        print(f"  Old: {old_line.strip()}")
                        print(f"  New: {new_line.strip()}")
                        break
                if patched:
                    break
    
    if patched:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"Saved patched notebook: {path}\n")
    else:
        print(f"ERROR: Could not find or patch create_dataset_v12 in {path}!\n")

robust_patch_notebook(r'd:\CS116\ProjectNumberOne\QuocKien\pir_pipeline_v12_submission.ipynb')
robust_patch_notebook(r'd:\CS116\ProjectNumberOne\QuocKien\pir_pipeline_v12.ipynb')
