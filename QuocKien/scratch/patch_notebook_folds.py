import json
from pathlib import Path

def patch_notebook(path):
    p = Path(path)
    if not p.exists():
        print(f"File {path} not found.")
        return
    
    with open(p, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    patched = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = cell['source']
            new_source = []
            for line in source:
                if 'f1 = get_fold(8, 9)' in line:
                    line = line.replace('f1 = get_fold(8, 9)', 'f1 = get_fold(9, 10)')
                    patched = True
                if 'f2 = get_fold(9, 10)' in line:
                    line = line.replace('f2 = get_fold(9, 10)', 'f2 = get_fold(10, 11)')
                    patched = True
                if 'f3 = get_fold(10, 11)' in line:
                    line = line.replace('f3 = get_fold(10, 11)', 'f3 = get_fold(11, 12)')
                    patched = True
                new_source.append(line)
            cell['source'] = new_source
            
    if patched:
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"Successfully patched {path}!")
    else:
        print(f"No patch needed for {path}.")

patch_notebook('pir_pipeline_v12_submission.ipynb')
patch_notebook('pir_pipeline_v12.ipynb')
