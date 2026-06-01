import json
import re

def patch_notebook(path):
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    modified = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            code = "".join(cell['source'])
            if 'study.optimize(objective,' in code:
                # Replace n_trials=... with n_trials=1
                new_code = re.sub(r'n_trials\s*=\s*\d+', 'n_trials=1', code)
                if new_code != code:
                    cell['source'] = [line + "\n" for line in new_code.split("\n")]
                    if cell['source']:
                        cell['source'][-1] = cell['source'][-1].rstrip("\n")
                    modified = True
                    print(f"Patched study.optimize in {path} to use n_trials=1!")
    
    if modified:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
    else:
        print(f"Could not find study.optimize in {path} or it was already patched.")

patch_notebook(r'd:\CS116\ProjectNumberOne\QuocKien\pir_pipeline_v12_submission.ipynb')
patch_notebook(r'd:\CS116\ProjectNumberOne\QuocKien\pir_pipeline_v12.ipynb')
