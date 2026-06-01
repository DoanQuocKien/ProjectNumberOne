import json

nb_path = r'd:\CS116\ProjectNumberOne\QuocKien\pir_pipeline_v12_submission.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        print(f"Cell {idx}: first line is: {source.splitlines()[0] if source.splitlines() else 'EMPTY'}")
        if "def objective" in source:
            print(f"  -> Found 'def objective' in Cell {idx}!")
