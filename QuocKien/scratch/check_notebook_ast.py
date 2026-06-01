import json
import ast

nb_path = r'd:\CS116\ProjectNumberOne\QuocKien\pir_pipeline_v12_submission.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

has_error = False
for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        try:
            ast.parse(source)
        except SyntaxError as e:
            print(f"CELL INDEX {idx} HAS SYNTAX ERROR: {e}")
            has_error = True

if not has_error:
    print("ALL CELLS PASSED PYTHON AST PARSER VERIFICATION!")
else:
    print("SYNTAX ERRORS DETECTED!")
