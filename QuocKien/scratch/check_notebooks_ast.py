import json
import ast

def check_ast(path):
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    for idx, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            code = "".join(cell['source'])
            try:
                ast.parse(code)
            except SyntaxError as e:
                print(f"Syntax Error in {path} Cell {idx}: {e}")
                return False
    print(f"All cells in {path} successfully passed AST parsing!")
    return True

check_ast(r'd:\CS116\ProjectNumberOne\QuocKien\pir_pipeline_v12_submission.ipynb')
check_ast(r'd:\CS116\ProjectNumberOne\QuocKien\pir_pipeline_v12.ipynb')
