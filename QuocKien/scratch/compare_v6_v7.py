import json

def compare():
    with open('pir_pipeline_v6.ipynb', 'r', encoding='utf-8') as f:
        v6 = json.load(f)
    with open('pir_pipeline_v7.ipynb', 'r', encoding='utf-8') as f:
        v7 = json.load(f)
        
    print("=== V6 CODE CELLS ===")
    for idx, cell in enumerate(v6['cells']):
        if cell['cell_type'] == 'code':
            src = "".join(cell['source'])
            if 'get_candidates' in src or 'V6' in src or 'V7' in src or 'class' in src:
                print(f"Cell {idx} has get_candidates or similar class.")
                # print(src[:500])
                
    print("\n=== V7 CODE CELLS ===")
    for idx, cell in enumerate(v7['cells']):
        if cell['cell_type'] == 'code':
            src = "".join(cell['source'])
            if 'get_candidates' in src or 'class' in src:
                print(f"Cell {idx} has get_candidates or similar class.")
                # print(src[:500])

if __name__ == '__main__':
    compare()
