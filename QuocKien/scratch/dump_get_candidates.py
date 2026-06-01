import json
import sys

def dump():
    sys.stdout.reconfigure(encoding='utf-8')
    with open('pir_pipeline_v6.ipynb', 'r', encoding='utf-8') as f:
        v6 = json.load(f)
    with open('pir_pipeline_v7.ipynb', 'r', encoding='utf-8') as f:
        v7 = json.load(f)
        
    print("=== SEARCHING V6 CELLS FOR 'get_candidates' ===")
    for idx, cell in enumerate(v6['cells']):
        if cell['cell_type'] == 'code':
            src = "".join(cell['source'])
            if 'get_candidates' in src:
                print(f"--- V6 Cell {idx} ---")
                print(src[:2000])
                print("="*60)
                
    print("\n=== SEARCHING V7 CELLS FOR 'get_candidates' ===")
    for idx, cell in enumerate(v7['cells']):
        if cell['cell_type'] == 'code':
            src = "".join(cell['source'])
            if 'get_candidates' in src:
                print(f"--- V7 Cell {idx} ---")
                print(src[:2000])
                print("="*60)

if __name__ == '__main__':
    dump()
