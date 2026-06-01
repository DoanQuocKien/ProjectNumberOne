import json
import sys

def inspect():
    sys.stdout.reconfigure(encoding='utf-8')
    with open('pir_pipeline_v13_direct_upgrade.ipynb', 'r', encoding='utf-8') as f:
        v13 = json.load(f)
        
    print("=== SEARCHING V13 FOR CRITICISM OR LEAKAGE COMMENTS ===")
    for idx, cell in enumerate(v13['cells']):
        src = "".join(cell['source'])
        if any(keyword in src.lower() for keyword in ['leak', 'look-ahead', 'lookahead', 'look ahead', 'critic', 'v12', 'error', 'violation']):
            print(f"--- Cell {idx} ({cell['cell_type']}) ---")
            print(src[:1000])
            print("="*60)

if __name__ == '__main__':
    inspect()
