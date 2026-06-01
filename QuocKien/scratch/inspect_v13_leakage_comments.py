import json
import sys

def inspect_criticism():
    sys.stdout.reconfigure(encoding='utf-8')
    with open('pir_pipeline_v13_direct_upgrade.ipynb', 'r', encoding='utf-8') as f:
        v13 = json.load(f)
        
    out = []
    for idx, cell in enumerate(v13['cells']):
        src = "".join(cell['source'])
        if cell['cell_type'] == 'markdown':
            if any(keyword in src.lower() for keyword in ['leak', 'look-ahead', 'lookahead', 'v12', 'critic', 'violation', 'error']):
                out.append(f"--- Cell {idx} ---")
                out.append(src)
                out.append("="*60 + "\n")
                
    with open('scratch/v13_criticisms.txt', 'w', encoding='utf-8') as f:
        f.write("\n".join(out))
    print("Criticism text written to scratch/v13_criticisms.txt")

if __name__ == '__main__':
    inspect_criticism()
