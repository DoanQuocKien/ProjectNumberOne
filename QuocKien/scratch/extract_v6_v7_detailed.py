import json

def extract_detailed():
    with open('pir_pipeline_v6.ipynb', 'r', encoding='utf-8') as f:
        v6 = json.load(f)
    with open('pir_pipeline_v7.ipynb', 'r', encoding='utf-8') as f:
        v7 = json.load(f)
        
    out = []
    
    out.append("=================== V6 CLASS RETRIEVER / GET_CANDIDATES ===================")
    for idx, cell in enumerate(v6['cells']):
        if cell['cell_type'] == 'code':
            src = "".join(cell['source'])
            if 'def get_candidates' in src or 'class Retriever' in src or 'class V6Retriever' in src or 'class V7Retriever' in src:
                out.append(f"--- Cell {idx} ---")
                out.append(src)
                out.append("\n" + "="*80 + "\n")
                
    out.append("=================== V7 CLASS RETRIEVER / GET_CANDIDATES ===================")
    for idx, cell in enumerate(v7['cells']):
        if cell['cell_type'] == 'code':
            src = "".join(cell['source'])
            if 'def get_candidates' in src or 'class Retriever' in src or 'class V6Retriever' in src or 'class V7Retriever' in src:
                out.append(f"--- Cell {idx} ---")
                out.append(src)
                out.append("\n" + "="*80 + "\n")
                
    with open('scratch/v6_v7_compare_raw.txt', 'w', encoding='utf-8') as f:
        f.write("\n".join(out))
    print("Extracted detailed comparison to scratch/v6_v7_compare_raw.txt")

if __name__ == '__main__':
    extract_detailed()
