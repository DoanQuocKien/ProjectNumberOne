import json

def extract():
    with open('pir_pipeline_v6.ipynb', 'r', encoding='utf-8') as f:
        v6 = json.load(f)
    with open('pir_pipeline_v7.ipynb', 'r', encoding='utf-8') as f:
        v7 = json.load(f)
        
    print("=== V6 GET CANDIDATES ===")
    for cell in v6['cells']:
        if cell['cell_type'] == 'code':
            src = "".join(cell['source'])
            if 'def get_candidates' in src or 'class V' in src or 'class Retriever' in src or 'Retriever' in src:
                if 'class' in src or 'def get_candidates' in src:
                    print(src[:2000])
                    print("="*40)
                    
    print("\n=== V7 GET CANDIDATES ===")
    for cell in v7['cells']:
        if cell['cell_type'] == 'code':
            src = "".join(cell['source'])
            if 'def get_candidates' in src or 'class V' in src or 'class Retriever' in src or 'Retriever' in src:
                if 'class' in src or 'def get_candidates' in src:
                    print(src[:2000])
                    print("="*40)

if __name__ == '__main__':
    extract()
