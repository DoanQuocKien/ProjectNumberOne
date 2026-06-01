import json

def dump_full():
    with open('pir_pipeline_v6.ipynb', 'r', encoding='utf-8') as f:
        v6 = json.load(f)
    with open('pir_pipeline_v7.ipynb', 'r', encoding='utf-8') as f:
        v7 = json.load(f)
        
    v6_ret = ""
    for cell in v6['cells']:
        if cell['cell_type'] == 'code':
            src = "".join(cell['source'])
            if 'class Retriever' in src or 'def get_candidates' in src:
                v6_ret += src + "\n\n"
                
    v7_ret = ""
    for cell in v7['cells']:
        if cell['cell_type'] == 'code':
            src = "".join(cell['source'])
            if 'class Retriever' in src or 'def get_candidates' in src:
                v7_ret += src + "\n\n"
                
    with open('scratch/v6_retriever.txt', 'w', encoding='utf-8') as f:
        f.write(v6_ret)
    with open('scratch/v7_retriever.txt', 'w', encoding='utf-8') as f:
        f.write(v7_ret)
    print("Dumped full retriever classes to scratch/v6_retriever.txt and scratch/v7_retriever.txt")

if __name__ == '__main__':
    dump_full()
