import json

def check_v7():
    with open('pir_pipeline_v7.ipynb', 'r', encoding='utf-8') as f:
        v7 = json.load(f)
        
    code = ""
    for cell in v7['cells']:
        if cell['cell_type'] == 'code':
            code += "".join(cell['source']) + "\n\n"
            
    print("=== SEARCHING V7 FOR FILTERING OR LOCATION INNER JOINS ===")
    lines = code.split('\n')
    for l in lines:
        if any(keyword in l.lower() for keyword in ['filter_candidates', 'item_hubs', 'item_p', 'avg_p', 'inner', 'explode']):
            print(l.strip())

if __name__ == '__main__':
    check_v7()
