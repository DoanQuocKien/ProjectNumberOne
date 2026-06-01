import json
import re

def parse_notebook(path):
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    code = ""
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            code += "".join(cell['source']) + "\n\n"
    return code

def analyze():
    v6 = parse_notebook('pir_pipeline_v6.ipynb')
    v7 = parse_notebook('pir_pipeline_v7.ipynb')
    
    report = []
    report.append("======================================================================")
    report.append("              PIR PIPELINE V6 vs V7 TECHNICAL COMPARISON")
    report.append("======================================================================")
    
    # 1. Candidate Source Analysis
    report.append("\n[1] CANDIDATE EXTRACTION PARAMETERS:")
    
    # Let's search for top K in SVD/I2I/Popular inside V6
    v6_params = {}
    v6_params['svd_comp'] = re.findall(r'n_components\s*=\s*(\d+)', v6)
    v6_params['argsorted'] = re.findall(r'argsorted\s*=\s*np\.argsort\(-s_s,\s*axis=1\)\s*\[:,\s*:\s*(\d+)\]', v6)
    if not v6_params['argsorted']:
        v6_params['argsorted'] = re.findall(r'np\.argsort\(-s_s,\s*axis=1\)\s*\[:,\s*:\s*(\d+)\]', v6)
    v6_params['argsorted_i2i'] = re.findall(r'np\.argsort\(-s_i,\s*axis=1\)\s*\[:,\s*:\s*(\d+)\]', v6)
    v6_params['global_pop'] = re.findall(r'head\((\d+)\)', v6)
    
    v7_params = {}
    v7_params['svd_comp'] = re.findall(r'n_components\s*=\s*(\d+)', v7)
    v7_params['argsorted'] = re.findall(r'np\.argsort\(-scores_svd,\s*axis=1\)\s*\[:,\s*:\s*(\d+)\]', v7)
    if not v7_params['argsorted']:
        v7_params['argsorted'] = re.findall(r'argsort\(-s_s,\s*axis=1\)\s*\[:,\s*:\s*(\d+)\]', v7)
    v7_params['argsorted_i2i'] = re.findall(r'np\.argsort\(-scores_i2i,\s*axis=1\)\s*\[:,\s*:\s*(\d+)\]', v7)
    v7_params['global_pop'] = re.findall(r'head\((\d+)\)', v7)
    
    report.append(f"V6 SVD Components: {v6_params['svd_comp']}")
    report.append(f"V6 SVD Candidates per User (argsorted): {v6_params['argsorted']}")
    report.append(f"V6 I2I Candidates per User (argsorted): {v6_params['argsorted_i2i']}")
    
    report.append(f"V7 SVD Components: {v7_params['svd_comp']}")
    report.append(f"V7 SVD Candidates per User (argsorted): {v7_params['argsorted']}")
    report.append(f"V7 I2I Candidates per User (argsorted): {v7_params['argsorted_i2i']}")
    
    # Let's extract candidate sources blocks
    report.append("\n[2] V6 CANDIDATE EXTRACTION CODE SNIPPETS:")
    for m in re.finditer(r'def get_candidates.*', v6):
        report.append(v6[m.start():m.start()+2500])
        report.append("\n" + "-"*40 + "\n")
        break
        
    report.append("\n[3] V7 CANDIDATE EXTRACTION CODE SNIPPETS:")
    for m in re.finditer(r'class Retriever.*', v7):
        report.append(v7[m.start():m.start()+2500])
        report.append("\n" + "-"*40 + "\n")
        break
        
    with open('scratch/v6_v7_candidates_comparison.txt', 'w', encoding='utf-8') as f:
        f.write("\n".join(report))
    print("Report written to scratch/v6_v7_candidates_comparison.txt")

if __name__ == '__main__':
    analyze()
