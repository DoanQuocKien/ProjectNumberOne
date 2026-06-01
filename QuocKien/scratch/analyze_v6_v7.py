import sys

def analyze():
    sys.stdout.reconfigure(encoding='utf-8')
    with open('scratch/v6_v7_compare_raw.txt', 'r', encoding='utf-8') as f:
        text = f.read()
        
    parts = text.split('===================')
    
    v6_code = ""
    v7_code = ""
    for part in parts:
        if 'V6 CLASS RETRIEVER' in part:
            v6_code = part
        elif 'V7 CLASS RETRIEVER' in part:
            v7_code = part
            
    print("=== TECHNICAL DETAILS OF V6 RETRIEVER ===")
    print(v6_code[:2000])
    print("\n" + "="*80 + "\n")
    print("=== TECHNICAL DETAILS OF V7 RETRIEVER ===")
    print(v7_code[:2000])

if __name__ == '__main__':
    analyze()
