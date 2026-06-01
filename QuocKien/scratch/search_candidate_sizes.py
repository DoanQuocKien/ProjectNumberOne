import sys

def search():
    sys.stdout.reconfigure(encoding='utf-8')
    with open('scratch/v6_v7_compare_raw.txt', 'r', encoding='utf-8') as f:
        text = f.read()
        
    lines = text.split('\n')
    
    print("=== Searching SVD, I2I, Popular numbers in V6/V7 comparative raw output ===")
    for line in lines:
        l = line.strip()
        if any(keyword in l.lower() for keyword in ['n_components', 'global_top', 'local_heroes', 'scores_svd', 'scores_i2i', 'argsort', 'head', 't60', 't80', 't100']):
            print(l)

if __name__ == '__main__':
    search()
