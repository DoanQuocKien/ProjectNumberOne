def search():
    with open('scratch/v6_retriever.txt', 'r', encoding='utf-8') as f:
        v6 = f.read()
    with open('scratch/v7_retriever.txt', 'r', encoding='utf-8') as f:
        v7 = f.read()
        
    print("=== V6 CONCATENATION STEP ===")
    lines_v6 = v6.split('\n')
    for line in lines_v6:
        if 'pl.concat' in line or 'unique()' in line or 'all_cands' in line:
            print(line.strip())
            
    print("\n=== V7 CONCATENATION STEP ===")
    lines_v7 = v7.split('\n')
    for line in lines_v7:
        if 'pl.concat' in line or 'unique()' in line or 'all_cands' in line:
            print(line.strip())

if __name__ == '__main__':
    search()
