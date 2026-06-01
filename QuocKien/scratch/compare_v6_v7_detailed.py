def compare():
    with open('scratch/v6_retriever.txt', 'r', encoding='utf-8') as f:
        v6 = f.read()
    with open('scratch/v7_retriever.txt', 'r', encoding='utf-8') as f:
        v7 = f.read()
        
    print("V6 retriever length:", len(v6))
    print("V7 retriever length:", len(v7))
    
    # Check if 'cat_trend' or 'brand_trend' exists in V6
    print("\nKeywords in V6:")
    for kw in ['cat_trend', 'brand_trend', 'trend', 'local', 'global', 'svd', 'i2i', 'rep', 'replenish']:
        print(f"  {kw}: {kw in v6.lower()}")
        
    print("\nKeywords in V7:")
    for kw in ['cat_trend', 'brand_trend', 'trend', 'local', 'global', 'svd', 'i2i', 'rep', 'replenish']:
        print(f"  {kw}: {kw in v7.lower()}")

if __name__ == '__main__':
    compare()
