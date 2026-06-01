import sys

def read():
    sys.stdout.reconfigure(encoding='utf-8')
    with open('scratch/v6_v7_candidates_comparison.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Let's print the top 1500 characters
    print(text[:2000])

if __name__ == '__main__':
    read()
