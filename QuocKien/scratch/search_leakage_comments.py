import sys

def search():
    sys.stdout.reconfigure(encoding='utf-8')
    with open('scratch/v13_criticisms.txt', 'r', encoding='utf-8') as f:
        text = f.read()
        
    print("=== SEARCHING FOR LEAKAGE COMMENTS IN V13 ===")
    lines = text.split('\n')
    for line in lines:
        if any(keyword in line.lower() for keyword in ['leak', 'look-ahead', 'lookahead', 'violation', 'error', 'wrong', 'flaw']):
            print(line.strip())

if __name__ == '__main__':
    search()
