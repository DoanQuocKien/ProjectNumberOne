import sys

def read():
    sys.stdout.reconfigure(encoding='utf-8')
    with open('scratch/v6_v7_candidates_comparison.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for i in range(min(50, len(lines))):
        print(lines[i].strip())

if __name__ == '__main__':
    read()
