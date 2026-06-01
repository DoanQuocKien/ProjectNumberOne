import sys

def print_full():
    sys.stdout.reconfigure(encoding='utf-8')
    with open('scratch/v13_criticisms.txt', 'r', encoding='utf-8') as f:
        text = f.read()
        
    parts = text.split('============================================================')
    for part in parts:
        if 'Important clarification:' in part or 'v12 does not leak' in part or 'does **not** leak' in part:
            print("=== CLARIFICATION FOUND ===")
            print(part.strip())

if __name__ == '__main__':
    print_full()
