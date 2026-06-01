import sys

def read():
    sys.stdout.reconfigure(encoding='utf-8')
    with open('scratch/v13_criticisms.txt', 'r', encoding='utf-8') as f:
        print(f.read())

if __name__ == '__main__':
    read()
