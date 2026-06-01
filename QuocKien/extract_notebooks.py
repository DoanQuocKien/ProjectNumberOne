import json
import glob

files = glob.glob('d:/CS116/ProjectNumberOne/QuocKien/Refresh/*.ipynb')
with open('d:/CS116/ProjectNumberOne/QuocKien/notebook_summaries.txt', 'w', encoding='utf-8') as out:
    for f in files:
        if any(x in f for x in ['01A', '02A', '03A', '01C', '02C', '03C', '01_', '02_']):
            out.write(f'--- {f} ---\n')
            try:
                with open(f, 'r', encoding='utf-8') as nf:
                    data = json.load(nf)
                    for cell in data.get('cells', []):
                        if cell['cell_type'] == 'markdown':
                            out.write(''.join(cell['source']) + '\n\n')
                        elif cell['cell_type'] == 'code':
                            source = cell['source']
                            if source:
                                out.write('CODE START: ' + source[0].strip() + '\n')
            except Exception as e:
                out.write(f'Error reading {f}: {e}\n')
            out.write('\n\n')
