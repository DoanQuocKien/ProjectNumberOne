import argparse
import pickle
import shutil
from pathlib import Path


def convert(input_path: Path, output_path: Path, backup: bool = True):
    if backup:
        bak = input_path.with_suffix(input_path.suffix + '.bak')
        shutil.copy2(input_path, bak)
        print(f"Backed up {input_path} -> {bak}")

    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        raise ValueError(f'Loaded object is not a dict, got {type(data)}')

    new = {}
    skipped = 0
    for k, v in data.items():
        try:
            new_k = int(k)
        except Exception:
            skipped += 1
            continue
        new[new_k] = v

    with open(output_path, 'wb') as f:
        pickle.dump(new, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f'Wrote {len(new):,} keys to {output_path} (skipped {skipped} non-intable keys)')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', '-i', default='QuocKien/submission.pkl')
    p.add_argument('--output', '-o', default='QuocKien/submission_int_keys_fixed.pkl')
    p.add_argument('--no-backup', dest='backup', action='store_false')
    args = p.parse_args()
    inp = Path(args.input)
    out = Path(args.output)
    if not inp.exists():
        print(f'Input not found: {inp}')
        raise SystemExit(1)
    convert(inp, out, backup=args.backup)
