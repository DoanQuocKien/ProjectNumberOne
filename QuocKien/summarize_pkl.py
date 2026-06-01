from __future__ import annotations

import argparse
import pickle
from collections import Counter
from pathlib import Path
from typing import Any


def count_unique_items(items: list[Any]) -> int:
    try:
        return len(set(items))
    except TypeError:
        return len({repr(item) for item in items})


def summarize_value(value: Any) -> dict[str, Any]:
    info: dict[str, Any] = {"type": type(value).__name__}
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        info["length"] = len(items)
        info["item_types"] = sorted({type(item).__name__ for item in items})
        info["unique_items"] = count_unique_items(items)
        if items:
            info["sample"] = items[:10]
    elif isinstance(value, dict):
        info["length"] = len(value)
        info["key_types"] = sorted({type(k).__name__ for k in value.keys()})
        info["value_types"] = sorted({type(v).__name__ for v in value.values()})
        info["sample_keys"] = list(value.keys())[:10]
    else:
        info["repr"] = repr(value)
    return info


def summarize_pickle(path: Path) -> None:
    with path.open("rb") as f:
        obj = pickle.load(f)

    print(f"File: {path}")
    print(f"Top-level type: {type(obj).__name__}")

    if isinstance(obj, dict):
        print(f"Top-level size: {len(obj)}")
        key_types = Counter(type(k).__name__ for k in obj.keys())
        value_types = Counter(type(v).__name__ for v in obj.values())
        print(f"Key types: {dict(key_types)}")
        print(f"Value types: {dict(value_types)}")

        if obj:
            sample_key = next(iter(obj))
            sample_value = obj[sample_key]
            print("Sample key:", sample_key)
            print("Sample value summary:", summarize_value(sample_value))

            if isinstance(sample_value, (list, tuple, set)):
                lengths = [len(v) for v in obj.values() if isinstance(v, (list, tuple, set))]
                if lengths:
                    print(
                        "Value lengths: min={0}, max={1}, avg={2:.2f}".format(
                            min(lengths), max(lengths), sum(lengths) / len(lengths)
                        )
                    )
                    print("Length distribution:", dict(Counter(lengths).most_common(10)))

            sample_keys = list(obj.keys())[:5]
            print("Sample entries:")
            for key in sample_keys:
                print(f"  {key!r} -> {summarize_value(obj[key])}")

    elif isinstance(obj, list):
        print(f"List length: {len(obj)}")
        print(f"Item types: {dict(Counter(type(x).__name__ for x in obj))}")
        print("Sample:", obj[:10])
    else:
        print("Object summary:", summarize_value(obj))


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize the structure of a pickle file.")
    parser.add_argument("pkl_path", nargs="?", default="submission_bulletproof.pkl", help="Path to the pickle file")
    args = parser.parse_args()

    path = Path(args.pkl_path)
    if not path.is_absolute():
        script_dir = Path(__file__).resolve().parent
        script_default = script_dir / path
        if script_default.exists():
            path = script_default
    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    summarize_pickle(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
