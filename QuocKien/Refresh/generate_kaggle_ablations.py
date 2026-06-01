import nbformat as nbf
import os

SOURCE_SCRIPT = "d:/CS116/ProjectNumberOne/QuocKien/Refresh/local_ablation_test.py"
OUTPUT_DIR = "d:/CS116/ProjectNumberOne/QuocKien/Refresh/ablation_notebooks"

KAGGLE_PATHS = """
# ──────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────
TRANSACTION_PATH = "/kaggle/input/qkindataset2/transaction_full_2025.parquet"
ITEM_PATH = "/kaggle/input/qkindataset2/items.parquet"
EVENT_PATH = "/kaggle/input/qkindataset2/event_full_2025.parquet"
"""

def generate():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(SOURCE_SCRIPT, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Extract everything before if __name__ == "__main__":
    main_idx = content.find("if __name__ == \"__main__\":")
    if main_idx == -1:
        raise ValueError("Could not find main block")
    
    core_code = content[:main_idx]
    
    # Replace the path block
    path_start = core_code.find("# PATHS")
    path_end = core_code.find("SAMPLE_N")
    
    if path_start != -1 and path_end != -1:
        # Move back to start of line for PATHS
        path_start = core_code.rfind("# ──", 0, path_start)
        core_code = core_code[:path_start] + KAGGLE_PATHS + "\n" + core_code[path_end:]
    
    # Define experiments
    experiments = {
        "0_Baseline": {},
        "1_FilterDead": {"filter_dead": True},
        "2_UseEvents": {"use_events": True},
        "3_CoPurchase": {"copurchase": True},
        "5_RichMetadata": {"rich_metadata": True},
        "6_DiscountFeatures": {"discount_features": True},
        "All_Changes": {
            "filter_dead": True,
            "use_events": True,
            "copurchase": True,
            "rich_metadata": True,
            "discount_features": True
        }
    }
    
    for name, flags in experiments.items():
        nb = nbf.v4.new_notebook()
        
        # Cell 1: imports and functions
        cell1 = nbf.v4.new_code_cell(core_code)
        nb.cells.append(cell1)
        
        # Cell 2: runner
        flag_str = ",\n    ".join(f'"{k}": {v}' for k, v in flags.items())
        runner_code = f"""
flags = {{
    {flag_str}
}}
print(f"Running Ablation: {name}")
run_ablation(flags)
"""
        cell2 = nbf.v4.new_code_cell(runner_code.strip())
        nb.cells.append(cell2)
        
        out_path = os.path.join(OUTPUT_DIR, f"PIR_Ablation_{name}.ipynb")
        with open(out_path, "w", encoding="utf-8") as f:
            nbf.write(nb, f)
        print(f"Generated: {out_path}")

if __name__ == "__main__":
    generate()
