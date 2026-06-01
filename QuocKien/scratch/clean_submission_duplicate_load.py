import json

nb_path = r'd:\CS116\ProjectNumberOne\QuocKien\pir_pipeline_v12_submission.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell_5_lines = nb['cells'][5]['source']
cell_5_code = "".join(cell_5_lines)

# Target duplicated head
duplicated_head = """def make_chunked_submission_pkl(history_df, items_df, target_users, model, all_feats, output_path, chunk_size=20000):
    # Load global integer-to-string item mapping
    with open("idx2item.pkl", "rb") as f_map:
        idx2item = pickle.load(f_map)
    # Load global integer-to-string item mapping
    with open("idx2item.pkl", "rb") as f_map:
        idx2item = pickle.load(f_map)
    pl.enable_string_cache()"""

clean_head = """def make_chunked_submission_pkl(history_df, items_df, target_users, model, all_feats, output_path, chunk_size=20000):
    # Load global integer-to-string item mapping
    with open("idx2item.pkl", "rb") as f_map:
        idx2item = pickle.load(f_map)"""

if duplicated_head in cell_5_code:
    cell_5_code = cell_5_code.replace(duplicated_head, clean_head)
    print("Cleaned up duplicated head using strict string match!")
else:
    # Fallback to general lines filtering
    print("Strict match not found, checking manual string replacements...")
    
    # We will split cell_5_code into lines and remove the duplicates manually
    lines = cell_5_code.split("\n")
    cleaned_lines = []
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if "pl.enable_string_cache()" in line and idx < 20: # remove only from top of the function
            idx += 1
            continue
        if "with open(\"idx2item.pkl\"" in line:
            # Check if we already added it
            if any("idx2item" in l for l in cleaned_lines[-3:]):
                # Skip duplicate open and load lines
                idx += 3
                continue
        cleaned_lines.append(line)
        idx += 1
    cell_5_code = "\n".join(cleaned_lines)

nb['cells'][5]['source'] = [line + "\n" for line in cell_5_code.split("\n")]
if nb['cells'][5]['source']:
    nb['cells'][5]['source'][-1] = nb['cells'][5]['source'][-1].rstrip("\n")

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Updated Cell 5 Head:")
print("".join(nb['cells'][5]['source'][:18]))
