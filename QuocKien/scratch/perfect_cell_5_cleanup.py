import json

nb_path = r'd:\CS116\ProjectNumberOne\QuocKien\pir_pipeline_v12_submission.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell_5_code = "".join(nb['cells'][5]['source'])

func_def = "def make_chunked_submission_pkl(history_df, items_df, target_users, model, all_feats, output_path, chunk_size=20000):"
retriever_init = "retriever = V12Retriever(history_df, items_df)"

idx_def = cell_5_code.find(func_def)
idx_retr = cell_5_code.find(retriever_init)

if idx_def != -1 and idx_retr != -1:
    before = cell_5_code[:idx_def + len(func_def)]
    after = cell_5_code[idx_retr:]
    
    # Inject exact clean mapping loading block
    clean_injection = """\n    # Load global integer-to-string item mapping
    with open("idx2item.pkl", "rb") as f_map:
        idx2item = pickle.load(f_map)\n    """
        
    cell_5_code = before + clean_injection + after
    print("Cleaned up Cell 5 flawlessly using relative block slicing!")
else:
    print(f"Error: indices not found. idx_def: {idx_def}, idx_retr: {idx_retr}")

nb['cells'][5]['source'] = [line + "\n" for line in cell_5_code.split("\n")]
if nb['cells'][5]['source']:
    nb['cells'][5]['source'][-1] = nb['cells'][5]['source'][-1].rstrip("\n")

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Updated Cell 5 Head:")
print("".join(nb['cells'][5]['source'][:20]))
