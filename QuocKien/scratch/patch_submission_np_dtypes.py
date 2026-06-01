import json

nb_path = r'd:\CS116\ProjectNumberOne\QuocKien\pir_pipeline_v12_submission.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell_2_code = "".join(nb['cells'][2]['source'])

# 1. Patch i_arr = np.array(self.idx2i)
target_i_arr = "        i_arr = np.array(self.idx2i)"
replacement_i_arr = "        i_arr = np.array(self.idx2i, dtype=np.int32)"

cell_2_code = cell_2_code.replace(target_i_arr, replacement_i_arr)

# 2. Patch u_b = np.array(t_u[i:i+chunk])
target_u_b = "                u_b = np.array(t_u[i:i+chunk])"
replacement_u_b = "                u_b = np.array(t_u[i:i+chunk], dtype=np.int64)"

cell_2_code = cell_2_code.replace(target_u_b, replacement_u_b)

nb['cells'][2]['source'] = [line + "\n" for line in cell_2_code.split("\n")]
if nb['cells'][2]['source']:
    nb['cells'][2]['source'][-1] = nb['cells'][2]['source'][-1].rstrip("\n")

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook Cell 2 successfully updated with explicit numpy numeric dtypes!")
