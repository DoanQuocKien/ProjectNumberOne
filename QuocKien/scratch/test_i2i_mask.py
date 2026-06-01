import numpy as np
import polars as pl
from scipy.sparse import csr_matrix

mtx = csr_matrix([[1, 0, 1], [0, 1, 0]])
i2i_sim = csr_matrix([[0, 0.5, 0.8], [0.5, 0, 0], [0.8, 0, 0]])

idx_chunk = [0, 1]
u_b = np.array([101, 102])
i_arr = np.array(["item_0", "item_1", "item_2"])

scores_i2i = mtx[idx_chunk].dot(i2i_sim).toarray()
t80 = np.argsort(-scores_i2i, axis=1)[:, :2]
mask = np.take_along_axis(scores_i2i, t80, axis=1) > 0

user_ids = np.repeat(u_b, 2)[mask.flatten()]
item_ids = i_arr[t80.flatten()][mask.flatten()]

df = pl.DataFrame({
    "customer_id": user_ids,
    "item_id": item_ids
})

with open("i2i_output.txt", "w", encoding="utf-8") as f:
    f.write(str(df))
