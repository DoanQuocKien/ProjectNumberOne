import numpy as np
import time

def test_partition():
    # 4000 users, 20000 items
    scores = np.random.rand(4000, 20000).astype(np.float32)
    K = 60
    
    # Method 1: argsort
    t0 = time.time()
    res_sort = np.argsort(-scores, axis=1)[:, :K]
    t1 = time.time()
    sort_time = t1 - t0
    print(f"np.argsort time: {sort_time:.4f} seconds")
    
    # Method 2: argpartition + local argsort
    t2 = time.time()
    part_idx = np.argpartition(-scores, K, axis=1)[:, :K]
    part_scores = np.take_along_axis(-scores, part_idx, axis=1)
    local_sort = np.argsort(part_scores, axis=1)
    res_part = np.take_along_axis(part_idx, local_sort, axis=1)
    t3 = time.time()
    part_time = t3 - t2
    print(f"np.argpartition time: {part_time:.4f} seconds")
    
    # Verification
    assert np.all(res_sort == res_part), "Results do not match!"
    print("SUCCESS: Results are identical!")
    print(f"Speedup factor: {sort_time / part_time:.2f}x")

if __name__ == '__main__':
    test_partition()
