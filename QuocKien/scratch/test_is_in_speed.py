import time
import polars as pl
import numpy as np

def benchmark():
    print("Generating mock data...")
    # Mock history of 10M rows
    n_rows = 10000000
    df = pl.DataFrame({
        'customer_id': np.random.randint(0, 800000, n_rows, dtype=np.int64),
        'item_id': np.random.randint(0, 20000, n_rows, dtype=np.int32)
    })
    
    # Target list of 100k users
    target_users = list(np.random.choice(np.arange(800000, dtype=np.int64), 100000, replace=False))
    target_df = pl.DataFrame({'customer_id': target_users})
    
    # Method 1: is_in
    print("Benchmarking pl.col().is_in(python_list)...")
    start = time.time()
    res1 = df.filter(pl.col('customer_id').is_in(target_users))
    res1_height = res1.height
    print(f"is_in completed in {time.time() - start:.2f} seconds (rows: {res1_height})")
    
    # Method 2: inner join
    print("Benchmarking inner join...")
    start = time.time()
    res2 = df.join(target_df, on='customer_id', how='inner')
    res2_height = res2.height
    print(f"inner join completed in {time.time() - start:.2f} seconds (rows: {res2_height})")

if __name__ == '__main__':
    benchmark()
