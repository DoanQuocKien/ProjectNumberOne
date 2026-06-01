import polars as pl
from datetime import datetime

TRANSACTION_PATH = "d:/CS116/ProjectNumberOne/transaction_full_2025.parquet"

print("Loading transactions...")
hist_inf = pl.scan_parquet(TRANSACTION_PATH).filter(pl.col("updated_date") < datetime(2025, 12, 1)).collect()
total_users = hist_inf["customer_id"].n_unique()
print(f"Total unique users: {total_users}")

counts = hist_inf.group_by("customer_id").len()
for thresh in [2, 3, 5, 10]:
    cnt = len(counts.filter(pl.col("len") >= thresh))
    print(f"Users with >= {thresh} transactions: {cnt} ({cnt/total_users*100:.2f}%)")
