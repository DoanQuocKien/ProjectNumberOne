import polars as pl

s = pl.Series("a", [1, 2, 3, 4, 5])
try:
    print(s.shuffle(seed=42))
except Exception as e:
    print("Error:", e)
