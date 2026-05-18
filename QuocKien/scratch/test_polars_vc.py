import polars as pl

df = pl.DataFrame({"c": ["a", "b", "a", "c", "b", "a"]})
vc = df["c"].value_counts()
with open("vc_cols.txt", "w") as f:
    f.write(", ".join(vc.columns))
