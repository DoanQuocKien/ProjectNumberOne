import polars as pl

df = pl.DataFrame({"c": ["a", "b", "a"]})
df = df.with_columns(pl.col("c").cast(pl.Categorical).to_physical().cast(pl.Int32).alias("c_id"))
with open("vc_physical.txt", "w") as f:
    f.write(str(df["c_id"].to_list()))
