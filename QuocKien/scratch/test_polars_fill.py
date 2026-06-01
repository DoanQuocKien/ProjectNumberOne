import polars as pl
import sys

df = pl.DataFrame({
    "a": [1, None, 3],
    "b": ["x", None, "z"]
})

with open("output.txt", "w", encoding="utf-8") as f:
    try:
        f.write(str(df.fill_null(0)))
    except Exception as e:
        f.write("Error: " + str(e))
