import json

nb_path = r'd:\CS116\ProjectNumberOne\QuocKien\pir_pipeline_v12_submission.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell_1_code = "".join(nb['cells'][1]['source'])

target_df_raw = """df_raw = pl.read_parquet(T_PATH).join(item_mapping, on='item_id', how='left').drop('item_id').rename({'item_int_id': 'item_id'}).select([
    pl.col('customer_id').cast(pl.Int64), # Must keep as Int64 to avoid overflow
    pl.col('item_id').cast(pl.Int32),     # Lightweight integer!
    pl.col('quantity').cast(pl.Int32),
    pl.col('price').cast(pl.Float32),
    # Map location strings to physical Int16 immediately
    pl.col('location').cast(pl.Categorical).to_physical().cast(pl.Int16), 
    pl.col('updated_date').cast(pl.Datetime).alias('event_ts')
]).with_columns([
    pl.col('event_ts').dt.month().alias('month').cast(pl.Int8),
    pl.col('event_ts').dt.weekday().alias('dow').cast(pl.Int8)
])"""

replacement_df_raw = """df_raw = pl.read_parquet(T_PATH).join(item_mapping, on='item_id', how='inner').drop('item_id').rename({'item_int_id': 'item_id'}).select([
    pl.col('customer_id').cast(pl.Int64), # Must keep as Int64 to avoid overflow
    pl.col('item_id').cast(pl.Int32),     # Lightweight integer!
    pl.col('quantity').cast(pl.Int32),
    pl.col('price').cast(pl.Float32),
    # Map location strings to physical Int16 immediately
    pl.col('location').cast(pl.Categorical).to_physical().cast(pl.Int16), 
    pl.col('updated_date').cast(pl.Datetime).alias('event_ts')
]).drop_nulls(subset=['item_id', 'customer_id']).with_columns([
    pl.col('event_ts').dt.month().alias('month').cast(pl.Int8),
    pl.col('event_ts').dt.weekday().alias('dow').cast(pl.Int8)
])"""

cell_1_code = cell_1_code.replace(target_df_raw, replacement_df_raw)

nb['cells'][1]['source'] = [line + "\n" for line in cell_1_code.split("\n")]
if nb['cells'][1]['source']:
    nb['cells'][1]['source'][-1] = nb['cells'][1]['source'][-1].rstrip("\n")

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook Cell 1 successfully updated with inner join and drop_nulls safeguard!")
