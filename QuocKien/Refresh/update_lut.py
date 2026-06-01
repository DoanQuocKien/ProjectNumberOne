import os

with open("local_ablation_test.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Add LUT tables right before print("  [LUT] All lookup tables ready.")
lut_insert = """
    # ── NEW: Basket & Temporal & Replenishment (Change 7) ──
    print("  [LUT] advanced basket & temporal...")
    # basket sizes
    tables["u_basket"] = lazy_ht.group_by(["customer_id", "bill_id"]).agg(pl.col("item_id").n_unique().alias("bill_items")).group_by("customer_id").agg(pl.col("bill_items").mean().cast(pl.Float32).alias("u_avg_items_per_bill")).collect(streaming=True)
    tables["i_basket"] = lazy_ht.group_by(["item_id", "bill_id"]).agg(pl.len().alias("qty")).group_by("item_id").agg(pl.col("qty").mean().cast(pl.Float32).alias("i_avg_items_in_its_bills")).collect(streaming=True)
    
    # temporal
    lazy_time = lazy_ht.with_columns([
        pl.col("event_ts").dt.hour().alias("hour"),
        (pl.col("event_ts").dt.weekday() >= 6).cast(pl.Float32).alias("is_weekend")
    ])
    tables["u_time"] = lazy_time.group_by("customer_id").agg([
        pl.col("is_weekend").mean().cast(pl.Float32).alias("u_weekend_ratio"),
        pl.col("hour").mean().cast(pl.Float32).alias("u_avg_hour")
    ]).collect(streaming=True)
    tables["i_time"] = lazy_time.group_by("item_id").agg([
        pl.col("is_weekend").mean().cast(pl.Float32).alias("i_weekend_ratio"),
        pl.col("hour").mean().cast(pl.Float32).alias("i_avg_hour")
    ]).collect(streaming=True)

    # replenishment median gap
    i_dates = lazy_ht.select(["customer_id", "item_id", "event_ts"]).sort(["customer_id", "item_id", "event_ts"]).collect(streaming=True)
    i_gaps = i_dates.with_columns(
        (pl.col("event_ts") - pl.col("event_ts").shift(1).over(["customer_id", "item_id"])).dt.total_days().alias("gap")
    ).filter(pl.col("gap").is_not_null() & (pl.col("gap") > 1))
    tables["item_median_gap"] = i_gaps.group_by("item_id").agg(pl.col("gap").median().cast(pl.Float32).alias("i_median_replenish_gap"))
"""
content = content.replace('print("  [LUT] All lookup tables ready.")', lut_insert + '\n    print("  [LUT] All lookup tables ready.")')

# 2. In assemble_dataset, add the joins
join_insert = """
    if "u_basket" in tables:
        df = df.join(tables["u_basket"], on="customer_id", how="left")
        df = df.join(tables["i_basket"], on="item_id", how="left")
        df = df.join(tables["u_time"], on="customer_id", how="left")
        df = df.join(tables["i_time"], on="item_id", how="left")
        df = df.join(tables["item_median_gap"], on="item_id", how="left")
"""
content = content.replace('if "u_discount" in tables:', join_insert + '\n    if "u_discount" in tables:')

# 3. Add the cross ratios
cross_ratio_insert = """
    if "u_avg_items_per_bill" in df.columns and "i_avg_items_in_its_bills" in df.columns:
        cross_cols.append((pl.col("u_avg_items_per_bill") - pl.col("i_avg_items_in_its_bills")).abs().alias("basket_size_mismatch"))
    if "u_weekend_ratio" in df.columns and "i_weekend_ratio" in df.columns:
        cross_cols.append((pl.col("u_weekend_ratio") * pl.col("i_weekend_ratio")).alias("weekend_shopper_match"))
    if "ui_days_since_last" in df.columns and "i_median_replenish_gap" in df.columns:
        cross_cols.append((pl.col("ui_days_since_last") - pl.col("i_median_replenish_gap")).alias("replenishment_overdue_days"))
"""
content = content.replace('if flags.get("discount_features") and "u_promo_purchase_ratio" in df.columns and "i_promo_sales_ratio" in df.columns:', cross_ratio_insert + '\n    if flags.get("discount_features") and "u_promo_purchase_ratio" in df.columns and "i_promo_sales_ratio" in df.columns:')

with open("local_ablation_test.py", "w", encoding="utf-8") as f:
    f.write(content)
print("local_ablation_test.py updated successfully!")
