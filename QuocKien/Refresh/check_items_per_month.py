import polars as pl

def analyze_items_per_month(parquet_path: str):
    print(f"Loading transactions from {parquet_path}...")
    
    # Load transactions and extract the month
    df = (
        pl.scan_parquet(parquet_path)
        .select([
            pl.col("customer_id"),
            pl.col("item_id"),
            pl.col("quantity"),
            pl.col("updated_date").cast(pl.Datetime).dt.month().alias("month")
        ])
        .collect()
    )
    
    # Group by customer and month to get total quantity (or unique items) per user per month
    user_monthly = (
        df.group_by(["customer_id", "month"])
        .agg([
            pl.col("quantity").sum().alias("total_items_bought"),
            pl.col("item_id").n_unique().alias("unique_items_bought")
        ])
    )
    
    # Calculate Mean and Median across all users for each month
    monthly_stats = (
        user_monthly.group_by("month")
        .agg([
            pl.col("total_items_bought").mean().alias("mean_total_items"),
            pl.col("total_items_bought").median().alias("median_total_items"),
            pl.col("unique_items_bought").mean().alias("mean_unique_items"),
            pl.col("unique_items_bought").median().alias("median_unique_items"),
            pl.len().alias("active_users")
        ])
        .sort("month")
    )
    
    print("\n=== Items Bought Per User Per Month ===")
    with pl.Config(tbl_formatting="ASCII_MARKDOWN"):
        print(monthly_stats)

if __name__ == "__main__":
    analyze_items_per_month("d:/CS116/ProjectNumberOne/transaction_full_2025.parquet")
