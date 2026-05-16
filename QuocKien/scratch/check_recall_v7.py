import polars as pl
import numpy as np
import os
from scipy.sparse import csr_matrix

T_PATH = '../transaction_full_2025.parquet'
I_PATH = '../items.parquet'

def check_v7_recall():
    print("=== Analyzing Candidate Recall (V7 Prototyping) ===")
    df_raw = pl.read_parquet(T_PATH).select([
        pl.col('customer_id').cast(pl.Int64),
        pl.col('item_id').cast(pl.Utf8),
        pl.col('updated_date').cast(pl.Datetime).alias('event_ts'),
        pl.col('location').cast(pl.Utf8),
        pl.col('quantity').cast(pl.Float32)
    ]).with_columns(pl.col('event_ts').dt.month().alias('month'))

    items_df = pl.read_parquet(I_PATH).select(['item_id', 'category_l1', 'brand'])
    items_df = items_df.with_columns(pl.col('item_id').cast(pl.Utf8))
    
    # Train/Val split
    df_train = df_raw.filter(pl.col('month') <= 10)
    df_val_truth = df_raw.filter(pl.col('month') == 11)
    
    target_users = df_val_truth['customer_id'].unique().head(10000).to_list()
    truth = df_val_truth.filter(pl.col('customer_id').is_in(target_users)).select(['customer_id', 'item_id']).unique()
    
    print(f"Target Users: {len(target_users)}, Truth Pairs: {truth.height}")
    max_ts = df_train['event_ts'].max()

    # --- SOURCE 1: Repurchases ---
    df_rep = df_train.filter(pl.col('customer_id').is_in(target_users)).select(['customer_id', 'item_id']).unique()
    
    # --- SOURCE 2: Global Trending (Top 100 30d) ---
    vol_30d = df_train.filter(pl.col('event_ts') >= max_ts - pl.duration(days=30)).group_by('item_id').len()
    top_100_trending = vol_30d.sort('len', descending=True).head(100).select('item_id')
    df_global_trend = pl.DataFrame({'customer_id': target_users}).join(top_100_trending.with_columns(pl.lit(1).alias('_k')), how='cross').drop('_k')

    # --- SOURCE 3: Multi-Category Popularity (Top 3 Cats, Top 20 items each) ---
    user_top_cats = df_train.filter(pl.col('customer_id').is_in(target_users)).join(items_df.select(['item_id', 'category_l1']), on='item_id').group_by(['customer_id', 'category_l1']).len().sort(['customer_id', 'len'], descending=[False, True]).group_by('customer_id').head(3)
    cat_pop = df_train.filter(pl.col('event_ts') >= max_ts - pl.duration(days=60)).join(items_df.select(['item_id', 'category_l1']), on='item_id').group_by(['category_l1', 'item_id']).len().sort(['category_l1', 'len'], descending=[False, True]).group_by('category_l1').head(20)
    df_multi_cat = user_top_cats.join(cat_pop, on='category_l1').select(['customer_id', 'item_id']).unique()

    # --- SOURCE 4: Local Trending (Top 50 per Location) ---
    user_loc = df_train.filter(pl.col('customer_id').is_in(target_users)).group_by('customer_id').agg(pl.col('location').mode().first().alias('location'))
    loc_pop = df_train.filter(pl.col('event_ts') >= max_ts - pl.duration(days=60)).group_by(['location', 'item_id']).len().sort(['location', 'len'], descending=[False, True]).group_by('location').head(50)
    df_local_trend = user_loc.join(loc_pop, on='location').select(['customer_id', 'item_id']).unique()

    # --- SOURCE 5: Brand Loyalty (Top 5 items from Top 3 Brands) ---
    user_brands = df_train.filter(pl.col('customer_id').is_in(target_users)).join(items_df.select(['item_id', 'brand']), on='item_id').filter(pl.col('brand') != 'Không xác định').group_by(['customer_id', 'brand']).len().sort(['customer_id', 'len'], descending=[False, True]).group_by('customer_id').head(3)
    brand_pop = df_train.filter(pl.col('event_ts') >= max_ts - pl.duration(days=90)).join(items_df.select(['item_id', 'brand']), on='item_id').filter(pl.col('brand') != 'Không xác định').group_by(['brand', 'item_id']).len().sort(['brand', 'len'], descending=[False, True]).group_by('brand').head(5)
    df_brand = user_brands.join(brand_pop, on='brand').select(['customer_id', 'item_id']).unique()

    # --- EVALUATE RECALL ---
    sources = [
        ('Repurchase', df_rep),
        ('Global Trend (Top 100)', df_global_trend),
        ('Multi-Category (Top 3x20)', df_multi_cat),
        ('Local Trend (Top 50)', df_local_trend),
        ('Brand Loyalty (Top 3x5)', df_brand)
    ]
    
    all_cands = []
    print("\n--- Individual Source Recall ---")
    for name, df in sources:
        h = df.join(truth, on=['customer_id', 'item_id'], how='inner').height
        print(f"  {name}: {h/truth.height:.4f} ({h} hits, {df.height//10000} cands/user)")
        all_cands.append(df)
        
    combined = pl.concat(all_cands).unique()
    combined_hits = combined.join(truth, on=['customer_id', 'item_id'], how='inner').height
    print(f"\n--- Combined V7 Recall (without I2I/SVD) ---")
    print(f"  Total Candidates per User (Avg): {combined.height / 10000:.1f}")
    print(f"  Combined Recall: {combined_hits/truth.height:.4f} ({combined_hits}/{truth.height})")

if __name__ == "__main__":
    check_v7_recall()
