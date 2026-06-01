
import gc
import polars as pl
import numpy as np
import lightgbm as lgb
import pickle
from datetime import datetime
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfTransformer

# ==============================================================
# 1. FILE PATHS  (replace on Kaggle)
# ==============================================================
TRANSACTION_PATH = "d:/CS116/ProjectNumberOne/transaction_full_2025.parquet"
ITEM_PATH        = "d:/CS116/ProjectNumberOne/items.parquet"
OUTPUT_PATH      = "/kaggle/working/submission_jan.pkl"

CHUNK_SIZE = 100000   # 100k users per chunk (Takes ~17GB RAM, perfect for Kaggle 30GB CPU)


# ==============================================================
# 2. HELPER FUNCTIONS  (archetype-based retriever, OOM-safe)
# ==============================================================
import gc, pickle, re
import polars as pl
import numpy as np
import lightgbm as lgb
from datetime import datetime
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

def standardize_age(text):
    raw_text = str(text).strip()
    clean_text = raw_text.lower()
    if re.search(r'(\*|x\d|cm)', clean_text): return None
    if re.search(r'\bb\d{2}\b', clean_text): return None
    if 's17' in clean_text: return 1.0
    if '110' in clean_text: return 5.0
    if "không xác định" in clean_text or not clean_text or clean_text == "none": return None
    
    diaper_map = {
        r'\bnb\b': 0.0, r'\bss\b': 0.0, r'\bsơ sinh\b': 0.0,
        r'\bs\b': 0.25, r'\bm\b': 0.6, r'\bl\b': 1.2,
        r'\bxl\b': 2.0, r'\bxxl\b': 3.5
    }
    for pattern, val in diaper_map.items():
        if re.search(pattern, clean_text): return val

    range_match = re.search(r'(\d+\.?\d*)\s*-\s*(\d+\.?\d*)', clean_text)
    if range_match:
        s, e = float(range_match.group(1)), float(range_match.group(2))
        avg = (s + e) / 2
        if any(x in clean_text for x in ['m', 'tháng']): return round(avg / 12, 3)
        return avg

    m_match = re.search(r'(\d+\.?\d*)\s*(m|tháng)', clean_text)
    if m_match: return round(float(m_match.group(1)) / 12, 3)
    
    y_match = re.search(r'(\d+\.?\d*)\s*(y|t|tuổi)', clean_text)
    if y_match: return float(y_match.group(1))

    pure_num = re.search(r'^(\d+)$', clean_text)
    if pure_num:
        val = float(pure_num.group(1))
        return round(val/12, 3) if val > 6 else val

    return None

def load_items():
    return pl.scan_parquet(ITEM_PATH).select(
        ["item_id", "category_l1", "brand", "price", "description"]
    ).with_columns([
        pl.col("item_id").cast(pl.Utf8),
        pl.col("category_l1").cast(pl.Utf8).fill_null("Unknown"),
        pl.col("brand").cast(pl.Utf8).fill_null("Unknown"),
        pl.col("price").cast(pl.Float32).fill_null(0.0),
        pl.col("description").cast(pl.Utf8).fill_null("Unknown"),
    ]).with_columns([
        pl.col("description").map_elements(standardize_age, return_dtype=pl.Float64).cast(pl.Float32).alias("i_target_age_years")
    ]).drop("description").collect()


def scan_tx(cutoff_start=None, cutoff_end=None):
    lf = pl.scan_parquet(TRANSACTION_PATH).with_columns([
        pl.col("customer_id").cast(pl.Int64),
        pl.col("item_id").cast(pl.Utf8),
        pl.col("quantity").cast(pl.Float32).fill_null(1.0),
        pl.col("location").cast(pl.Int32),
        pl.col("updated_date").cast(pl.Datetime).alias("event_ts"),
        pl.col("price").cast(pl.Float32).fill_null(0.0),
    ]).drop("updated_date")
    if cutoff_start:
        lf = lf.filter(pl.col("event_ts") >= cutoff_start)
    if cutoff_end:
        lf = lf.filter(pl.col("event_ts") < cutoff_end)
    valid_users = lf.select("customer_id").unique().head(150).collect().get_column("customer_id")
    return lf.filter(pl.col("customer_id").is_in(valid_users))


# ── PRECOMPUTE ALL LOOKUP TABLES (small, reusable) ─────────────

def precompute_lookup_tables(hist_tx, items, all_users):
    '''
    Precomputes all transaction-only lookup tables for candidate generation and features.
    '''
    tables = {}
    max_ts = hist_tx["event_ts"].max()
    
    # ── User Archetypes Classification ──
    print("  [LUT] user archetypes...")
    target_users = pl.DataFrame({"customer_id": all_users})
    hist_items = hist_tx.join(items.select(["item_id", "category_l1", "brand", "i_target_age_years"]), on="item_id", how="left")
    
    loc_hhi = (
        hist_tx.group_by(["customer_id", "location"]).agg(pl.len().alias("visits"))
        .with_columns((pl.col("visits") / pl.col("visits").sum().over("customer_id")).alias("share"))
        .group_by("customer_id").agg([(pl.col("share") * pl.col("share")).sum().alias("loc_hhi")])
    )
    
    cat_hhi = (
        hist_items.group_by(["customer_id", "category_l1"]).agg(pl.len().alias("cat_visits"))
        .with_columns((pl.col("cat_visits") / pl.col("cat_visits").sum().over("customer_id")).alias("cat_share"))
        .group_by("customer_id").agg([
            (pl.col("cat_share") * pl.col("cat_share")).sum().alias("cat_hhi"),
            pl.col("category_l1").n_unique().alias("unique_cats")
        ])
    )
    
    profile = (
        hist_tx.group_by("customer_id")
        .agg([
            pl.len().alias("total_tx"),
            pl.col("item_id").n_unique().alias("unique_items"),
            (pl.lit(max_ts) - pl.col("event_ts").min()).dt.total_days().alias("tenure_days"),
            (pl.lit(max_ts) - pl.col("event_ts").max()).dt.total_days().alias("recency_days"),
        ])
        .join(loc_hhi, on="customer_id", how="left")
        .join(cat_hhi, on="customer_id", how="left")
        .with_columns([
            pl.col("loc_hhi").fill_null(1.0),
            pl.col("cat_hhi").fill_null(1.0),
            pl.col("unique_cats").fill_null(1),
            pl.col("recency_days").fill_null(999.0)
        ])
        .with_columns(
            pl.when(pl.col("recency_days") >= 90).then(pl.lit("Dormant"))
            .when(pl.col("tenure_days") <= 60).then(pl.lit("New"))
            .when((pl.col("cat_hhi") >= 0.7) & (pl.col("total_tx") >= 3)).then(pl.lit("Habitual"))
            .when(pl.col("unique_cats") >= 4).then(pl.lit("Explorer"))
            .otherwise(pl.lit("Standard"))
            .alias("archetype")
        )
    )
    
    tables["user_archetypes"] = (
        target_users.join(profile, on="customer_id", how="left").with_columns(
            pl.col("archetype").fill_null("Dormant")
        )
    )
    del target_users, loc_hhi, cat_hhi, profile; gc.collect()

    print("  [LUT] user child age estimates...")
    tx_with_age = hist_items.filter(pl.col("i_target_age_years").is_not_null())
    tables["user_child_age_estimates"] = (
        tx_with_age.sort(["customer_id", "event_ts"], descending=[False, True])
        .group_by("customer_id").head(1)
        .with_columns(
            (pl.col("i_target_age_years") + (pl.lit(max_ts) - pl.col("event_ts")).dt.total_days() / 365.0).cast(pl.Float32).alias("u_child_age_estimate")
        )
        .select(["customer_id", "u_child_age_estimate"])
    )
    del tx_with_age; gc.collect()

    # ── ui_hist: Repeat filtered history ──
    print("  [LUT] ui_hist (repeat filtered)...")
    non_repeatables = ["Thời trang", "Đồ chơi & Sách", "Phụ kiện", "Gói Hội Viên"]
    tables["ui_hist"] = (
        hist_items
        .group_by(["customer_id", "item_id", "category_l1"])
        .agg([
            pl.len().cast(pl.Float32).alias("ui_purchases"),
            (pl.col("event_ts").max() - pl.col("event_ts").min()).dt.total_days().cast(pl.Float32).alias("ui_duration"),
            pl.col("event_ts").max().alias("last_purchase_ts"),
        ])
        .filter(
            ~pl.col("category_l1").is_in(non_repeatables) |
            (pl.col("last_purchase_ts") >= max_ts - pl.duration(days=15))
        )
        .sort(["customer_id", "last_purchase_ts", "ui_purchases"], descending=[False, True, True])
        .with_columns(pl.int_range(1, pl.len()+1).over("customer_id").cast(pl.Int32).alias("rank"))
        .select(["customer_id", "item_id", pl.col("rank").cast(pl.Int64)])
    )

    # ── user locations & local bestsellers ──
    print("  [LUT] user locations & local popular bestsellers...")
    tables["user_loc"] = (
        hist_tx.group_by(["customer_id", "location"]).agg(pl.len().alias("v"))
        .sort(["customer_id", "v"], descending=[False, True])
        .group_by("customer_id").head(1).select(["customer_id", "location"])
    )
    recent60 = hist_tx.filter(pl.col("event_ts") >= max_ts - pl.duration(days=60))
    tables["loc_top"] = (
        recent60.group_by(["location", "item_id"]).agg(pl.col("quantity").sum().alias("qty"))
        .sort(["location", "qty"], descending=[False, True]).group_by("location").head(500)
        .with_columns(pl.int_range(1, pl.len() + 1).over("location").cast(pl.Int64).alias("rank"))
        .select(["location", "item_id", "rank"])
    )
    del recent60; gc.collect()

    # ── user top categories & category bestsellers ──
    print("  [LUT] category popular bestsellers...")
    item_cat = items.select(["item_id", "category_l1"])
    
    print("  [LUT] location assortment & category habituation...")
    tables["loc_item_whitelist"] = (
        hist_tx.select(["location", "item_id"]).unique()
        .with_columns(pl.lit(1.0).alias("ui_in_local_stock").cast(pl.Float32))
    )
    ui_counts = hist_items.group_by(["customer_id", "category_l1", "item_id"]).agg(pl.len().alias("ui_qty"))
    tables["cat_habitual"] = (
        ui_counts.group_by("category_l1").agg([
            pl.col("ui_qty").sum().alias("total_purchases"),
            pl.col("ui_qty").filter(pl.col("ui_qty") > 1).sum().alias("repurchases")
        ])
        .with_columns((pl.col("repurchases") / pl.col("total_purchases")).cast(pl.Float32).alias("cat_habitual_score"))
        .select(["category_l1", "cat_habitual_score"])
    )
    del ui_counts; gc.collect()

    print("  [LUT] Top categories per user...")
    tables["user_cats"] = (
        hist_items.group_by(["customer_id", "category_l1"]).agg(pl.len().alias("cat_qty"))
        .sort(["customer_id", "cat_qty"], descending=[False, True]).group_by("customer_id").head(5)
        .with_columns(pl.int_range(1, pl.len() + 1).over("customer_id").cast(pl.Int64).alias("cat_rank"))
        .select(["customer_id", "category_l1", "cat_rank"])
    )

    print("  [LUT] Category bestsellers...")
    tables["cat_top"] = (
        hist_items.filter(pl.col("category_l1") != "Unknown")
        .group_by(["category_l1", "item_id"]).agg(pl.len().alias("qty"))
        .sort(["category_l1", "qty"], descending=[False, True]).group_by("category_l1").head(80)
        .with_columns(pl.int_range(1, pl.len() + 1).over("category_l1").cast(pl.Int64).alias("item_rank"))
        .select(["category_l1", "item_id", "item_rank"])
    )

    # ── user top brands & brand bestsellers ──
    print("  [LUT] Top brands per user...")
    tables["user_brands"] = (
        hist_items.filter((pl.col("brand") != "Unknown") & (pl.col("brand") != "Không xác định"))
        .group_by(["customer_id", "brand"]).agg(pl.len().alias("brand_qty"))
        .sort(["customer_id", "brand_qty"], descending=[False, True]).group_by("customer_id").head(5)
        .with_columns(pl.int_range(1, pl.len() + 1).over("customer_id").cast(pl.Int64).alias("brand_rank"))
        .select(["customer_id", "brand", "brand_rank"])
    )

    print("  [LUT] Brand bestsellers...")
    tables["brand_top"] = (
        hist_items.filter((pl.col("brand") != "Unknown") & (pl.col("brand") != "Không xác định"))
        .group_by(["brand", "item_id"]).agg(pl.len().alias("qty"))
        .sort(["brand", "qty"], descending=[False, True]).group_by("brand").head(50)
        .with_columns(pl.int_range(1, pl.len() + 1).over("brand").cast(pl.Int64).alias("item_rank"))
        .select(["brand", "item_id", "item_rank"])
    )

    # ── global bestsellers ──
    print("  [LUT] Global bestsellers...")
    tables["global_top"] = (
        hist_tx.group_by("item_id").agg(pl.len().alias("qty"))
        .sort("qty", descending=True).head(300)
        .with_columns(pl.int_range(1, pl.len() + 1).cast(pl.Int64).alias("rank"))
        .select(["item_id", "rank"])
    )

    # ── category trending ──
    print("  [LUT] category trending...")
    t_recent = max_ts - pl.duration(days=30)
    t_prior = max_ts - pl.duration(days=60)
    rs = hist_tx.filter(pl.col("event_ts") >= t_recent).group_by("item_id").agg(pl.col("quantity").sum().alias("qty_recent"))
    ps = hist_tx.filter((pl.col("event_ts") >= t_prior) & (pl.col("event_ts") < t_recent)).group_by("item_id").agg(pl.col("quantity").sum().alias("qty_prior"))
    tables["cat_trend"] = (
        rs.join(ps, on="item_id", how="full").fill_null(0.0)
        .with_columns((pl.col("qty_recent") - pl.col("qty_prior")).alias("momentum"))
        .join(items.select(["item_id", "category_l1"]), on="item_id", how="left").filter(pl.col("category_l1") != "Unknown")
        .sort(["category_l1", "momentum"], descending=[False, True]).group_by("category_l1").head(50)
        .with_columns(pl.int_range(1, pl.len() + 1).over("category_l1").cast(pl.Int64).alias("item_rank"))
        .select(["category_l1", "item_id", "item_rank"])
    )
    del rs, ps; gc.collect()

    # ── SVD & I2I Factorization (30 components, transaction-only, memory-safe) ──
    print("  [LUT] SVD (30 components)...", flush=True)
    tx_weighted = (
        hist_tx
        .with_columns(((max_ts - pl.col("event_ts")).dt.total_days() / 30.0).alias("months_ago"))
        .with_columns((pl.col("quantity") * pl.lit(0.70).pow(pl.col("months_ago"))).cast(pl.Float32).alias("weight"))
        .group_by(["customer_id", "item_id"]).agg(pl.col("weight").sum().alias("weight"))
    )
    u_map = tx_weighted["customer_id"].unique().to_list()
    i_map = hist_tx["item_id"].unique().to_list()
    
    # Representative sample of up to 50,000 users for training SVD and I2I
    np.random.seed(42)
    if len(u_map) > 50000:
        svd_users = np.random.choice(u_map, size=50000, replace=False)
    else:
        svd_users = np.array(u_map)
        
    u_df_train = pl.DataFrame({"customer_id": svd_users, "u_idx": np.arange(len(svd_users), dtype=np.int32)})
    i_df = pl.DataFrame({"item_id": i_map, "i_idx": np.arange(len(i_map), dtype=np.int32)})
    
    idx_train = tx_weighted.join(u_df_train, on="customer_id").join(i_df, on="item_id")
    
    mtx_train = csr_matrix(
        (idx_train["weight"].to_numpy(), (idx_train["u_idx"].to_numpy(), idx_train["i_idx"].to_numpy())),
        shape=(len(svd_users), len(i_map))
    )
    del idx_train; gc.collect()
    print("  [LUT] SVD embeddings (100 components)...", flush=True)
    
    print(f"  [LUT] Applying TF-IDF to matrix...", flush=True)
    mtx_train.data = np.log1p(mtx_train.data) * 2.0
    tfidf = TfidfTransformer()
    mtx_train = tfidf.fit_transform(mtx_train)
    
    # Using arpack solver
    k_comp = min(100, len(svd_users) - 1)
    k_comp_30 = min(30, len(svd_users) - 1)
    svd = TruncatedSVD(n_components=k_comp_30, algorithm='arpack', random_state=42)
    u_emb = svd.fit_transform(mtx_train).astype(np.float32)
    i_emb = svd.components_.T.astype(np.float32)  # shape: (N_items, 100)
    
    print("  [LUT] I2I similarity...", flush=True)
    norm_m = normalize(mtx_train, norm='l2', axis=0)
    i2i_sim = (norm_m.T.dot(norm_m))  # keep sparse -- do NOT call .toarray() here
    i2i_sim = i2i_sim.tocsr()  # ensure CSR format for fast row slicing
    i2i_sim.setdiag(0)
    i2i_sim.eliminate_zeros()
    del mtx_train, norm_m; gc.collect()
    
    # Full mtx for projection during inference
    u_df_all = pl.DataFrame({"customer_id": u_map, "u_idx": np.arange(len(u_map), dtype=np.int32)})
    idx_all = tx_weighted.join(u_df_all, on="customer_id").join(i_df, on="item_id")
    mtx_all = csr_matrix(
        (idx_all["weight"].to_numpy(), (idx_all["u_idx"].to_numpy(), idx_all["i_idx"].to_numpy())),
        shape=(len(u_map), len(i_map))
    )
    mtx_all.data = np.log1p(mtx_all.data) * 2.0
    del idx_all, tx_weighted; gc.collect()
    
    tables["svd_i_emb"] = i_emb
    tables["u_idx_df"] = u_df_all
    tables["svd_i_arr"] = np.array(i_map, dtype=object)  # item string IDs
    tables["i2i_sim"] = i2i_sim  # sparse CSR -- never convert to dense
    tables["mtx"] = mtx_all
    del u_df_train, i_df; gc.collect()

    # ── Champion Feature Precomputations ──
    print("  [LUT] Champion feature precomputations...")
    u_loc_hhi = (
        hist_tx.group_by(["customer_id", "location"]).agg(pl.len().alias("loc_qty"))
        .with_columns((pl.col("loc_qty") / pl.col("loc_qty").sum().over("customer_id")).alias("share"))
        .group_by("customer_id").agg([(pl.col("share") * pl.col("share")).sum().alias("u_loc_hhi")])
    )
    u_cat_hhi = (
        hist_items.group_by(["customer_id", "category_l1"]).agg(pl.len().alias("cat_qty"))
        .with_columns((pl.col("cat_qty") / pl.col("cat_qty").sum().over("customer_id")).alias("share"))
        .group_by("customer_id").agg([(pl.col("share") * pl.col("share")).sum().alias("u_cat_hhi")])
    )
    
    item_brand = items.select(["item_id", "brand"])
    u_brand_hhi = (
        hist_items.group_by(["customer_id", "brand"]).agg(pl.len().alias("brand_qty"))
        .with_columns((pl.col("brand_qty") / pl.col("brand_qty").sum().over("customer_id")).alias("share"))
        .group_by("customer_id").agg([(pl.col("share") * pl.col("share")).sum().alias("u_brand_hhi")])
    )
    
    tables["user_features_champ"] = (
        hist_tx.group_by("customer_id").agg([
            pl.len().alias("u_total_tx"),
            pl.col("quantity").sum().alias("u_total_qty"),
            (pl.col("quantity") * pl.col("price")).sum().alias("u_total_spend"),
            pl.col("item_id").n_unique().alias("u_unique_items"),
            (max_ts - pl.col("event_ts").min()).dt.total_days().alias("u_tenure_days"),
            (max_ts - pl.col("event_ts").max()).dt.total_days().alias("u_recency_days"),
        ])
        .join(u_loc_hhi, on="customer_id", how="left")
        .join(u_cat_hhi, on="customer_id", how="left")
        .join(u_brand_hhi, on="customer_id", how="left")
        .with_columns([
            (pl.col("u_total_qty") / pl.col("u_total_tx")).alias("u_avg_basket_size"),
            (pl.col("u_total_spend") / pl.col("u_total_tx")).alias("u_avg_order_value"),
            (pl.col("u_total_tx") / (pl.col("u_tenure_days") + 1.0)).alias("u_tx_velocity"),
            pl.col("u_loc_hhi").fill_null(1.0),
            pl.col("u_cat_hhi").fill_null(1.0),
            pl.col("u_brand_hhi").fill_null(1.0),
            pl.col("u_recency_days").fill_null(999.0)
        ])
        .cast({
            "u_total_tx": pl.Float32,
            "u_total_qty": pl.Float32,
            "u_total_spend": pl.Float32,
            "u_unique_items": pl.Float32,
            "u_tenure_days": pl.Float32,
            "u_recency_days": pl.Float32,
            "u_loc_hhi": pl.Float32,
            "u_cat_hhi": pl.Float32,
            "u_brand_hhi": pl.Float32,
            "u_avg_basket_size": pl.Float32,
            "u_avg_order_value": pl.Float32,
            "u_tx_velocity": pl.Float32,
        })
    )
    del u_loc_hhi, u_cat_hhi, u_brand_hhi; gc.collect()

    print("    [LUT] item_features_champ...")
    t_30 = max_ts - pl.duration(days=30)
    t_60 = max_ts - pl.duration(days=60)
    t_90 = max_ts - pl.duration(days=90)

    i_sales_all = hist_tx.group_by("item_id").agg(pl.len().alias("i_sales_all"))
    i_sales_30d = hist_tx.filter(pl.col("event_ts") >= t_30).group_by("item_id").agg(pl.len().alias("i_sales_30d"))
    i_sales_60d = hist_tx.filter((pl.col("event_ts") >= t_60) & (pl.col("event_ts") < t_30)).group_by("item_id").agg(pl.len().alias("i_sales_60_to_30d"))
    i_sales_90d = hist_tx.filter((pl.col("event_ts") >= t_90) & (pl.col("event_ts") < t_60)).group_by("item_id").agg(pl.len().alias("i_sales_90_to_60d"))

    i_user_purchases = hist_tx.group_by(["item_id", "customer_id"]).agg(pl.len().alias("qty"))
    i_repeat_stats = i_user_purchases.group_by("item_id").agg([
        pl.col("qty").sum().alias("total_item_purchases"),
        pl.col("qty").filter(pl.col("qty") > 1).sum().alias("repeat_item_purchases")
    ]).with_columns(
        (pl.col("repeat_item_purchases") / pl.col("total_item_purchases")).alias("item_repeat_propensity")
    ).select(["item_id", "item_repeat_propensity"])

    i_launch = hist_tx.group_by("item_id").agg([
        (max_ts - pl.col("event_ts").min()).dt.total_days().alias("item_launch_age_days")
    ])

    cat_sales_30d = hist_tx.filter(pl.col("event_ts") >= t_30).join(items, on="item_id", how="left").group_by("category_l1").agg(pl.len().alias("cat_sales_30d"))
    cat_sales_60d = hist_tx.filter((pl.col("event_ts") >= t_60) & (pl.col("event_ts") < t_30)).join(items, on="item_id", how="left").group_by("category_l1").agg(pl.len().alias("cat_sales_60d"))
    cat_trend_feat = cat_sales_30d.join(cat_sales_60d, on="category_l1", how="left").fill_null(0).with_columns((pl.col("cat_sales_30d") - pl.col("cat_sales_60d")).alias("cat_momentum"))

    tables["item_features_champ"] = (
        items.select(["item_id", "category_l1", "brand", "price"])
        .join(i_sales_all, on="item_id", how="left")
        .join(i_sales_30d, on="item_id", how="left")
        .join(i_sales_60d, on="item_id", how="left")
        .join(i_sales_90d, on="item_id", how="left")
        .join(cat_trend_feat, on="category_l1", how="left")
        .join(i_repeat_stats, on="item_id", how="left")
        .join(i_launch, on="item_id", how="left")
        .fill_null(0)
        .with_columns([
            (pl.col("i_sales_30d") - pl.col("i_sales_60_to_30d")).alias("i_momentum_30d"),
            (pl.col("i_sales_60_to_30d") - pl.col("i_sales_90_to_60d")).alias("i_momentum_60d")
        ])
        .cast({
            "price": pl.Float32,
            "i_sales_all": pl.Float32,
            "i_sales_30d": pl.Float32,
            "i_sales_60_to_30d": pl.Float32,
            "i_sales_90_to_60d": pl.Float32,
            "cat_sales_30d": pl.Float32,
            "cat_sales_60d": pl.Float32,
            "cat_momentum": pl.Float32,
            "i_momentum_30d": pl.Float32,
            "i_momentum_60d": pl.Float32,
            "item_repeat_propensity": pl.Float32,
            "item_launch_age_days": pl.Float32,
        })
    )
    del i_sales_all, i_sales_30d, i_sales_60d, i_sales_90d, cat_sales_30d, cat_sales_60d, cat_trend_feat; gc.collect()

    print("    [LUT] ui_features_champ...")
    tables["ui_features_champ"] = (
        hist_tx.group_by(["customer_id", "item_id"])
        .agg([
            pl.len().alias("ui_purchase_count"),
            pl.col("quantity").sum().alias("ui_total_qty"),
            (max_ts - pl.col("event_ts").max()).dt.total_days().alias("ui_days_since_last"),
            (max_ts - pl.col("event_ts").min()).dt.total_days().alias("ui_days_since_first")
        ])
        .join(tables["user_child_age_estimates"], on="customer_id", how="left")
        .join(items.select(["item_id", "i_target_age_years"]), on="item_id", how="left")
        .with_columns([
            (pl.col("ui_total_qty") / pl.col("ui_purchase_count")).alias("ui_avg_qty_per_order"),
            (pl.col("ui_purchase_count") / (pl.col("ui_days_since_first") + 1.0)).alias("ui_purchase_velocity"),
            (pl.col("ui_days_since_last") > 22.0).cast(pl.Float32).alias("ui_replenishment_due"),
            pl.when(pl.col("u_child_age_estimate").is_not_null() & pl.col("i_target_age_years").is_not_null()).then(
                pl.col("u_child_age_estimate") - pl.col("i_target_age_years")
            ).otherwise(0.0).alias("ui_age_delta"),
            (-pl.col("ui_days_since_last") / 15.0).exp().cast(pl.Float32).alias("ui_recency_weight")
        ])
        .cast({
            "ui_purchase_count": pl.Float32,
            "ui_total_qty": pl.Float32,
            "ui_days_since_last": pl.Float32,
            "ui_days_since_first": pl.Float32,
            "ui_avg_qty_per_order": pl.Float32,
            "ui_purchase_velocity": pl.Float32,
            "ui_replenishment_due": pl.Float32,
            "ui_age_delta": pl.Float32,
            "ui_recency_weight": pl.Float32,
        })
    )

    print("    [LUT] user-category and user-brand affinities...")
    tables["u_cat_affinity"] = (
        hist_items.group_by(["customer_id", "category_l1"])
        .agg(pl.len().alias("u_cat_purchases"))
        .with_columns((pl.col("u_cat_purchases") / pl.col("u_cat_purchases").sum().over("customer_id")).alias("u_cat_share_of_wallet"))
        .cast({"u_cat_purchases": pl.Float32, "u_cat_share_of_wallet": pl.Float32})
    )
    tables["u_brand_affinity"] = (
        hist_items.group_by(["customer_id", "brand"])
        .agg(pl.len().alias("u_brand_purchases"))
        .with_columns((pl.col("u_brand_purchases") / pl.col("u_brand_purchases").sum().over("customer_id")).alias("u_brand_share_of_wallet"))
        .cast({"u_brand_purchases": pl.Float32, "u_brand_share_of_wallet": pl.Float32})
    )
    del hist_items; gc.collect()

    print("    [LUT] Categorical mappings...")
    cat1_unique = items["category_l1"].unique().to_list()
    brand_unique = items["brand"].unique().to_list()
    tables["cat1_map"] = {c: i for i, c in enumerate(cat1_unique)}
    tables["brand_map"] = {b: i for i, b in enumerate(brand_unique)}

    print("  [LUT] All lookup tables ready.")
    return tables


def candidates_for_chunk(chunk_users, tables):
    '''
    Dynamic archetype-based candidate generation per chunk of users.
    '''
    chunk_df = pl.DataFrame({"customer_id": chunk_users})
    
    chA = (
        tables["ui_hist"].filter(pl.col("customer_id").is_in(chunk_users))
        .select(["customer_id", "item_id", "rank"])
        .with_columns(pl.lit("A_history").alias("channel"))
    )
    
    chB = (
        tables["user_loc"].filter(pl.col("customer_id").is_in(chunk_users))
        .join(tables["loc_top"], on="location", how="inner")
        .select(["customer_id", "item_id", "rank"])
        .with_columns(pl.lit("B_local").alias("channel"))
    )
    
    u_idx_df = tables["u_idx_df"]
    i_emb = tables["svd_i_emb"]
    i_arr = tables["svd_i_arr"]
    mtx = tables["mtx"]
    
    chunk_u_idx = pl.DataFrame({"customer_id": chunk_users}).join(u_idx_df, on="customer_id", how="inner")
    target = chunk_u_idx["customer_id"].to_numpy()
    t_idx = chunk_u_idx["u_idx"].to_numpy()
    
    svd_k = 1000
    if len(target) > 0:
        # OOM-safe SVD top-k: compute user embeddings (5000x50), then score against item embeddings.
        # NEVER materialize the full (5000 x N_items) scores matrix -- that's 4GB per chunk!
        # Instead: u_emb (5000x50) @ i_emb.T (50xN_items) but only take top-k via argpartition.
        mtx_chunk = mtx[t_idx]          # sparse (5000 x N_items)
        u_emb_chunk = mtx_chunk.dot(i_emb)  # dense (5000 x 50) -- only 1MB
        # Materialize scores row-wise in batches of 2000 users to cap peak at ~2000x200k x4 = 1.6GB
        SVD_BATCH = 2000
        all_top_k_users = []
        all_top_k_items = []
        all_top_k_ranks = []
        k = min(svd_k, i_emb.shape[0])
        for bi in range(0, len(target), SVD_BATCH):
            u_batch = u_emb_chunk[bi:bi+SVD_BATCH]  # (<=2000, 50)
            scores_batch = u_batch @ i_emb.T           # (<=2000, N_items)
            top_k_b = np.argpartition(-scores_batch, kth=min(k-1, scores_batch.shape[1]-1), axis=1)[:, :k]
            order_b = np.argsort(-scores_batch[np.arange(len(u_batch))[:, None], top_k_b], axis=1)
            top_k_b = top_k_b[np.arange(len(u_batch))[:, None], order_b]
            all_top_k_users.append(np.repeat(target[bi:bi+SVD_BATCH], k))
            all_top_k_items.append(i_arr[top_k_b.flatten()])
            all_top_k_ranks.append(np.tile(np.arange(1, k+1), len(u_batch)))
            del scores_batch, top_k_b, order_b
        del u_emb_chunk, mtx_chunk; gc.collect()
        chC = pl.DataFrame({
            "customer_id": pl.Series(np.concatenate(all_top_k_users), dtype=pl.Int64),
            "item_id": np.concatenate(all_top_k_items),
            "rank": pl.Series(np.concatenate(all_top_k_ranks), dtype=pl.Int64),
            "channel": pl.Series(["C_svd"] * sum(len(x) for x in all_top_k_users), dtype=pl.Utf8),
        })
        del all_top_k_users, all_top_k_items, all_top_k_ranks; gc.collect()
    else:
        chC = pl.DataFrame(schema={"customer_id": pl.Int64, "item_id": pl.Utf8, "rank": pl.Int64, "channel": pl.Utf8})
        
    i2i_sim = tables["i2i_sim"]  # sparse CSR (N_items x N_items)
    i2i_k = 1000
    d2i_users, d2i_items, d2i_ranks = [], [], []
    if len(target) > 0:
        # Batched I2I: score I2I_BATCH users at a time via vectorized .toarray().
        # Each batch: (1000 x N_items)
        I2I_BATCH = 1000
        mtx_sub = mtx[t_idx]  # sparse (n_chunk x N_items)
        n_items  = i2i_sim.shape[1]
        k_i2i    = min(i2i_k, n_items)
        for bi in range(0, len(target), I2I_BATCH):
            batch_mtx = mtx_sub[bi:bi + I2I_BATCH]
            scores_b  = batch_mtx.dot(i2i_sim).toarray().astype(np.float32)  # (<=1000, N_items)
            batch_tgt = target[bi:bi + I2I_BATCH]
            n_b = len(batch_tgt)
            top_i  = np.argpartition(-scores_b, kth=k_i2i - 1, axis=1)[:, :k_i2i]
            order_i = np.argsort(-scores_b[np.arange(n_b)[:, None], top_i], axis=1)
            top_i   = top_i[np.arange(n_b)[:, None], order_i]
            top_s   = scores_b[np.arange(n_b)[:, None], top_i]
            del scores_b, order_i
            
            mask = top_s > 0.0
            valid_counts = mask.sum(axis=1)
            d2i_users.append(np.repeat(batch_tgt, valid_counts))
            d2i_items.append(i_arr[top_i[mask]])
            rank_mtx = np.tile(np.arange(1, k_i2i + 1), (n_b, 1))
            d2i_ranks.append(rank_mtx[mask])
            
            del top_i, top_s, mask, rank_mtx
        del mtx_sub; gc.collect()
    if len(d2i_users) > 0:
        chD = pl.DataFrame({
            "customer_id": pl.Series(np.concatenate(d2i_users), dtype=pl.Int64),
            "item_id": pl.Series(np.concatenate(d2i_items), dtype=pl.Utf8),
            "rank": pl.Series(np.concatenate(d2i_ranks), dtype=pl.Int64),
            "channel": pl.Series(["D_i2i"] * sum(len(x) for x in d2i_users), dtype=pl.Utf8),
        })
    else:
        chD = pl.DataFrame(schema={"customer_id": pl.Int64, "item_id": pl.Utf8, "rank": pl.Int64, "channel": pl.Utf8})
    del d2i_users, d2i_items, d2i_ranks; gc.collect()
        
    chE = (
        tables["user_cats"].filter(pl.col("customer_id").is_in(chunk_users))
        .join(tables["cat_top"], on="category_l1", how="inner")
        .sort(["customer_id", "cat_rank", "item_rank"])
        .group_by("customer_id").head(400)
        .with_columns(pl.int_range(1, pl.len() + 1).over("customer_id").cast(pl.Int64).alias("rank"),
                      pl.lit("E_cat").alias("channel"))
        .select(["customer_id", "item_id", "rank", "channel"])
    )
    
    chF = (
        tables["user_brands"].filter(pl.col("customer_id").is_in(chunk_users))
        .join(tables["brand_top"], on="brand", how="inner")
        .sort(["customer_id", "brand_rank", "item_rank"])
        .group_by("customer_id").head(250)
        .with_columns(pl.int_range(1, pl.len() + 1).over("customer_id").cast(pl.Int64).alias("rank"),
                      pl.lit("F_brand").alias("channel"))
        .select(["customer_id", "item_id", "rank", "channel"])
    )
    
    chG = (
        chunk_df.join(tables["global_top"].with_columns(pl.lit(1).alias("_k")), how="cross").drop("_k")
        .with_columns(pl.lit("G_global").alias("channel"))
    )
    
    chH = (
        tables["user_cats"].filter(pl.col("customer_id").is_in(chunk_users))
        .join(tables["cat_trend"], on="category_l1", how="inner")
        .sort(["customer_id", "cat_rank", "item_rank"])
        .group_by("customer_id").head(150)
        .with_columns(pl.int_range(1, pl.len() + 1).over("customer_id").cast(pl.Int64).alias("rank"),
                      pl.lit("H_trend").alias("channel"))
        .select(["customer_id", "item_id", "rank", "channel"])
    )
    
    stacked = pl.concat([chA, chB, chC, chD, chE, chF, chG, chH])
    del chA, chB, chC, chD, chE, chF, chG, chH; gc.collect()
    
    arch_chunk = tables["user_archetypes"].filter(pl.col("customer_id").is_in(chunk_users))
    merged = stacked.join(arch_chunk.select(["customer_id", "archetype"]), on="customer_id", how="inner")
    
    weight_map = {
        "Habitual": {"A_history": 5.0, "B_local": 1.0, "C_svd": 0.5, "D_i2i": 0.5, "E_cat": 2.0, "F_brand": 3.0, "G_global": 0.1, "H_trend": 0.5},
        "Explorer": {"A_history": 1.0, "B_local": 1.0, "C_svd": 4.0, "D_i2i": 4.0, "E_cat": 3.0, "F_brand": 1.0, "G_global": 1.0, "H_trend": 2.0},
        "Dormant":  {"A_history": 3.0, "B_local": 4.0, "C_svd": 0.5, "D_i2i": 0.5, "E_cat": 1.0, "F_brand": 2.0, "G_global": 5.0, "H_trend": 1.0},
        "New":      {"A_history": 0.5, "B_local": 3.0, "C_svd": 2.5, "D_i2i": 2.5, "E_cat": 3.0, "F_brand": 1.0, "G_global": 4.0, "H_trend": 4.0},
        "Standard": {"A_history": 2.0, "B_local": 2.0, "C_svd": 1.5, "D_i2i": 1.5, "E_cat": 2.0, "F_brand": 2.0, "G_global": 1.0, "H_trend": 1.5},
    }
    w_df = pl.DataFrame([
        {"archetype": a, "channel": c, "ch_weight": w}
        for a, cw in weight_map.items() for c, w in cw.items()
    ])
    
    scored = (
        merged.join(w_df, on=["archetype", "channel"], how="left")
        .with_columns(pl.col("ch_weight").fill_null(1.0))
        .with_columns((pl.col("ch_weight") / (pl.col("rank").cast(pl.Float32) + 60.0)).alias("score"))
        .group_by(["customer_id", "item_id", "archetype"])
        .agg(
            pl.col("score").sum().cast(pl.Float32).alias("final_score"),
            pl.col("rank").filter(pl.col("channel") == "A_history").first().alias("rank_A_history"),
            pl.col("rank").filter(pl.col("channel") == "B_local").first().alias("rank_B_local"),
            pl.col("rank").filter(pl.col("channel") == "C_svd").first().alias("rank_C_svd"),
            pl.col("rank").filter(pl.col("channel") == "D_i2i").first().alias("rank_D_i2i"),
            pl.col("rank").filter(pl.col("channel") == "E_cat").first().alias("rank_E_cat"),
            pl.col("rank").filter(pl.col("channel") == "F_brand").first().alias("rank_F_brand"),
            pl.col("rank").filter(pl.col("channel") == "G_global").first().alias("rank_G_global"),
            pl.col("rank").filter(pl.col("channel") == "H_trend").first().alias("rank_H_trend")
        )
        .with_columns([
            pl.col("rank_A_history").fill_null(999).cast(pl.Int32),
            pl.col("rank_B_local").fill_null(999).cast(pl.Int32),
            pl.col("rank_C_svd").fill_null(999).cast(pl.Int32),
            pl.col("rank_D_i2i").fill_null(999).cast(pl.Int32),
            pl.col("rank_E_cat").fill_null(999).cast(pl.Int32),
            pl.col("rank_F_brand").fill_null(999).cast(pl.Int32),
            pl.col("rank_G_global").fill_null(999).cast(pl.Int32),
            pl.col("rank_H_trend").fill_null(999).cast(pl.Int32)
        ])
        .sort(["customer_id", "final_score"], descending=[False, True])
    )
    
    budget_map = {
        "Habitual": 125,
        "Explorer": 400,
        "Dormant": 60,
        "New": 175,
        "Standard": 225
    }
    b_df = pl.DataFrame([{"archetype": k, "budget": v} for k, v in budget_map.items()])
    
    final_chunk = (
        scored.with_columns(pl.int_range(1, pl.len() + 1).over("customer_id").cast(pl.Int64).alias("final_rank"))
        .join(b_df, on="archetype", how="left")
        .with_columns(pl.col("budget").fill_null(100))
        .filter(pl.col("final_rank") <= pl.col("budget"))
        .select([
            "customer_id", "item_id", "final_rank", "final_score",
            "rank_A_history", "rank_B_local", "rank_C_svd", "rank_D_i2i",
            "rank_E_cat", "rank_F_brand", "rank_G_global", "rank_H_trend"
        ])
    )
    del stacked, merged, scored, b_df; gc.collect()
    return final_chunk


def assemble_dataset(candidates, target_tx, tables, is_inference=False):
    df = (
        candidates
        .join(tables["user_loc"], on="customer_id", how="left")
        .join(tables["loc_item_whitelist"], on=["location", "item_id"], how="left")
        .with_columns(pl.col("ui_in_local_stock").fill_null(0.0))
        .join(tables["user_features_champ"], on="customer_id", how="inner")
        .join(tables["item_features_champ"], on="item_id", how="inner")
        .join(tables["ui_features_champ"], on=["customer_id", "item_id"], how="left")
        .join(tables["u_cat_affinity"], on=["customer_id", "category_l1"], how="left")
        .join(tables["u_brand_affinity"], on=["customer_id", "brand"], how="left")
        .join(tables["user_child_age_estimates"], on="customer_id", how="left")
        .join(tables["cat_habitual"], on="category_l1", how="left")
    )
    
    df = df.with_columns([
        pl.col("ui_purchase_count").fill_null(0.0),
        pl.col("ui_total_qty").fill_null(0.0),
        pl.col("ui_days_since_last").fill_null(999.0),
        pl.col("ui_days_since_first").fill_null(999.0),
        pl.col("ui_avg_qty_per_order").fill_null(0.0),
        pl.col("ui_purchase_velocity").fill_null(0.0),
        pl.col("ui_replenishment_due").fill_null(0.0),
        pl.col("ui_recency_weight").fill_null(0.0),
        pl.col("u_cat_purchases").fill_null(0.0),
        pl.col("u_cat_share_of_wallet").fill_null(0.0),
        pl.col("u_brand_purchases").fill_null(0.0),
        pl.col("u_brand_share_of_wallet").fill_null(0.0),
        pl.col("u_child_age_estimate").fill_null(-99.0),
    ])
    
    df = df.with_columns([
        (pl.col("u_cat_share_of_wallet") * pl.col("i_momentum_30d")).alias("cross_cat_momentum"),
        (pl.col("u_brand_share_of_wallet") * pl.col("i_momentum_30d")).alias("cross_brand_momentum"),
        (pl.col("price") / (pl.col("u_avg_order_value") + 1.0)).alias("cross_price_ratio"),
        pl.when(pl.col("ui_purchase_count") > 0).then(
            pl.col("cat_habitual_score")
        ).otherwise(
            1.0 - pl.col("cat_habitual_score")
        ).fill_null(0.5).alias("ui_habitual_match")
    ])
    
    df = df.with_columns([
        pl.col("category_l1").replace_strict(tables["cat1_map"], default=len(tables["cat1_map"])).cast(pl.Int32).alias("category_l1_idx"),
        pl.col("brand").replace_strict(tables["brand_map"], default=len(tables["brand_map"])).cast(pl.Int32).alias("brand_idx"),
    ])
    
    # SVD score feature: for each (user, item) row compute dot(u_emb, i_emb[item])
    u_idx_df = tables["u_idx_df"]
    i_emb    = tables["svd_i_emb"]  # (N_items, n_comp)
    i_arr    = tables["svd_i_arr"]  # item string IDs as numpy array
    mtx_all  = tables["mtx"]

    unique_users = df["customer_id"].unique().to_list()
    u_map_local  = {u: idx for idx, u in enumerate(unique_users)}
    u_emb_dense  = np.zeros((len(unique_users), i_emb.shape[1]), dtype=np.float32)

    unique_u_df = pl.DataFrame({"customer_id": unique_users}, schema={"customer_id": pl.Int64})
    joined      = unique_u_df.join(u_idx_df, on="customer_id", how="inner")
    valid_u_ids  = joined["customer_id"].to_numpy()
    valid_u_idxs = joined["u_idx"].to_numpy()
    if len(valid_u_ids) > 0:
        valid_emb = mtx_all[valid_u_idxs].dot(i_emb)  # (n_valid, n_comp)
        for j, uid in enumerate(valid_u_ids):
            u_emb_dense[u_map_local[uid]] = valid_emb[j]
        del valid_emb
    del unique_u_df, joined; gc.collect()

    # Build per-row indices via Polars join (avoids giant dict replace_strict)
    i_map_df   = pl.DataFrame({"item_id": i_arr.tolist(),
                                "i_idx_local": np.arange(len(i_arr), dtype=np.int32)}, schema={"item_id": pl.Utf8, "i_idx_local": pl.Int32})
    u_map_pl   = pl.DataFrame({"customer_id": list(u_map_local.keys()),
                                "u_idx_local": list(u_map_local.values())}, schema={"customer_id": pl.Int64, "u_idx_local": pl.Int32})
    df_idx     = (df.join(i_map_df, on="item_id", how="left")
                    .join(u_map_pl,  on="customer_id", how="left"))
    u_idxs = df_idx["u_idx_local"].fill_null(-1).to_numpy()
    i_idxs = df_idx["i_idx_local"].fill_null(-1).to_numpy()
    del df_idx, i_map_df, u_map_pl; gc.collect()

    svd_scores  = np.zeros(len(df), dtype=np.float32)
    valid_mask  = (u_idxs != -1) & (i_idxs != -1)
    if valid_mask.any():
        # Vectorized dot product -- no per-dimension Python loop
        svd_scores[valid_mask] = (u_emb_dense[u_idxs[valid_mask]] * i_emb[i_idxs[valid_mask]]).sum(axis=1)
    del u_emb_dense, u_idxs, i_idxs, valid_mask; gc.collect()
    df = df.with_columns(pl.Series("svd_score", svd_scores))
    df = df.drop(["category_l1", "brand"])

    if not is_inference:
        truth = (
            target_tx
            .with_columns([pl.col("customer_id").cast(pl.Int64),
                           pl.col("item_id").cast(pl.Utf8),
                           pl.lit(1).alias("label")])
            .select(["customer_id", "item_id", "label"]).unique()
        )
        df = df.join(truth, on=["customer_id", "item_id"], how="left").with_columns(
            pl.col("label").fill_null(0))
    return df



def build_training_dataset(hist_tx, target_tx, items, sample_n, tables):
    '''Build labeled training Polars DataFrame for sample_n users.
    Used for Phase A (20k users) where the full DF fits in memory for train/valid split.
    '''
    sample_users = (
        target_tx["customer_id"].unique().cast(pl.Int64)
        .sample(n=min(sample_n, target_tx["customer_id"].n_unique()), seed=42).to_list()
    )
    sub_chunk = 2000
    total = (len(sample_users) + sub_chunk - 1) // sub_chunk
    all_chunks = []
    for i in range(0, len(sample_users), sub_chunk):
        sub = sample_users[i:i + sub_chunk]
        cands_chunk = candidates_for_chunk(sub, tables)
        engineered_chunk = assemble_dataset(cands_chunk, target_tx, tables, is_inference=False)
        all_chunks.append(engineered_chunk)
        del cands_chunk, engineered_chunk; gc.collect()
        print(f"  chunk {i // sub_chunk + 1}/{total} done")
    df = pl.concat(all_chunks)
    del all_chunks; gc.collect()
    return df


def build_lgb_dataset_streaming(target_tx, tables, final_features, sample_n):
    '''OOM-safe Phase B training for large sample_n (100k users).
    Streams 2k-user chunks directly to narrow numpy arrays (feature cols + label only).
    NEVER materializes the full Polars DataFrame -- eliminates the 45M-row concat OOM.
    Returns an lgb.Dataset ready for lgb.train().
    '''
    sample_users = (
        target_tx["customer_id"].unique().cast(pl.Int64)
        .sample(n=min(sample_n, target_tx["customer_id"].n_unique()), seed=42).to_list()
    )
    sub_chunk = 2000
    total = (len(sample_users) + sub_chunk - 1) // sub_chunk
    all_X, all_y, all_groups = [], [], []
    for i in range(0, len(sample_users), sub_chunk):
        sub = sample_users[i:i + sub_chunk]
        cands = candidates_for_chunk(sub, tables)
        df    = assemble_dataset(cands, target_tx, tables, is_inference=False)
        del cands; gc.collect()
        df = df.sort("customer_id")
        all_groups.append(df.group_by("customer_id", maintain_order=True).len()["len"].to_numpy())
        # Extract ONLY the feature + label columns as numpy -- discard all Polars metadata
        all_X.append(df[final_features].to_numpy().astype(np.float32))
        all_y.append(df["label"].to_numpy().astype(np.float32))
        del df; gc.collect()
        print(f"  chunk {i // sub_chunk + 1}/{total} done")
    X      = np.concatenate(all_X);      del all_X;      gc.collect()
    y      = np.concatenate(all_y);      del all_y;      gc.collect()
    groups = np.concatenate(all_groups); del all_groups; gc.collect()
    dataset = lgb.Dataset(X, label=y, group=groups, free_raw_data=False)
    del X, y, groups; gc.collect()
    return dataset


def infer_chunk(chunk_users, tables, final_model, FINAL_FEATURES):
    '''Predict for one chunk of users, using OOM-safe sub-chunks of 5000 users.'''
    sub_chunk = 5000
    all_top_k = []
    
    for i in range(0, len(chunk_users), sub_chunk):
        sub = chunk_users[i : i + sub_chunk]
        cands_sub = candidates_for_chunk(sub, tables)
        df_sub = assemble_dataset(cands_sub, None, tables, is_inference=True)
        del cands_sub; gc.collect()
        
        preds = final_model.predict(df_sub[FINAL_FEATURES].to_numpy())
        res_sub = df_sub.select(["customer_id", "item_id"]).with_columns(pl.Series("pred", preds))
        del df_sub; gc.collect()
        
        top_k_sub = res_sub.sort(["customer_id", "pred"], descending=[False, True]).group_by("customer_id").head(3)
        all_top_k.append(top_k_sub)
        del res_sub; gc.collect()
        
    top_k = pl.concat(all_top_k)
    del all_top_k; gc.collect()
    
    agg_df = top_k.group_by("customer_id").agg(pl.col("item_id").alias("items"))
    del top_k; gc.collect()
    
    out = {}
    for r in agg_df.iter_rows(named=True):
        cid = int(r["customer_id"])
        items = [str(x) for x in r["items"]]
        if len(items) > 0:
            out[cid] = tuple([items[0]] * 10)
    return out


items = load_items()


import optuna
import gc
from datetime import datetime
import polars as pl

print('Building Nov Train Dataset...')
hist_nov = scan_tx(cutoff_end=datetime(2025, 11, 1)).collect()
targ_nov = scan_tx(cutoff_start=datetime(2025, 11, 1), cutoff_end=datetime(2025, 12, 1)).collect()
tables_nov = precompute_lookup_tables(hist_nov, items, targ_nov['customer_id'].unique().cast(pl.Int64).to_list())

dummy_df = build_training_dataset(hist_nov, targ_nov, items, sample_n=50, tables=tables_nov)
FEATURES = [c for c in dummy_df.columns if c not in ['customer_id', 'item_id', 'label', 'final_rank']]
del dummy_df; gc.collect()

print(f'Total features used for tuning: {len(FEATURES)}')

lgb_train = build_lgb_dataset_streaming(targ_nov, tables_nov, FEATURES, sample_n=50)
del hist_nov, targ_nov, tables_nov; gc.collect()

print('Building Dec Valid Dataset...')
hist_dec = scan_tx(cutoff_end=datetime(2025, 12, 1)).collect()
targ_dec = scan_tx(cutoff_start=datetime(2025, 12, 1)).collect()
tables_dec = precompute_lookup_tables(hist_dec, items, targ_dec['customer_id'].unique().cast(pl.Int64).to_list())

lgb_valid = build_lgb_dataset_streaming(targ_dec, tables_dec, FEATURES, sample_n=50)
del hist_dec, targ_dec, tables_dec; gc.collect()

valid_y = lgb_valid.get_label()
valid_groups = lgb_valid.get_group()

def compute_hybrid_score(preds, labels, groups):
    import numpy as np
    idx = 0
    mrr_sum = p10_sum = map10_sum = hr10_sum = 0.0
    num_users = len(groups)
    
    for g in groups:
        p = preds[idx:idx+g]
        y = labels[idx:idx+g]
        idx += g
        
        if y.sum() == 0:
            num_users -= 1
            continue
            
        order = np.argsort(-p)
        y_sorted = y[order]
        y_top10 = y_sorted[:10]
        
        hits = y_top10.sum()
        if hits > 0:
            hr10_sum += 1.0
            p10_sum += hits / 10.0
            
            hit_indices = np.where(y_sorted > 0)[0]
            mrr_sum += 1.0 / (hit_indices[0] + 1)
            
            hit_positions = np.where(y_top10 > 0)[0]
            precisions = np.arange(1, len(hit_positions) + 1) / (hit_positions + 1)
            map10_sum += precisions.sum() / min(10.0, y.sum())
            
    if num_users == 0: return 0.0
    return 0.35 * (mrr_sum / num_users) + 0.30 * (p10_sum / num_users) + 0.20 * (map10_sum / num_users) + 0.15 * (hr10_sum / num_users)

import lightgbm as lgb
def objective(trial):
    param = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'eval_at': 10,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 31, 255),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 50, 1000),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-3, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-3, 10.0, log=True),
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1,
        'feature_pre_filter': False
    }

    model = lgb.train(
        param, lgb_train, num_boost_round=100,
        valid_sets=[lgb_valid], valid_names=['valid'],
        callbacks=[lgb.early_stopping(15, verbose=False)]
    )
    
    preds = model.predict(lgb_valid.data)
    score = compute_hybrid_score(preds, valid_y, valid_groups)
    return score

study = optuna.create_study(sampler=optuna.samplers.TPESampler(n_startup_trials=0, multivariate=True, constant_liar=True), direction='maximize')
study.optimize(objective, n_trials=1)

print('==============================================')
print('Best trial:')
trial = study.best_trial
print(f'  Value (Hybrid Score): {trial.value}')
print('  Params: ')
for key, value in trial.params.items():
    print(f'    "{key}": {value},')
print('==============================================')
