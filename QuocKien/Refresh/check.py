
import gc
import polars as pl
import numpy as np
import lightgbm as lgb
import pickle
import optuna
from datetime import datetime
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

# ==============================================================
# 1. FILE PATHS  (replace on Kaggle)
# ==============================================================
TRANSACTION_PATH = r"d:\CS116\ProjectNumberOne\transaction_full_2025.parquet"
ITEM_PATH        = r"d:\CS116\ProjectNumberOne\items.parquet"
EVENT_PATH       = "/kaggle/input/datasets/kinonquc/qkindataset2/event_full_2025.parquet"

MODEL_OUT_PATH   = "/kaggle/working/final_lgb_model.txt"
FEATURES_OUT_PATH= "/kaggle/working/final_features.pkl"

import gc, pickle, re
import polars as pl
import numpy as np
import lightgbm as lgb
from datetime import datetime
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfTransformer


# ==============================================================
# HARDCODED FLAGS FOR GPU TRAINING
# ==============================================================
flags = {
    "filter_dead": True,
    "use_events": False,
    "copurchase": True,
    "rich_metadata": True,
    "discount_features": True
}

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


def load_items(flags):
    df = pl.read_parquet(ITEM_PATH)
    # Always load full schema now
    cols = ["item_id", "category_l1", "brand", "price", "description", "sale_status"]
    if flags.get("rich_metadata"):
        cols += ["category_l2", "category_l3", "manufacturer", "size"]
    df = df.select([c for c in cols if c in df.columns]).with_columns([
        pl.col("item_id").cast(pl.Utf8),
        pl.col("category_l1").cast(pl.Utf8).fill_null("Unknown"),
        pl.col("brand").cast(pl.Utf8).fill_null("Unknown"),
        pl.col("price").cast(pl.Float32).fill_null(0.0),
        pl.col("sale_status").cast(pl.Int32).fill_null(0),
    ])
    if "description" in df.columns:
        df = df.with_columns(
            pl.col("description").cast(pl.Utf8).fill_null("Unknown")
                .map_elements(standardize_age, return_dtype=pl.Float64)
                .cast(pl.Float32).alias("i_target_age_years")
        ).drop("description")
    if flags.get("rich_metadata"):
        for c in ["category_l2", "category_l3", "manufacturer", "size"]:
            if c in df.columns:
                df = df.with_columns(pl.col(c).cast(pl.Utf8).fill_null("Unknown"))

    if flags.get("filter_dead"):
        active_count = df.filter(pl.col("sale_status") == 1).height
        total_count = df.height
        print(f"  [FILTER_DEAD] Keeping {active_count}/{total_count} active items")
        df = df.filter(pl.col("sale_status") == 1)
    return df


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
    return lf


def scan_tx_full(cutoff_start=None, cutoff_end=None):
    """Load with discount and bill_id columns too."""
    lf = pl.scan_parquet(TRANSACTION_PATH).with_columns([
        pl.col("customer_id").cast(pl.Int64),
        pl.col("item_id").cast(pl.Utf8),
        pl.col("quantity").cast(pl.Float32).fill_null(1.0),
        pl.col("location").cast(pl.Int32),
        pl.col("updated_date").cast(pl.Datetime).alias("event_ts"),
        pl.col("price").cast(pl.Float32).fill_null(0.0),
        pl.col("discount").cast(pl.Float32).fill_null(0.0),
        pl.col("bill_id").cast(pl.Int64),
    ]).drop("updated_date")
    if cutoff_start:
        lf = lf.filter(pl.col("event_ts") >= cutoff_start)
    if cutoff_end:
        lf = lf.filter(pl.col("event_ts") < cutoff_end)
    return lf


def scan_events(cutoff_start=None, cutoff_end=None):
    lf = pl.scan_parquet(EVENT_PATH).with_columns([
        pl.col("customer_id").cast(pl.Int64),
        pl.col("item_id").cast(pl.Utf8),
        pl.col("event_type").cast(pl.Utf8),
        pl.col("updated_date").cast(pl.Datetime).alias("event_ts"),
    ]).select(["customer_id", "item_id", "event_type", "event_ts"])
    if cutoff_start:
        lf = lf.filter(pl.col("event_ts") >= cutoff_start)
    if cutoff_end:
        lf = lf.filter(pl.col("event_ts") < cutoff_end)
    return lf


# ──────────────────────────────────────────
# PRECOMPUTE LOOKUP TABLES
# ──────────────────────────────────────────
def precompute_lookup_tables(hist_tx, items, all_users, flags, hist_tx_full=None, events=None):
    tables = {}
    max_ts = hist_tx["event_ts"].max()

    # ── GLOBAL PRE-AGGREGATION (DO ONCE) ──
    print("  [LUT] Global pre-aggregations...")
    u_tx_counts = hist_tx.group_by("customer_id").agg(pl.len().cast(pl.Float32).alias("u_total_tx"))
    ui_counts = hist_tx.group_by(["customer_id", "item_id"]).agg(pl.len().alias("ui_purchases"))
    
    u_loc_hhi = (
        hist_tx.group_by(["customer_id", "location"]).agg(pl.len().alias("loc_qty"))
        .join(u_tx_counts, on="customer_id")
        .with_columns((pl.col("loc_qty") / pl.col("u_total_tx")).alias("share"))
        .group_by("customer_id").agg([(pl.col("share") * pl.col("share")).sum().alias("u_loc_hhi")])
    )

    ui_champ = ui_counts.join(items.select(["item_id", "category_l1", "brand"]), on="item_id", how="left")
    
    tables["u_cat_affinity"] = (
        ui_champ.group_by(["customer_id", "category_l1"]).agg(pl.col("ui_purchases").sum().cast(pl.Float32).alias("u_cat_purchases"))
        .join(u_tx_counts, on="customer_id")
        .with_columns((pl.col("u_cat_purchases") / pl.col("u_total_tx")).alias("u_cat_share_of_wallet"))
        .drop("u_total_tx")
    )
    u_cat_hhi = (
        tables["u_cat_affinity"].group_by("customer_id").agg([
            (pl.col("u_cat_share_of_wallet") * pl.col("u_cat_share_of_wallet")).sum().alias("u_cat_hhi"),
            pl.col("category_l1").n_unique().alias("unique_cats")
        ])
    )

    tables["u_brand_affinity"] = (
        ui_champ.group_by(["customer_id", "brand"]).agg(pl.col("ui_purchases").sum().cast(pl.Float32).alias("u_brand_purchases"))
        .join(u_tx_counts, on="customer_id")
        .with_columns((pl.col("u_brand_purchases") / pl.col("u_total_tx")).alias("u_brand_share_of_wallet"))
        .drop("u_total_tx")
    )
    u_brand_hhi = (
        tables["u_brand_affinity"].group_by("customer_id").agg(
            [(pl.col("u_brand_share_of_wallet") * pl.col("u_brand_share_of_wallet")).sum().alias("u_brand_hhi")]
        )
    )
    del ui_champ; gc.collect()

    # ── Champion feature precomputations ──
    print("  [LUT] Champion feature precomputations...")
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
            pl.col("u_loc_hhi").fill_null(1.0), pl.col("u_cat_hhi").fill_null(1.0),
            pl.col("u_brand_hhi").fill_null(1.0), pl.col("u_recency_days").fill_null(999.0)
        ])
        .cast({c: pl.Float32 for c in [
            "u_total_tx", "u_total_qty", "u_total_spend", "u_unique_items",
            "u_tenure_days", "u_recency_days", "u_loc_hhi", "u_cat_hhi", "u_brand_hhi",
            "u_avg_basket_size", "u_avg_order_value", "u_tx_velocity",
        ]})
    )
    del u_loc_hhi, u_cat_hhi, u_brand_hhi; gc.collect()

    # ── User Archetypes ──
    print("  [LUT] user archetypes...")
    target_users = pl.DataFrame({"customer_id": all_users})
    archetype_df = tables["user_features_champ"].with_columns(
        pl.when(pl.col("u_recency_days") >= 90).then(pl.lit("Dormant"))
        .when(pl.col("u_tenure_days") <= 60).then(pl.lit("New"))
        .when((pl.col("u_cat_hhi") >= 0.7) & (pl.col("u_total_tx") >= 3)).then(pl.lit("Habitual"))
        .when(pl.col("unique_cats") >= 4).then(pl.lit("Explorer"))
        .otherwise(pl.lit("Standard")).alias("archetype")
    )
    tables["user_archetypes"] = target_users.join(archetype_df.select(["customer_id", "archetype"]), on="customer_id", how="left").with_columns(pl.col("archetype").fill_null("Dormant"))
    del target_users, archetype_df; gc.collect()

    # ── Child age estimates ──
    print("  [LUT] user child age estimates...")
    hist_items_age = hist_tx.join(items.select(["item_id", "i_target_age_years"]), on="item_id", how="left")
    tx_with_age = hist_items_age.filter(pl.col("i_target_age_years").is_not_null())
    tables["user_child_age_estimates"] = (
        tx_with_age.sort(["customer_id", "event_ts"], descending=[False, True])
        .group_by("customer_id").head(1)
        .with_columns(
            (pl.col("i_target_age_years") + (pl.lit(max_ts) - pl.col("event_ts")).dt.total_days() / 365.0).cast(pl.Float32).alias("u_child_age_estimate")
        ).select(["customer_id", "u_child_age_estimate"])
    )
    del hist_items_age, tx_with_age; gc.collect()

    # ── ui_hist ──
    print("  [LUT] ui_hist...")
    tables["ui_hist"] = (
        hist_tx.group_by(["customer_id", "item_id"]).agg([
            pl.len().cast(pl.Float32).alias("ui_purchases"),
            pl.col("event_ts").max().alias("last_purchase_ts"),
        ])
        .sort(["customer_id", "last_purchase_ts", "ui_purchases"], descending=[False, True, True])
        .with_columns(pl.int_range(1, pl.len()+1).over("customer_id").cast(pl.Int32).alias("rank"))
        .select(["customer_id", "item_id", pl.col("rank").cast(pl.Int64)])
    )

    # ── User locations & local bestsellers ──
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

    # ── Location whitelist & category habituation ──
    print("  [LUT] location assortment & category habituation...")
    tables["loc_item_whitelist"] = (
        hist_tx.select(["location", "item_id"]).unique()
        .with_columns(pl.lit(1.0).alias("ui_in_local_stock").cast(pl.Float32))
    )
    tables["cat_habitual"] = (
        ui_counts.join(items.select(["item_id", "category_l1"]), on="item_id", how="left")
        .group_by("category_l1").agg([
            pl.col("ui_purchases").sum().alias("total_purchases"),
            pl.col("ui_purchases").filter(pl.col("ui_purchases") > 1).sum().alias("repurchases")
        ])
        .with_columns((pl.col("repurchases") / pl.col("total_purchases")).cast(pl.Float32).alias("cat_habitual_score"))
        .select(["category_l1", "cat_habitual_score"])
    )

    # ── User categories & category bestsellers ──
    print("  [LUT] user categories & category bestsellers...")
    tables["user_cats"] = (
        tables["u_cat_affinity"]
        .sort(["customer_id", "u_cat_purchases"], descending=[False, True]).group_by("customer_id").head(5)
        .with_columns(pl.int_range(1, pl.len() + 1).over("customer_id").cast(pl.Int64).alias("cat_rank"))
        .select(["customer_id", "category_l1", "cat_rank"])
    )
    tables["cat_top"] = (
        ui_counts.join(items.select(["item_id", "category_l1"]), on="item_id", how="left")
        .filter(pl.col("category_l1") != "Unknown")
        .group_by(["category_l1", "item_id"]).agg(pl.col("ui_purchases").sum().alias("qty"))
        .sort(["category_l1", "qty"], descending=[False, True]).group_by("category_l1").head(80)
        .with_columns(pl.int_range(1, pl.len() + 1).over("category_l1").cast(pl.Int64).alias("item_rank"))
        .select(["category_l1", "item_id", "item_rank"])
    )

    # ── User brands & brand bestsellers ──
    print("  [LUT] user brands & brand bestsellers...")
    tables["user_brands"] = (
        tables["u_brand_affinity"].filter((pl.col("brand") != "Unknown") & (pl.col("brand") != "Không xác định"))
        .sort(["customer_id", "u_brand_purchases"], descending=[False, True]).group_by("customer_id").head(5)
        .with_columns(pl.int_range(1, pl.len() + 1).over("customer_id").cast(pl.Int64).alias("brand_rank"))
        .select(["customer_id", "brand", "brand_rank"])
    )
    tables["brand_top"] = (
        ui_counts.join(items.select(["item_id", "brand"]), on="item_id", how="left")
        .filter((pl.col("brand") != "Unknown") & (pl.col("brand") != "Không xác định"))
        .group_by(["brand", "item_id"]).agg(pl.col("ui_purchases").sum().alias("qty"))
        .sort(["brand", "qty"], descending=[False, True]).group_by("brand").head(50)
        .with_columns(pl.int_range(1, pl.len() + 1).over("brand").cast(pl.Int64).alias("item_rank"))
        .select(["brand", "item_id", "item_rank"])
    )

    # ── Global bestsellers ──
    print("  [LUT] global bestsellers...")
    tables["global_top"] = (
        hist_tx.group_by("item_id").agg(pl.len().alias("qty"))
        .sort("qty", descending=True).head(300)
        .with_columns(pl.int_range(1, pl.len() + 1).cast(pl.Int64).alias("rank"))
        .select(["item_id", "rank"])
    )

    # ── Category trending ──
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

    # ── SVD & I2I ──
    print("  [LUT] SVD & I2I...")
    tx_weighted = (
        hist_tx
        .with_columns(((max_ts - pl.col("event_ts")).dt.total_days() / 30.0).alias("months_ago"))
        .with_columns((pl.col("quantity") * pl.lit(0.70).pow(pl.col("months_ago"))).cast(pl.Float32).alias("weight"))
        .group_by(["customer_id", "item_id"]).agg(pl.col("weight").sum().alias("weight"))
    )
    u_map = tx_weighted["customer_id"].unique().to_list()
    i_map = hist_tx["item_id"].unique().to_list()
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
    svd_components = 100
    mtx_train.data = np.log1p(mtx_train.data) * 2.0
    tfidf = TfidfTransformer()
    mtx_train = tfidf.fit_transform(mtx_train).astype(np.float32)
    svd = TruncatedSVD(n_components=svd_components, algorithm='arpack', random_state=42)
    u_emb = svd.fit_transform(mtx_train).astype(np.float32)
    i_emb = svd.components_.T.astype(np.float32)
    print("  [LUT] I2I similarity...")
    norm_m = normalize(mtx_train, norm='l2', axis=0)
    i2i_sim = (norm_m.T.dot(norm_m)).tocsr()
    i2i_sim.setdiag(0)
    k_trunc = 100
    for i in range(i2i_sim.shape[0]):
        start, end = i2i_sim.indptr[i], i2i_sim.indptr[i+1]
        if end - start > k_trunc:
            row_data = i2i_sim.data[start:end]
            top_k_idx = np.argpartition(row_data, -k_trunc)[-k_trunc:]
            mask = np.ones(end - start, dtype=bool)
            mask[top_k_idx] = False
            row_data[mask] = 0
    i2i_sim.eliminate_zeros()
    del mtx_train, norm_m; gc.collect()

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
    tables["svd_i_arr"] = np.array(i_map, dtype=object)
    tables["i2i_sim"] = i2i_sim
    tables["mtx"] = mtx_all
    del u_df_train, i_df; gc.collect()

    t_30 = max_ts - pl.duration(days=30)
    t_60 = max_ts - pl.duration(days=60)
    t_90 = max_ts - pl.duration(days=90)
    i_sales_all = hist_tx.group_by("item_id").agg(pl.len().alias("i_sales_all"))
    i_sales_30d = hist_tx.filter(pl.col("event_ts") >= t_30).group_by("item_id").agg(pl.len().alias("i_sales_30d"))
    i_sales_60d = hist_tx.filter((pl.col("event_ts") >= t_60) & (pl.col("event_ts") < t_30)).group_by("item_id").agg(pl.len().alias("i_sales_60_to_30d"))
    i_sales_90d = hist_tx.filter((pl.col("event_ts") >= t_90) & (pl.col("event_ts") < t_60)).group_by("item_id").agg(pl.len().alias("i_sales_90_to_60d"))
    i_user_purchases = ui_counts.rename({"ui_purchases": "qty"})
    i_repeat_stats = i_user_purchases.group_by("item_id").agg([
        pl.col("qty").sum().alias("total_item_purchases"),
        pl.col("qty").filter(pl.col("qty") > 1).sum().alias("repeat_item_purchases")
    ]).with_columns(
        (pl.col("repeat_item_purchases") / pl.col("total_item_purchases")).alias("item_repeat_propensity")
    ).select(["item_id", "item_repeat_propensity"])
    i_launch = hist_tx.group_by("item_id").agg([(max_ts - pl.col("event_ts").min()).dt.total_days().alias("item_launch_age_days")])
    cat_sales_30d = hist_tx.filter(pl.col("event_ts") >= t_30).join(items, on="item_id", how="left").group_by("category_l1").agg(pl.len().alias("cat_sales_30d"))
    cat_sales_60d = hist_tx.filter((pl.col("event_ts") >= t_60) & (pl.col("event_ts") < t_30)).join(items, on="item_id", how="left").group_by("category_l1").agg(pl.len().alias("cat_sales_60d"))
    cat_trend_feat = cat_sales_30d.join(cat_sales_60d, on="category_l1", how="left").fill_null(0).with_columns((pl.col("cat_sales_30d") - pl.col("cat_sales_60d")).alias("cat_momentum"))

    item_feat_cols = ["item_id", "category_l1", "brand", "price"]
    if flags.get("rich_metadata"):
        item_feat_cols += [c for c in ["category_l2", "category_l3", "manufacturer", "size"] if c in items.columns]
    tables["item_features_champ"] = (
        items.select([c for c in item_feat_cols + ["sale_status"] if c in items.columns])
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
        .cast({c: pl.Float32 for c in [
            "price", "i_sales_all", "i_sales_30d", "i_sales_60_to_30d", "i_sales_90_to_60d",
            "cat_sales_30d", "cat_sales_60d", "cat_momentum", "i_momentum_30d", "i_momentum_60d",
            "item_repeat_propensity", "item_launch_age_days",
        ]})
    )
    del i_sales_all, i_sales_30d, i_sales_60d, i_sales_90d, cat_sales_30d, cat_sales_60d, cat_trend_feat; gc.collect()

    # ui_features_champ
    tables["ui_features_champ"] = (
        hist_tx.group_by(["customer_id", "item_id"]).agg([
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
        .cast({c: pl.Float32 for c in [
            "ui_purchase_count", "ui_total_qty", "ui_days_since_last", "ui_days_since_first",
            "ui_avg_qty_per_order", "ui_purchase_velocity", "ui_replenishment_due", "ui_age_delta", "ui_recency_weight",
        ]})
    )

    # Categorical maps
    cat1_unique = items["category_l1"].unique().to_list()
    brand_unique = items["brand"].unique().to_list()
    tables["cat1_map"] = {c: i for i, c in enumerate(cat1_unique)}
    tables["brand_map"] = {b: i for i, b in enumerate(brand_unique)}

    # ── NEW: Rich metadata affinities (Change 5) ──
    if flags.get("rich_metadata"):
        print("  [LUT] rich metadata affinities (cat_l2, cat_l3, manufacturer)...")
        ui_rich = ui_counts.join(items.select(["item_id", "category_l2", "category_l3", "manufacturer"]), on="item_id", how="left")
        for col in ["category_l2", "category_l3", "manufacturer"]:
            if col in ui_rich.columns:
                tables[f"u_{col}_affinity"] = (
                    ui_rich.group_by(["customer_id", col]).agg(pl.col("ui_purchases").sum().cast(pl.Float32).alias(f"u_{col}_purchases"))
                    .join(u_tx_counts, on="customer_id")
                    .with_columns((pl.col(f"u_{col}_purchases") / pl.col("u_total_tx")).alias(f"u_{col}_share"))
                    .drop("u_total_tx")
                )
                all_vals = items[col].unique().to_list()
                tables[f"{col}_map"] = {v: i for i, v in enumerate(all_vals)}
        del ui_rich; gc.collect()
    
    del u_tx_counts, ui_counts; gc.collect()

    # ── NEW: Discount features (Change 6) ──
    if flags.get("discount_features") and hist_tx_full is not None:
        print("  [LUT] discount features...")
        lazy_htf = hist_tx_full if isinstance(hist_tx_full, pl.LazyFrame) else hist_tx_full.lazy()
        # Only use last 90 days for discount behavior to keep it relevant and memory-light
        lazy_htf = lazy_htf.filter(pl.col("event_ts") >= (max_ts - pl.duration(days=90)))
        lazy_htf = lazy_htf.select(["customer_id", "item_id", "discount", "price"]).with_columns(
            (pl.col("discount") / (pl.col("price") + 1e-5)).alias("discount_rate")
        ).with_columns((pl.col("discount_rate") > 0.05).cast(pl.Float32).alias("is_promo"))
        
        tables["u_discount"] = lazy_htf.group_by("customer_id").agg([
            pl.col("discount_rate").mean().cast(pl.Float32).alias("u_avg_discount_rate"),
            pl.col("is_promo").mean().cast(pl.Float32).alias("u_promo_purchase_ratio")
        ]).collect(streaming=True)
        tables["i_discount"] = lazy_htf.group_by("item_id").agg([
            pl.col("discount_rate").mean().cast(pl.Float32).alias("i_avg_discount_rate"),
            pl.col("is_promo").mean().cast(pl.Float32).alias("i_promo_sales_ratio")
        ]).collect(streaming=True)

    # ── NEW: Event features (Change 2) ──
    if flags.get("use_events") and events is not None:
        print("  [LUT] event features (view_item + add_to_cart)...")
        lazy_ev = events if isinstance(events, pl.LazyFrame) else events.lazy()
        # Only use last 90 days of events
        lazy_ev = lazy_ev.filter(pl.col("event_ts") >= (max_ts - pl.duration(days=90)))
        lazy_ev = lazy_ev.select(["customer_id", "item_id", "event_type", "event_ts"])
        
        views = lazy_ev.filter(pl.col("event_type") == "view_item")
        atcs = lazy_ev.filter(pl.col("event_type") == "add_to_cart")
        
        tables["ui_views"] = views.group_by(["customer_id", "item_id"]).agg([
            pl.len().cast(pl.Float32).alias("ui_view_count"),
            (pl.lit(max_ts) - pl.col("event_ts").max()).dt.total_days().cast(pl.Float32).alias("ui_days_since_last_view"),
        ]).collect(streaming=True)
        tables["ui_atcs"] = atcs.group_by(["customer_id", "item_id"]).agg([
            pl.len().cast(pl.Float32).alias("ui_atc_count"),
            (pl.lit(max_ts) - pl.col("event_ts").max()).dt.total_days().cast(pl.Float32).alias("ui_days_since_last_atc"),
        ]).collect(streaming=True)
        
        tables["i_views_30d"] = views.filter(pl.col("event_ts") >= t_30).group_by("item_id").agg(
            pl.len().cast(pl.Float32).alias("i_view_count_30d")
        ).collect(streaming=True)
        tables["i_atc_30d"] = atcs.filter(pl.col("event_ts") >= t_30).group_by("item_id").agg(
            pl.len().cast(pl.Float32).alias("i_atc_count_30d")
        ).collect(streaming=True)
        
        # Event-based candidates: items viewed/ATC'd but not purchased
        purchased_pairs = hist_tx.select(["customer_id", "item_id"]).unique().lazy()
        
        # Score by frequency and recency
        event_scored = (
            lazy_ev.group_by(["customer_id", "item_id"]).agg([
                pl.len().alias("ev_count"),
                (pl.lit(max_ts) - pl.col("event_ts").max()).dt.total_days().alias("ev_recency"),
                pl.col("event_type").filter(pl.col("event_type") == "add_to_cart").len().alias("atc_count"),
            ])
            .join(purchased_pairs, on=["customer_id", "item_id"], how="anti")
            .with_columns(
                (pl.col("ev_count") + pl.col("atc_count") * 3.0).alias("ev_score")
            )
            .sort(["customer_id", "ev_score"], descending=[False, True])
            .group_by("customer_id").head(75)
            .with_columns(pl.int_range(1, pl.len() + 1).over("customer_id").cast(pl.Int64).alias("rank"))
            .select(["customer_id", "item_id", "rank"])
        ).collect(streaming=True)
        tables["event_candidates"] = event_scored

    # ── NEW: Co-purchase candidates (Change 3) ──
    if flags.get("copurchase") and hist_tx_full is not None:
        print("  [LUT] co-purchase matrix from bill_id...")
        lazy_htf = hist_tx_full if isinstance(hist_tx_full, pl.LazyFrame) else hist_tx_full.lazy()
        max_ts_cop = max_ts  # we can just use max_ts from hist_tx
        
        recent_tx = lazy_htf.filter(pl.col("event_ts") >= (max_ts_cop - pl.duration(days=30)))
        bill_items = recent_tx.select(["bill_id", "item_id"]).unique()
        
        # Only keep bills with 2-15 items (filter out single-item bills and wholesale bulk)
        bill_sizes = bill_items.group_by("bill_id").agg(pl.len().alias("n_items")).filter(
            (pl.col("n_items") >= 2) & (pl.col("n_items") <= 15)
        )
        bill_items = bill_items.join(bill_sizes.select("bill_id"), on="bill_id")
        
        # Self-join to get pairs
        pairs = bill_items.join(bill_items, on="bill_id", suffix="_r").filter(
            pl.col("item_id") < pl.col("item_id_r")
        )
        copurchase = pairs.group_by(["item_id", "item_id_r"]).agg(pl.len().alias("co_count"))
        
        # Keep top-30 co-purchased items per item to save memory
        cp_top = copurchase.sort(["item_id", "co_count"], descending=[False, True]).group_by("item_id").head(30)
        # Also reverse direction
        cp_rev = cp_top.rename({"item_id": "item_id_r", "item_id_r": "item_id"})
        
        tables["copurchase_map"] = pl.concat([
            cp_top.select(["item_id", "item_id_r", "co_count"]),
            cp_rev.select(["item_id", "item_id_r", "co_count"])
        ]).collect(streaming=True)

    print("  [LUT] All lookup tables ready.")
    return tables

# ──────────────────────────────────────────
# ──────────────────────────────────────────
# CANDIDATE GENERATION
# ──────────────────────────────────────────
def candidates_for_chunk(chunk_users, tables, flags):
    chunk_df = pl.DataFrame({"customer_id": chunk_users})

    chA = tables["ui_hist"].filter(pl.col("customer_id").is_in(chunk_users)).select(["customer_id", "item_id", "rank"]).with_columns(pl.lit("A_history").alias("channel"))
    chB = tables["user_loc"].filter(pl.col("customer_id").is_in(chunk_users)).join(tables["loc_top"], on="location", how="inner").select(["customer_id", "item_id", "rank"]).with_columns(pl.lit("B_local").alias("channel"))

    # SVD candidates
    u_idx_df = tables["u_idx_df"]; i_emb = tables["svd_i_emb"]; i_arr = tables["svd_i_arr"]; mtx = tables["mtx"]
    chunk_u_idx = pl.DataFrame({"customer_id": chunk_users}).join(u_idx_df, on="customer_id", how="inner")
    target = chunk_u_idx["customer_id"].to_numpy()
    t_idx = chunk_u_idx["u_idx"].to_numpy()
    svd_k = 1000; chC_list = []
    if len(target) > 0:
        mtx_chunk = mtx[t_idx]; u_emb_chunk = mtx_chunk.dot(i_emb)
        k = min(svd_k, i_emb.shape[0])
        scores_batch = u_emb_chunk @ i_emb.T
        top_k_b = np.argpartition(-scores_batch, kth=min(k-1, scores_batch.shape[1]-1), axis=1)[:, :k]
        order_b = np.argsort(-scores_batch[np.arange(len(target))[:, None], top_k_b], axis=1)
        top_k_b = top_k_b[np.arange(len(target))[:, None], order_b]
        chC_list.append(pl.DataFrame({
            "customer_id": pl.Series(np.repeat(target, k), dtype=pl.Int64),
            "item_id": i_arr[top_k_b.flatten()],
            "rank": pl.Series(np.tile(np.arange(1, k+1), len(target)), dtype=pl.Int64),
            "channel": pl.Series(["C_svd"] * (len(target) * k), dtype=pl.Utf8),
        }))
        del u_emb_chunk, mtx_chunk; gc.collect()
    chC = pl.concat(chC_list) if chC_list else pl.DataFrame(schema={"customer_id": pl.Int64, "item_id": pl.Utf8, "rank": pl.Int64, "channel": pl.Utf8})

    # I2I candidates
    i2i_sim = tables["i2i_sim"]; i2i_k = 1000; chD_list = []
    if len(target) > 0:
        mtx_sub = mtx[t_idx]
        scores_b = mtx_sub.dot(i2i_sim).toarray().astype(np.float32)
        k_i2i = min(i2i_k, i2i_sim.shape[1])
        top_i = np.argpartition(-scores_b, kth=k_i2i - 1, axis=1)[:, :k_i2i]
        order_i = np.argsort(-scores_b[np.arange(len(target))[:, None], top_i], axis=1)
        top_i = top_i[np.arange(len(target))[:, None], order_i]
        top_s = scores_b[np.arange(len(target))[:, None], top_i]
        mask = top_s > 0.0; valid_counts = mask.sum(axis=1)
        chD_list.append(pl.DataFrame({
            "customer_id": pl.Series(np.repeat(target, valid_counts), dtype=pl.Int64),
            "item_id": i_arr[top_i[mask]],
            "rank": pl.Series(np.tile(np.arange(1, k_i2i + 1), (len(target), 1))[mask], dtype=pl.Int64),
            "channel": ["D_i2i"] * valid_counts.sum()
        }))
        del mtx_sub; gc.collect()
    chD = pl.concat(chD_list) if chD_list else pl.DataFrame(schema={"customer_id": pl.Int64, "item_id": pl.Utf8, "rank": pl.Int64, "channel": pl.Utf8})

    chE = tables["user_cats"].filter(pl.col("customer_id").is_in(chunk_users)).join(tables["cat_top"], on="category_l1", how="inner").sort(["customer_id", "cat_rank", "item_rank"]).group_by("customer_id").head(200).with_columns(pl.int_range(1, pl.len() + 1).over("customer_id").cast(pl.Int64).alias("rank"), pl.lit("E_cat").alias("channel")).select(["customer_id", "item_id", "rank", "channel"])
    chF = tables["user_brands"].filter(pl.col("customer_id").is_in(chunk_users)).join(tables["brand_top"], on="brand", how="inner").sort(["customer_id", "brand_rank", "item_rank"]).group_by("customer_id").head(125).with_columns(pl.int_range(1, pl.len() + 1).over("customer_id").cast(pl.Int64).alias("rank"), pl.lit("F_brand").alias("channel")).select(["customer_id", "item_id", "rank", "channel"])
    chG = chunk_df.join(tables["global_top"].with_columns(pl.lit(1).alias("_k")), how="cross").drop("_k").with_columns(pl.lit("G_global").alias("channel"))
    chH = tables["user_cats"].filter(pl.col("customer_id").is_in(chunk_users)).join(tables["cat_trend"], on="category_l1", how="inner").sort(["customer_id", "cat_rank", "item_rank"]).group_by("customer_id").head(75).with_columns(pl.int_range(1, pl.len() + 1).over("customer_id").cast(pl.Int64).alias("rank"), pl.lit("H_trend").alias("channel")).select(["customer_id", "item_id", "rank", "channel"])

    all_channels = [chA, chB, chC, chD, chE, chF, chG, chH]

    # NEW: Event candidates (Change 2)
    if flags.get("use_events") and "event_candidates" in tables:
        chI = tables["event_candidates"].filter(pl.col("customer_id").is_in(chunk_users)).with_columns(pl.lit("I_event").alias("channel"))
        all_channels.append(chI)

    # NEW: Co-purchase candidates (Change 3)
    if flags.get("copurchase") and "copurchase_map" in tables:
        user_recent = tables["ui_hist"].filter(
            pl.col("customer_id").is_in(chunk_users) & (pl.col("rank") <= 10)
        ).select(["customer_id", "item_id"])
        chJ = (
            user_recent.join(tables["copurchase_map"], on="item_id", how="inner")
            .select(["customer_id", pl.col("item_id_r").alias("item_id"), pl.col("co_count")])
            .sort(["customer_id", "co_count"], descending=[False, True])
            .group_by("customer_id").head(75)
            .with_columns(
                pl.int_range(1, pl.len() + 1).over("customer_id").cast(pl.Int64).alias("rank"),
                pl.lit("J_copurchase").alias("channel")
            )
            .select(["customer_id", "item_id", "rank", "channel"])
        )
        all_channels.append(chJ)

    stacked = pl.concat(all_channels)
    del all_channels; gc.collect()

    arch_chunk = tables["user_archetypes"].filter(pl.col("customer_id").is_in(chunk_users))
    merged = stacked.join(arch_chunk.select(["customer_id", "archetype"]), on="customer_id", how="inner")

    weight_map = {
        "Habitual": {"A_history": 5.0, "B_local": 1.0, "C_svd": 0.5, "D_i2i": 0.5, "E_cat": 2.0, "F_brand": 3.0, "G_global": 0.1, "H_trend": 0.5, "I_event": 4.0, "J_copurchase": 3.0},
        "Explorer": {"A_history": 1.0, "B_local": 1.0, "C_svd": 4.0, "D_i2i": 4.0, "E_cat": 3.0, "F_brand": 1.0, "G_global": 1.0, "H_trend": 2.0, "I_event": 3.0, "J_copurchase": 2.0},
        "Dormant":  {"A_history": 3.0, "B_local": 4.0, "C_svd": 0.5, "D_i2i": 0.5, "E_cat": 1.0, "F_brand": 2.0, "G_global": 5.0, "H_trend": 1.0, "I_event": 2.0, "J_copurchase": 1.0},
        "New":      {"A_history": 0.5, "B_local": 3.0, "C_svd": 2.5, "D_i2i": 2.5, "E_cat": 3.0, "F_brand": 1.0, "G_global": 4.0, "H_trend": 4.0, "I_event": 3.0, "J_copurchase": 1.0},
        "Standard": {"A_history": 2.0, "B_local": 2.0, "C_svd": 1.5, "D_i2i": 1.5, "E_cat": 2.0, "F_brand": 2.0, "G_global": 1.0, "H_trend": 1.5, "I_event": 3.0, "J_copurchase": 2.0},
    }
    w_df = pl.DataFrame([{"archetype": a, "channel": c, "ch_weight": w} for a, cw in weight_map.items() for c, w in cw.items()])

    all_channels_in_data = merged["channel"].unique().to_list()
    rank_exprs = []
    for ch in ["A_history", "B_local", "C_svd", "D_i2i", "E_cat", "F_brand", "G_global", "H_trend", "I_event", "J_copurchase"]:
        rank_exprs.append(pl.col("rank").filter(pl.col("channel") == ch).first().alias(f"rank_{ch}"))

    scored = (
        merged.join(w_df, on=["archetype", "channel"], how="left")
        .with_columns(pl.col("ch_weight").fill_null(1.0))
        .with_columns((pl.col("ch_weight") / (pl.col("rank").cast(pl.Float32) + 60.0)).alias("score"))
        .group_by(["customer_id", "item_id", "archetype"])
        .agg([pl.col("score").sum().cast(pl.Float32).alias("final_score")] + rank_exprs)
    )
    for ch in ["A_history", "B_local", "C_svd", "D_i2i", "E_cat", "F_brand", "G_global", "H_trend", "I_event", "J_copurchase"]:
        col = f"rank_{ch}"
        if col in scored.columns:
            scored = scored.with_columns(pl.col(col).fill_null(999).cast(pl.Int32))
        else:
            scored = scored.with_columns(pl.lit(999).cast(pl.Int32).alias(col))

    scored = scored.sort(["customer_id", "final_score", "item_id"], descending=[False, True, False])
    budget_map = {"Habitual": 125, "Explorer": 400, "Dormant": 60, "New": 175, "Standard": 225}
    b_df = pl.DataFrame([{"archetype": k, "budget": v} for k, v in budget_map.items()])
    final_chunk = (
        scored.with_columns(pl.int_range(1, pl.len() + 1).over("customer_id").cast(pl.Int64).alias("final_rank"))
        .join(b_df, on="archetype", how="left").with_columns(pl.col("budget").fill_null(100))
        .filter(pl.col("final_rank") <= pl.col("budget"))
        .select([
            "customer_id", "item_id", "final_rank", "final_score",
            "rank_A_history", "rank_B_local", "rank_C_svd", "rank_D_i2i",
            "rank_E_cat", "rank_F_brand", "rank_G_global", "rank_H_trend",
            "rank_I_event", "rank_J_copurchase"
        ])
    )
    del stacked, merged, scored; gc.collect()
    return final_chunk


# ──────────────────────────────────────────
# FEATURE ASSEMBLY
# ──────────────────────────────────────────
def assemble_dataset(candidates, target_tx, tables, flags, is_inference=False):
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

    # NEW: Event features (Change 2)
    if flags.get("use_events"):
        if "ui_views" in tables:
            df = df.join(tables["ui_views"], on=["customer_id", "item_id"], how="left")
        if "ui_atcs" in tables:
            df = df.join(tables["ui_atcs"], on=["customer_id", "item_id"], how="left")
        if "i_views_30d" in tables:
            df = df.join(tables["i_views_30d"], on="item_id", how="left")
        if "i_atc_30d" in tables:
            df = df.join(tables["i_atc_30d"], on="item_id", how="left")

    # NEW: Discount features (Change 6)
    if flags.get("discount_features"):
        if "u_discount" in tables:
            df = df.join(tables["u_discount"], on="customer_id", how="left")
        if "i_discount" in tables:
            df = df.join(tables["i_discount"], on="item_id", how="left")

    # NEW: Rich metadata affinities (Change 5)
    if flags.get("rich_metadata"):
        for col in ["category_l2", "category_l3", "manufacturer"]:
            aff_key = f"u_{col}_affinity"
            if aff_key in tables and col in df.columns:
                df = df.join(tables[aff_key], on=["customer_id", col], how="left")

    # Fill nulls
    fill_cols = {
        "ui_purchase_count": 0.0, "ui_total_qty": 0.0, "ui_days_since_last": 999.0,
        "ui_days_since_first": 999.0, "ui_avg_qty_per_order": 0.0, "ui_purchase_velocity": 0.0,
        "ui_replenishment_due": 0.0, "ui_recency_weight": 0.0,
        "u_cat_purchases": 0.0, "u_cat_share_of_wallet": 0.0,
        "u_brand_purchases": 0.0, "u_brand_share_of_wallet": 0.0,
        "u_child_age_estimate": -99.0,
        # Event features
        "ui_view_count": 0.0, "ui_atc_count": 0.0,
        "ui_days_since_last_view": 999.0, "ui_days_since_last_atc": 999.0,
        "i_view_count_30d": 0.0, "i_atc_count_30d": 0.0,
        # Discount features
        "u_avg_discount_rate": 0.0, "u_promo_purchase_ratio": 0.0,
        "i_avg_discount_rate": 0.0, "i_promo_sales_ratio": 0.0,
        # Rich metadata
        "u_category_l2_purchases": 0.0, "u_category_l2_share": 0.0,
        "u_category_l3_purchases": 0.0, "u_category_l3_share": 0.0,
        "u_manufacturer_purchases": 0.0, "u_manufacturer_share": 0.0,
    }
    for col, val in fill_cols.items():
        if col in df.columns:
            df = df.with_columns(pl.col(col).fill_null(val))

    # Cross features
    cross_cols = [
        (pl.col("u_cat_share_of_wallet") * pl.col("i_momentum_30d")).alias("cross_cat_momentum"),
        (pl.col("u_brand_share_of_wallet") * pl.col("i_momentum_30d")).alias("cross_brand_momentum"),
        (pl.col("price") / (pl.col("u_avg_order_value") + 1.0)).alias("cross_price_ratio"),
        pl.when(pl.col("ui_purchase_count") > 0).then(pl.col("cat_habitual_score")).otherwise(1.0 - pl.col("cat_habitual_score")).fill_null(0.5).alias("ui_habitual_match"),
    ]
    if flags.get("discount_features") and "u_promo_purchase_ratio" in df.columns and "i_promo_sales_ratio" in df.columns:
        cross_cols.append((pl.col("u_promo_purchase_ratio") * pl.col("i_promo_sales_ratio")).alias("cross_promo_affinity"))
    df = df.with_columns(cross_cols)

    # Categorical encoding
    df = df.with_columns([
        pl.col("category_l1").replace_strict(tables["cat1_map"], default=len(tables["cat1_map"])).cast(pl.Int32).alias("category_l1_idx"),
        pl.col("brand").replace_strict(tables["brand_map"], default=len(tables["brand_map"])).cast(pl.Int32).alias("brand_idx"),
    ])
    if flags.get("rich_metadata"):
        for col in ["category_l2", "category_l3", "manufacturer"]:
            map_key = f"{col}_map"
            if map_key in tables and col in df.columns:
                df = df.with_columns(
                    pl.col(col).replace_strict(tables[map_key], default=len(tables[map_key])).cast(pl.Int32).alias(f"{col}_idx")
                )

    # SVD score
    u_idx_df = tables["u_idx_df"]; i_emb = tables["svd_i_emb"]; i_arr = tables["svd_i_arr"]; mtx_all = tables["mtx"]
    unique_users = df["customer_id"].unique().to_list()
    u_map_local = {u: idx for idx, u in enumerate(unique_users)}
    u_emb_dense = np.zeros((len(unique_users), i_emb.shape[1]), dtype=np.float32)
    unique_u_df = pl.DataFrame({"customer_id": unique_users}, schema={"customer_id": pl.Int64})
    joined = unique_u_df.join(u_idx_df, on="customer_id", how="inner")
    valid_u_ids = joined["customer_id"].to_numpy()
    valid_u_idxs = joined["u_idx"].to_numpy()
    if len(valid_u_ids) > 0:
        valid_emb = mtx_all[valid_u_idxs].dot(i_emb)
        valid_local_idxs = np.array([u_map_local[uid] for uid in valid_u_ids], dtype=np.int32)
        u_emb_dense[valid_local_idxs] = valid_emb
    del unique_u_df, joined; gc.collect()
    i_map_df = pl.DataFrame({"item_id": i_arr.tolist(), "i_idx": np.arange(len(i_arr), dtype=np.int32)}, schema={"item_id": pl.Utf8, "i_idx": pl.Int32})
    u_map_pl = pl.DataFrame({"customer_id": list(u_map_local.keys()), "u_local_idx": list(u_map_local.values())}, schema={"customer_id": pl.Int64, "u_local_idx": pl.Int32})
    df_idx = df.join(i_map_df, on="item_id", how="left").join(u_map_pl, on="customer_id", how="left")
    u_idxs = df_idx["u_local_idx"].fill_null(-1).to_numpy()
    i_idxs = df_idx["i_idx"].fill_null(-1).to_numpy()
    del df_idx, i_map_df, u_map_pl; gc.collect()
    svd_scores = np.zeros(len(df), dtype=np.float32)
    valid_mask = (u_idxs != -1) & (i_idxs != -1)
    if valid_mask.any():
        svd_scores[valid_mask] = (u_emb_dense[u_idxs[valid_mask]] * i_emb[i_idxs[valid_mask]]).sum(axis=1)
    del u_emb_dense, u_idxs, i_idxs, valid_mask; gc.collect()
    df = df.with_columns(pl.Series("svd_score", svd_scores))

    # Drop string columns
    drop_cols = ["category_l1", "brand", "location"]
    if flags.get("rich_metadata"):
        drop_cols += ["category_l2", "category_l3", "manufacturer", "size"]
    df = df.drop([c for c in drop_cols if c in df.columns])
    if "sale_status" in df.columns:
        df = df.drop("sale_status")

    if not is_inference:
        truth = (
            target_tx.with_columns([pl.col("customer_id").cast(pl.Int64), pl.col("item_id").cast(pl.Utf8), pl.lit(1).alias("label")])
            .select(["customer_id", "item_id", "label"]).unique()
        )
        df = df.join(truth, on=["customer_id", "item_id"], how="left").with_columns(pl.col("label").fill_null(0))
    return df


# ──────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────


def build_lgb_dataset_streaming(target_tx, tables, sample_n, flags, reference=None):
    '''OOM-safe Phase B training for large sample_n (100k users).
    Preallocates numpy arrays to completely avoid np.vstack memory duplication.
    '''
    sample_users = (
        target_tx["customer_id"].unique().cast(pl.Int64)
        .sample(n=min(sample_n, target_tx["customer_id"].n_unique()), seed=42).to_list()
    )
    sub_chunk = 2000
    total = (len(sample_users) + sub_chunk - 1) // sub_chunk
    
    X_mat, y_mat = None, None
    all_groups = []
    final_features = None
    current_idx = 0
    
    for i in range(0, len(sample_users), sub_chunk):
        sub = sample_users[i:i + sub_chunk]
        cands = candidates_for_chunk(sub, tables, flags)
        df    = assemble_dataset(cands, target_tx, tables, is_inference=False, flags=flags)
        del cands; gc.collect()
        
        df = df.sort("customer_id")
        
        if final_features is None:
            final_features = [c for c in df.columns if c not in ["customer_id", "item_id", "label", "channel", "rank", "final_rank"]]
            
        chunk_X = df.select(final_features).to_numpy()
        chunk_y = df["label"].to_numpy()
        chunk_groups = df.group_by("customer_id", maintain_order=True).len()["len"].to_numpy()
        
        if X_mat is None:
            # Preallocate with a generous 1050 candidates per user buffer to avoid re-allocation
            max_rows = len(sample_users) * 1050
            X_mat = np.empty((max_rows, chunk_X.shape[1]), dtype=np.float32)
            y_mat = np.empty(max_rows, dtype=np.int32)
            
        rows = chunk_X.shape[0]
        # Dynamically resize if we somehow exceed the buffer (very rare)
        if current_idx + rows > X_mat.shape[0]:
            extra_rows = max(rows, 500000)
            X_mat = np.vstack([X_mat, np.empty((extra_rows, chunk_X.shape[1]), dtype=np.float32)])
            y_mat = np.concatenate([y_mat, np.empty(extra_rows, dtype=np.int32)])
            
        X_mat[current_idx : current_idx + rows] = chunk_X
        y_mat[current_idx : current_idx + rows] = chunk_y
        all_groups.append(chunk_groups)
        
        current_idx += rows
        del df, chunk_X, chunk_y, chunk_groups; gc.collect()
        print(f"  Streamed chunk {i//sub_chunk+1}/{total} (RAM: ~safe)", end="\r")
        
    print("")
    # Slice off the unused preallocated space (creates a view, ZERO memory copy!)
    X_mat = X_mat[:current_idx]
    y_mat = y_mat[:current_idx]
    group_mat = np.concatenate(all_groups)
    del all_groups; gc.collect()
    
    ds = lgb.Dataset(X_mat, label=y_mat, group=group_mat, reference=reference, free_raw_data=False)
    ds.construct()  # Force construction early to build bin mappers
    
    return ds, final_features


items = load_items(flags)

print("Collecting Nov & Dec history for TRUE temporal validation...")
# Validation Split (Target = Dec, History = Jan-Nov)
hist_dec = scan_tx(cutoff_end=datetime(2025, 12, 1)).collect()
targ_dec = scan_tx(cutoff_start=datetime(2025, 12, 1)).collect()

# Training Split (Target = Nov, History = Jan-Oct)
hist_nov = scan_tx(cutoff_end=datetime(2025, 11, 1)).collect()
targ_nov = scan_tx(cutoff_start=datetime(2025, 11, 1), cutoff_end=datetime(2025, 12, 1)).collect()

all_valid_users = targ_dec["customer_id"].unique().cast(pl.Int64).to_list()
all_train_users = targ_nov["customer_id"].unique().cast(pl.Int64).to_list()

print("Building lookup tables for Dec history (Validation)...")
tables_dec = precompute_lookup_tables(hist_dec, items, all_valid_users, flags)
del hist_dec; gc.collect()

print("Building lookup tables for Nov history (Training)...")
tables_nov = precompute_lookup_tables(hist_nov, items, all_train_users, flags)
del hist_nov; gc.collect()


print("Sampling subsets (60k from Nov, 20k from Dec) to keep GPU tuning fast and memory-safe...")
np.random.seed(42)
train_users = np.random.choice(all_train_users, size=min(60000, len(all_train_users)), replace=False).tolist()
valid_users = np.random.choice(all_valid_users, size=min(20000, len(all_valid_users)), replace=False).tolist()

train_tx = targ_nov.filter(pl.col("customer_id").is_in(train_users))
valid_tx = targ_dec.filter(pl.col("customer_id").is_in(valid_users))

print("Building streaming training dataset (Nov Target)...")
train_data, FINAL_FEATURES = build_lgb_dataset_streaming(train_tx, tables_nov, sample_n=60000, flags=flags)

print("Building streaming validation dataset (Dec Target)...")
valid_data, _ = build_lgb_dataset_streaming(valid_tx, tables_dec, sample_n=20000, flags=flags, reference=train_data)

del targ_nov, train_tx, valid_tx; gc.collect()

print("Saving FINAL_FEATURES list to disk...")
with open(FEATURES_OUT_PATH, "wb") as f:
    pickle.dump(FINAL_FEATURES, f)

def objective(trial):
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'eval_at': 10,
        'boosting_type': 'dart',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 31, 127),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 200),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 0.9),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 0.9),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-3, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-3, 10.0, log=True),
        'device_type': 'gpu',
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[valid_data],
        valid_names=['valid'],
        num_boost_round=100
    )
    
    # In LightGBM 4.0+, best_score is directly available on the booster
    best_ndcg = model.best_score['valid']['ndcg@10']
    del model; gc.collect()
    return best_ndcg

print("Running Optuna Hyperparameter Tuning (10 trials)...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)

print(f"Best trial: {study.best_value}")
print(f"Best params: {study.best_params}")


print("Training final model with best params on ALL December data...")
# Now that we found best params using Nov->Dec validation, we retrain on Dec data!
del train_data, valid_data, tables_nov; gc.collect()

print("Rebuilding dataset using Dec Target (100k users)...")
np.random.seed(42)
final_users = np.random.choice(all_valid_users, size=min(100000, len(all_valid_users)), replace=False).tolist()
final_tx = targ_dec.filter(pl.col("customer_id").is_in(final_users))

final_lgb_train, _ = build_lgb_dataset_streaming(final_tx, tables_dec, sample_n=100000, flags=flags)
del final_tx, targ_dec; gc.collect()

best_params = study.best_params
best_params['objective'] = 'lambdarank'
best_params['metric'] = 'ndcg'
best_params['eval_at'] = 10
best_params['boosting_type'] = 'dart'
best_params['device_type'] = 'gpu'
best_params['random_state'] = 42
best_params['n_jobs'] = -1
best_params['verbose'] = -1

final_model = lgb.train(
    best_params, 
    final_lgb_train, 
    num_boost_round=300
)

final_model.save_model(MODEL_OUT_PATH)
print(f"Model saved successfully to {MODEL_OUT_PATH}!")
