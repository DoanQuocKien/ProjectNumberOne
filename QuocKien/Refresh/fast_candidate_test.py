import gc
import polars as pl
import numpy as np
from datetime import datetime
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfTransformer
import re
import time

TRANSACTION_PATH = r"d:\CS116\ProjectNumberOne\transaction_full_2025.parquet"
ITEM_PATH        = r"d:\CS116\ProjectNumberOne\items.parquet"

def standardize_age(text):
    raw_text = str(text).strip()
    clean_text = raw_text.lower()
    if re.search(r'(\*|x\d|cm)', clean_text): return None
    if re.search(r'\bb\d{2}\b', clean_text): return None
    if 's17' in clean_text: return 1.0
    if '110' in clean_text: return 5.0
    if "không xác định" in clean_text or not clean_text or clean_text == "none": return None
    diaper_map = {r'\bnb\b': 0.0, r'\bss\b': 0.0, r'\bsơ sinh\b': 0.0, r'\bs\b': 0.25, r'\bm\b': 0.6, r'\bl\b': 1.2, r'\bxl\b': 2.0, r'\bxxl\b': 3.5}
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
    return pl.scan_parquet(ITEM_PATH).select(["item_id", "category_l1", "brand", "price", "description"])\
      .with_columns([pl.col("item_id").cast(pl.Utf8), pl.col("category_l1").cast(pl.Utf8).fill_null("Unknown"), pl.col("brand").cast(pl.Utf8).fill_null("Unknown"), pl.col("price").cast(pl.Float32).fill_null(0.0), pl.col("description").cast(pl.Utf8).fill_null("Unknown")])\
      .with_columns([pl.col("description").map_elements(standardize_age, return_dtype=pl.Float64).cast(pl.Float32).alias("i_target_age_years")]).drop("description").collect()

def scan_tx(cutoff_start=None, cutoff_end=None, sample_fraction=1.0):
    lf = pl.scan_parquet(TRANSACTION_PATH).with_columns([pl.col("customer_id").cast(pl.Int64), pl.col("item_id").cast(pl.Utf8), pl.col("quantity").cast(pl.Float32).fill_null(1.0), pl.col("location").cast(pl.Int32), pl.col("updated_date").cast(pl.Datetime).alias("event_ts"), pl.col("price").cast(pl.Float32).fill_null(0.0)]).drop("updated_date")
    if cutoff_start: lf = lf.filter(pl.col("event_ts") >= cutoff_start)
    if cutoff_end: lf = lf.filter(pl.col("event_ts") < cutoff_end)
    return lf

def precompute_lookup_tables(hist_tx, items, all_users):
    tables = {}
    max_ts = hist_tx["event_ts"].max()
    target_users = pl.DataFrame({"customer_id": all_users})
    hist_items = hist_tx.join(items.select(["item_id", "category_l1", "brand", "i_target_age_years"]), on="item_id", how="left")
    
    loc_hhi = hist_tx.group_by(["customer_id", "location"]).agg(pl.len().alias("visits")).with_columns((pl.col("visits") / pl.col("visits").sum().over("customer_id")).alias("share")).group_by("customer_id").agg([(pl.col("share") * pl.col("share")).sum().alias("loc_hhi")])
    cat_hhi = hist_items.group_by(["customer_id", "category_l1"]).agg(pl.len().alias("cat_visits")).with_columns((pl.col("cat_visits") / pl.col("cat_visits").sum().over("customer_id")).alias("cat_share")).group_by("customer_id").agg([(pl.col("cat_share") * pl.col("cat_share")).sum().alias("cat_hhi"), pl.col("category_l1").n_unique().alias("unique_cats")])
    profile = hist_tx.group_by("customer_id").agg([pl.len().alias("total_tx"), pl.col("item_id").n_unique().alias("unique_items"), (pl.lit(max_ts) - pl.col("event_ts").min()).dt.total_days().alias("tenure_days"), (pl.lit(max_ts) - pl.col("event_ts").max()).dt.total_days().alias("recency_days")]).join(loc_hhi, on="customer_id", how="left").join(cat_hhi, on="customer_id", how="left").with_columns([pl.col("loc_hhi").fill_null(1.0), pl.col("cat_hhi").fill_null(1.0), pl.col("unique_cats").fill_null(1), pl.col("recency_days").fill_null(999.0)]).with_columns(pl.when(pl.col("recency_days") >= 90).then(pl.lit("Dormant")).when(pl.col("tenure_days") <= 60).then(pl.lit("New")).when((pl.col("cat_hhi") >= 0.7) & (pl.col("total_tx") >= 3)).then(pl.lit("Habitual")).when(pl.col("unique_cats") >= 4).then(pl.lit("Explorer")).otherwise(pl.lit("Standard")).alias("archetype"))
    tables["user_archetypes"] = target_users.join(profile, on="customer_id", how="left").with_columns(pl.col("archetype").fill_null("Dormant"))
    
    tables["ui_hist"] = hist_items.group_by(["customer_id", "item_id", "category_l1"]).agg([pl.len().cast(pl.Float32).alias("ui_purchases"), (pl.col("event_ts").max() - pl.col("event_ts").min()).dt.total_days().cast(pl.Float32).alias("ui_duration"), pl.col("event_ts").max().alias("last_purchase_ts")]).sort(["customer_id", "last_purchase_ts", "ui_purchases"], descending=[False, True, True]).with_columns(pl.int_range(1, pl.len()+1).over("customer_id").cast(pl.Int32).alias("rank")).select(["customer_id", "item_id", pl.col("rank").cast(pl.Int64)])
    tables["user_loc"] = hist_tx.group_by(["customer_id", "location"]).agg(pl.len().alias("v")).sort(["customer_id", "v"], descending=[False, True]).group_by("customer_id").head(1).select(["customer_id", "location"])
    tables["loc_top"] = hist_tx.filter(pl.col("event_ts") >= max_ts - pl.duration(days=60)).group_by(["location", "item_id"]).agg(pl.col("quantity").sum().alias("qty")).sort(["location", "qty"], descending=[False, True]).group_by("location").head(500).with_columns(pl.int_range(1, pl.len() + 1).over("location").cast(pl.Int64).alias("rank")).select(["location", "item_id", "rank"])
    tables["user_cats"] = hist_items.group_by(["customer_id", "category_l1"]).agg(pl.len().alias("cat_qty")).sort(["customer_id", "cat_qty"], descending=[False, True]).group_by("customer_id").head(5).with_columns(pl.int_range(1, pl.len() + 1).over("customer_id").cast(pl.Int64).alias("cat_rank")).select(["customer_id", "category_l1", "cat_rank"])
    tables["cat_top"] = hist_items.filter(pl.col("category_l1") != "Unknown").group_by(["category_l1", "item_id"]).agg(pl.len().alias("qty")).sort(["category_l1", "qty"], descending=[False, True]).group_by("category_l1").head(80).with_columns(pl.int_range(1, pl.len() + 1).over("category_l1").cast(pl.Int64).alias("item_rank")).select(["category_l1", "item_id", "item_rank"])
    tables["user_brands"] = hist_items.filter((pl.col("brand") != "Unknown") & (pl.col("brand") != "Không xác định")).group_by(["customer_id", "brand"]).agg(pl.len().alias("brand_qty")).sort(["customer_id", "brand_qty"], descending=[False, True]).group_by("customer_id").head(5).with_columns(pl.int_range(1, pl.len() + 1).over("customer_id").cast(pl.Int64).alias("brand_rank")).select(["customer_id", "brand", "brand_rank"])
    tables["brand_top"] = hist_items.filter((pl.col("brand") != "Unknown") & (pl.col("brand") != "Không xác định")).group_by(["brand", "item_id"]).agg(pl.len().alias("qty")).sort(["brand", "qty"], descending=[False, True]).group_by("brand").head(50).with_columns(pl.int_range(1, pl.len() + 1).over("brand").cast(pl.Int64).alias("item_rank")).select(["brand", "item_id", "item_rank"])
    tables["global_top"] = hist_tx.group_by("item_id").agg(pl.len().alias("qty")).sort("qty", descending=True).head(300).with_columns(pl.int_range(1, pl.len() + 1).cast(pl.Int64).alias("rank")).select(["item_id", "rank"])
    t_recent = max_ts - pl.duration(days=30); t_prior = max_ts - pl.duration(days=60)
    rs = hist_tx.filter(pl.col("event_ts") >= t_recent).group_by("item_id").agg(pl.col("quantity").sum().alias("qty_recent"))
    ps = hist_tx.filter((pl.col("event_ts") >= t_prior) & (pl.col("event_ts") < t_recent)).group_by("item_id").agg(pl.col("quantity").sum().alias("qty_prior"))
    tables["cat_trend"] = rs.join(ps, on="item_id", how="full").fill_null(0.0).with_columns((pl.col("qty_recent") - pl.col("qty_prior")).alias("momentum")).join(items.select(["item_id", "category_l1"]), on="item_id", how="left").filter(pl.col("category_l1") != "Unknown").sort(["category_l1", "momentum"], descending=[False, True]).group_by("category_l1").head(50).with_columns(pl.int_range(1, pl.len() + 1).over("category_l1").cast(pl.Int64).alias("item_rank")).select(["category_l1", "item_id", "item_rank"])
    
    tx_weighted = hist_tx.with_columns(((max_ts - pl.col("event_ts")).dt.total_days() / 30.0).alias("months_ago")).with_columns((pl.col("quantity") * pl.lit(0.70).pow(pl.col("months_ago"))).cast(pl.Float32).alias("weight")).group_by(["customer_id", "item_id"]).agg(pl.col("weight").sum().alias("weight"))
    u_map = tx_weighted["customer_id"].unique().to_list(); i_map = hist_tx["item_id"].unique().to_list()
    np.random.seed(42)
    
    svd_users = np.array(u_map)
    u_df_train = pl.DataFrame({"customer_id": svd_users, "u_idx": np.arange(len(svd_users), dtype=np.int32)})
    i_df = pl.DataFrame({"item_id": i_map, "i_idx": np.arange(len(i_map), dtype=np.int32)})
    idx_train = tx_weighted.join(u_df_train, on="customer_id").join(i_df, on="item_id")
    mtx_train = csr_matrix((idx_train["weight"].to_numpy(), (idx_train["u_idx"].to_numpy(), idx_train["i_idx"].to_numpy())), shape=(len(svd_users), len(i_map)))
    
    print(f"  [LUT] Applying TF-IDF to matrix...", flush=True)
    mtx_train.data = np.log1p(mtx_train.data) * 2.0
    tfidf = TfidfTransformer()
    mtx_train = tfidf.fit_transform(mtx_train)
    
    svd_components = 100
    print(f"  [LUT] SVD embeddings ({svd_components} components)...", flush=True)
    svd = TruncatedSVD(n_components=svd_components, algorithm='randomized', random_state=42)
    u_emb = svd.fit_transform(mtx_train).astype(np.float32)
    i_emb = svd.components_.T.astype(np.float32)
    norm_m = normalize(mtx_train, norm='l2', axis=0)
    i2i_sim = (norm_m.T.dot(norm_m))
    i2i_sim = i2i_sim.tocsr()
    i2i_sim.setdiag(0)
    i2i_sim.eliminate_zeros()
    u_df_all = pl.DataFrame({"customer_id": u_map, "u_idx": np.arange(len(u_map), dtype=np.int32)})
    idx_all = tx_weighted.join(u_df_all, on="customer_id").join(i_df, on="item_id")
    mtx_all = csr_matrix((idx_all["weight"].to_numpy(), (idx_all["u_idx"].to_numpy(), idx_all["i_idx"].to_numpy())), shape=(len(u_map), len(i_map)))
    mtx_all.data = np.log1p(mtx_all.data) * 2.0
    tables["svd_i_emb"] = i_emb; tables["u_idx_df"] = u_df_all; tables["svd_i_arr"] = np.array(i_map, dtype=object); tables["i2i_sim"] = i2i_sim; tables["mtx"] = mtx_all
    return tables

def candidates_for_chunk(chunk_users, tables):
    chunk_df = pl.DataFrame({"customer_id": chunk_users})
    chA = tables["ui_hist"].filter(pl.col("customer_id").is_in(chunk_users)).select(["customer_id", "item_id", "rank"]).with_columns(pl.lit("A_history").alias("channel"))
    chB = tables["user_loc"].filter(pl.col("customer_id").is_in(chunk_users)).join(tables["loc_top"], on="location", how="inner").select(["customer_id", "item_id", "rank"]).with_columns(pl.lit("B_local").alias("channel"))
    
    u_idx_df = tables["u_idx_df"]
    i_emb = tables["svd_i_emb"]
    i_arr = tables["svd_i_arr"]
    mtx = tables["mtx"]
    chunk_u_idx = chunk_df.join(u_idx_df, on="customer_id", how="inner")
    target = chunk_u_idx["customer_id"].to_numpy()
    t_idx = chunk_u_idx["u_idx"].to_numpy()
    
    svd_k = 1000
    chC_list = []
    if len(target) > 0:
        mtx_chunk = mtx[t_idx]
        u_emb_chunk = mtx_chunk.dot(i_emb)
        all_top_k_users, all_top_k_items, all_top_k_ranks = [], [], []
        k = min(svd_k, i_emb.shape[0])
        SVD_BATCH = 2000
        for bi in range(0, len(target), SVD_BATCH):
            u_batch = u_emb_chunk[bi:bi+SVD_BATCH]
            scores_batch = u_batch @ i_emb.T
            top_k_b = np.argpartition(-scores_batch, kth=min(k-1, scores_batch.shape[1]-1), axis=1)[:, :k]
            order_b = np.argsort(-scores_batch[np.arange(len(u_batch))[:, None], top_k_b], axis=1)
            top_k_b = top_k_b[np.arange(len(u_batch))[:, None], order_b]
            all_top_k_users.append(np.repeat(target[bi:bi+SVD_BATCH], k))
            all_top_k_items.append(i_arr[top_k_b.flatten()])
            all_top_k_ranks.append(np.tile(np.arange(1, k+1), len(u_batch)))
        chC = pl.DataFrame({"customer_id": pl.Series(np.concatenate(all_top_k_users), dtype=pl.Int64), "item_id": np.concatenate(all_top_k_items), "rank": pl.Series(np.concatenate(all_top_k_ranks), dtype=pl.Int64), "channel": pl.Series(["C_svd"] * sum(len(x) for x in all_top_k_users), dtype=pl.Utf8)})
    else:
        chC = pl.concat(chC_list) if len(chC_list) > 0 else pl.DataFrame(schema={"customer_id": pl.Int64, "item_id": pl.Utf8, "rank": pl.Int64, "channel": pl.Utf8})
        
    i2i_sim = tables["i2i_sim"]
    i2i_k = 1000
    chD_list = []
    d2i_users, d2i_items, d2i_ranks = [], [], []
    if len(target) > 0:
        I2I_BATCH = 1000
        mtx_sub = mtx[t_idx]
        n_items  = i2i_sim.shape[1]
        k_i2i    = min(i2i_k, n_items)
        for bi in range(0, len(target), I2I_BATCH):
            batch_mtx = mtx_sub[bi:bi + I2I_BATCH]
            scores_b  = batch_mtx.dot(i2i_sim).toarray().astype(np.float32)
            batch_tgt = target[bi:bi + I2I_BATCH]
            n_b = len(batch_tgt)
            top_i  = np.argpartition(-scores_b, kth=k_i2i - 1, axis=1)[:, :k_i2i]
            order_i = np.argsort(-scores_b[np.arange(n_b)[:, None], top_i], axis=1)
            top_i   = top_i[np.arange(n_b)[:, None], order_i]
            top_s   = scores_b[np.arange(n_b)[:, None], top_i]
            mask = top_s > 0.0
            valid_counts = mask.sum(axis=1)
            d2i_users.append(np.repeat(batch_tgt, valid_counts))
            d2i_items.append(i_arr[top_i[mask]])
            rank_mtx = np.tile(np.arange(1, k_i2i + 1), (n_b, 1))
            d2i_ranks.append(rank_mtx[mask])
    if len(d2i_users) > 0:
        chD = pl.DataFrame({"customer_id": pl.Series(np.concatenate(d2i_users), dtype=pl.Int64), "item_id": pl.Series(np.concatenate(d2i_items), dtype=pl.Utf8), "rank": pl.Series(np.concatenate(d2i_ranks), dtype=pl.Int64), "channel": pl.Series(["D_i2i"] * sum(len(x) for x in d2i_users), dtype=pl.Utf8)})
    else:
        chD = pl.concat(chD_list) if len(chD_list) > 0 else pl.DataFrame(schema={"customer_id": pl.Int64, "item_id": pl.Utf8, "rank": pl.Int64, "channel": pl.Utf8})
        
    chE = tables["user_cats"].filter(pl.col("customer_id").is_in(chunk_users)).join(tables["cat_top"], on="category_l1", how="inner").sort(["customer_id", "cat_rank", "item_rank"]).group_by("customer_id").head(400).with_columns(pl.int_range(1, pl.len() + 1).over("customer_id").cast(pl.Int64).alias("rank"), pl.lit("E_cat").alias("channel")).select(["customer_id", "item_id", "rank", "channel"])
    chF = tables["user_brands"].filter(pl.col("customer_id").is_in(chunk_users)).join(tables["brand_top"], on="brand", how="inner").sort(["customer_id", "brand_rank", "item_rank"]).group_by("customer_id").head(250).with_columns(pl.int_range(1, pl.len() + 1).over("customer_id").cast(pl.Int64).alias("rank"), pl.lit("F_brand").alias("channel")).select(["customer_id", "item_id", "rank", "channel"])
    chG = chunk_df.join(tables["global_top"].with_columns(pl.lit(1).alias("_k")), how="cross").drop("_k").with_columns(pl.lit("G_global").alias("channel"))
    chH = tables["user_cats"].filter(pl.col("customer_id").is_in(chunk_users)).join(tables["cat_trend"], on="category_l1", how="inner").sort(["customer_id", "cat_rank", "item_rank"]).group_by("customer_id").head(150).with_columns(pl.int_range(1, pl.len() + 1).over("customer_id").cast(pl.Int64).alias("rank"), pl.lit("H_trend").alias("channel")).select(["customer_id", "item_id", "rank", "channel"])
    
    stacked = pl.concat([chA, chB, chC, chD, chE, chF, chG, chH])
    arch_chunk = tables["user_archetypes"].filter(pl.col("customer_id").is_in(chunk_users))
    merged = stacked.join(arch_chunk.select(["customer_id", "archetype"]), on="customer_id", how="inner")
    
    weight_map = {
        "Habitual": {"A_history": 5.0, "B_local": 1.0, "C_svd": 0.5, "D_i2i": 0.5, "E_cat": 2.0, "F_brand": 3.0, "G_global": 0.1, "H_trend": 0.5},
        "Explorer": {"A_history": 1.0, "B_local": 1.0, "C_svd": 4.0, "D_i2i": 4.0, "E_cat": 3.0, "F_brand": 1.0, "G_global": 1.0, "H_trend": 2.0},
        "Dormant":  {"A_history": 3.0, "B_local": 4.0, "C_svd": 0.5, "D_i2i": 0.5, "E_cat": 1.0, "F_brand": 2.0, "G_global": 5.0, "H_trend": 1.0},
        "New":      {"A_history": 0.5, "B_local": 3.0, "C_svd": 2.5, "D_i2i": 2.5, "E_cat": 3.0, "F_brand": 1.0, "G_global": 4.0, "H_trend": 4.0},
        "Standard": {"A_history": 2.0, "B_local": 2.0, "C_svd": 1.5, "D_i2i": 1.5, "E_cat": 2.0, "F_brand": 2.0, "G_global": 1.0, "H_trend": 1.5},
    }
    w_df = pl.DataFrame([{"archetype": a, "channel": c, "ch_weight": w} for a, cw in weight_map.items() for c, w in cw.items()])
    
    scored = merged.join(w_df, on=["archetype", "channel"], how="left").with_columns(pl.col("ch_weight").fill_null(1.0)).with_columns((pl.col("ch_weight") / (pl.col("rank").cast(pl.Float32) + 60.0)).alias("score")).group_by(["customer_id", "item_id", "archetype"]).agg(pl.col("score").sum().cast(pl.Float32).alias("final_score")).sort(["customer_id", "final_score"], descending=[False, True])
    
    # Fast Budgets
    budget_map = {
        "Habitual": 125,
        "Explorer": 400,
        "Dormant": 60,
        "New": 175,
        "Standard": 225
    }
    b_df = pl.DataFrame([{"archetype": k, "budget": v} for k, v in budget_map.items()])
    
    final_chunk = scored.with_columns(pl.int_range(1, pl.len() + 1).over("customer_id").cast(pl.Int64).alias("final_rank")).join(b_df, on="archetype", how="left").with_columns(pl.col("budget").fill_null(100)).filter(pl.col("final_rank") <= pl.col("budget")).select(["customer_id", "item_id", "final_rank"])
    return final_chunk

def test_recall(num_samples=1000, fraction=1.0):
    print("Loading data...")
    t0 = time.time()
    items = load_items()
    # Read entire history but allow sampling to make it incredibly fast for local test
    hist_dec = scan_tx(cutoff_end=datetime(2025, 12, 1)).collect()
    
    # To reduce RAM/compute if requested, just sample the history completely
    if fraction < 1.0:
        hist_dec = hist_dec.sample(fraction=fraction, seed=42)
    
    targ_dec = scan_tx(cutoff_start=datetime(2025, 12, 1)).collect()
    
    # Sample Target users for testing effectiveness
    all_dec_users = targ_dec["customer_id"].unique().cast(pl.Int64).to_list()
    sample_users = pl.Series(all_dec_users).sample(n=min(num_samples, len(all_dec_users)), seed=42).to_list()
    
    print(f"Testing on {len(sample_users)} sampled users.")
    print(f"Data loading took {time.time() - t0:.1f}s")
    
    print("Building lookup tables (this may take ~30s on full data)...")
    t1 = time.time()
    tables = precompute_lookup_tables(hist_dec, items, sample_users)
    print(f"Table building took {time.time() - t1:.1f}s")
    
    print("Generating candidates (Fast)...")
    t2 = time.time()
    cands = candidates_for_chunk(sample_users, tables)
    print(f"Generation took {time.time() - t2:.1f}s")
    
    print("Computing Recall...")
    truth = targ_dec.filter(pl.col("customer_id").is_in(sample_users)).select(["customer_id", "item_id"]).unique().with_columns(pl.lit(1).alias("label"))
    merged = truth.join(cands, on=["customer_id", "item_id"], how="left")
    
    total_items = len(merged)
    found_items = merged.filter(pl.col("final_rank").is_not_null()).height
    item_recall = found_items / total_items
    
    total_users = truth["customer_id"].n_unique()
    users_with_hits = merged.filter(pl.col("final_rank").is_not_null())["customer_id"].n_unique()
    user_recall = users_with_hits / total_users
    
    print(f"\n--- RESULTS ---")
    print(f"Users Evaluated: {len(sample_users)}")
    print(f"Retrieval Recall (User-level): {user_recall:.4f}")
    print(f"Retrieval Recall (Item-level): {item_recall:.4f}")
    print(f"Total Time: {time.time() - t0:.1f}s")

if __name__ == "__main__":
    # Test on 500 users for an extremely fast local test. 
    # (Pass fraction=0.1 to sample history too if you just want to test code execution instantly)
    test_recall(num_samples=500, fraction=1.0)
