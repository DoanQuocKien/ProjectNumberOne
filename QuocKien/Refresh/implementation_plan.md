# PIR System Overhaul: 9.24% → 15%+ MAP

## Current Score: **9.24% MAP** (0.03% improvement from baseline)

After a deep audit of the entire pipeline, I've identified **7 critical weaknesses** that are each individually costing us 1-3% MAP. Fixing them together should compound to a major breakthrough.

---

## Diagnosis: What's Wrong

### The Data We're Ignoring

| Data Source | Status | Impact |
|---|---|---|
| **Event data** (30.3M rows: 23.6M `view_item` + 6.7M `add_to_cart`) | ❌ **Completely unused** | HIGH — direct purchase intent signals |
| Transaction `discount` column | ❌ Unused | MEDIUM — promo sensitivity features |
| Transaction `bill_id` column | ❌ Unused | MEDIUM — co-purchase (basket) signals |
| Item `category_l2` (136 unique) | ❌ Unused | MEDIUM — finer-grained category matching |
| Item `category_l3` (479 unique) | ❌ Unused | MEDIUM — very specific product type |
| Item `manufacturer` (836 unique) | ❌ Unused | LOW-MEDIUM |
| Item `sale_status` (0=inactive, 1=active) | ❌ Unused | HIGH — we're recommending dead items |
| Item `size` (97 unique) | ❌ Unused | LOW |

### The Architecture Gaps

| Gap | Details | Impact |
|---|---|---|
| **No co-purchase (basket) candidates** | `bill_id` enables "bought-together" mining — the #1 technique in winning recsys solutions | HIGH |
| **No event-based candidates** | view→purchase and ATC→purchase are THE strongest intent signals | HIGH |
| **Recommending discontinued items** | 22,973 items have `sale_status=0` — we're wasting 3-4 of our 10 slots on items that can't be bought | CRITICAL |
| **Training protocol broken** | Phase B trains on Dec target but infers for Jan — we should retrain on ALL data (as the competition protocol Step 4 specifies) | HIGH |
| **SVD trained on 50k random users** | Out of 3M users, only 1.7% inform the latent space | MEDIUM |
| **No Item2Vec / Word2Vec embeddings** | Sequential co-occurrence patterns completely missed | MEDIUM |
| **Candidate recall ceiling = 68.8%** | Even with perfect ranking, we can never hit items not in candidate set | FUNDAMENTAL |

---

## Proposed Changes

> [!IMPORTANT]
> The changes are ordered by estimated impact. Each change is independent and can be tested incrementally.

---

### Change 1: Filter Out Dead Items (sale_status=0) ⚡ Quick Win

#### [MODIFY] [build_notebooks.py](file:///d:/CS116/ProjectNumberOne/QuocKien/Refresh/build_notebooks.py)

**What**: After loading items, filter to `sale_status == 1` (6,850 active items). Apply this filter to ALL candidate channels and the I2I/SVD matrices.

**Why**: Currently we have ~29,823 items in the candidate pool. 77% of them are discontinued (`sale_status=0`). The model wastes enormous capacity ranking items that physically cannot be purchased in January 2026. This single change should immediately boost precision by removing noise from the top-10.

**Risk**: LOW — purely additive quality filter.

**Estimated impact**: +0.5-1.5% MAP

---

### Change 2: Add Event Data (view_item + add_to_cart) as Candidate Channel + Features

#### [MODIFY] [build_notebooks.py](file:///d:/CS116/ProjectNumberOne/QuocKien/Refresh/build_notebooks.py)

**What**:
1. **New candidate channel `I_events`**: For each user, retrieve items they viewed or added-to-cart but never purchased. These are the highest-intent candidates we're currently missing.
2. **New features**:
   - `ui_view_count`: How many times user viewed this item
   - `ui_atc_count`: How many times user added this item to cart
   - `ui_days_since_last_view`: Recency of view
   - `ui_days_since_last_atc`: Recency of ATC
   - `ui_view_to_purchase_ratio`: User's general conversion rate
   - `i_view_count_30d`: Item's recent view popularity
   - `i_atc_to_view_ratio`: Item-level conversion rate (higher = more compelling)

**Why**: 30.3 million intent signals are sitting untouched. In the OTTO competition (most similar Kaggle challenge), co-visitation matrices from event data were THE dominant technique used by ALL top-10 teams. A user who views an item 5 times but hasn't bought it is FAR more likely to buy it next month than a random category bestseller.

**Data**: 729,097 unique customers in event data × overlap with 3M transaction customers to check.

**Risk**: MEDIUM — needs careful memory management on Kaggle. The event file is 160MB on disk (manageable).

**Estimated impact**: +1.5-3.0% MAP

---

### Change 3: Add Co-Purchase (Basket) Candidates via bill_id

#### [MODIFY] [build_notebooks.py](file:///d:/CS116/ProjectNumberOne/QuocKien/Refresh/build_notebooks.py)

**What**:
1. Build a **co-purchase matrix**: For every pair of items that appeared in the same `bill_id`, count co-occurrence frequency.
2. **New candidate channel `J_copurchase`**: For each user's recently purchased items, recommend the top-K co-purchased items they haven't bought yet.
3. **New feature**: `copurchase_score` — aggregate co-purchase strength between user's history and this candidate.

**Why**: This is the #1 winning technique from OTTO and H&M competitions (Chris Deotte's approach). When a parent buys diapers (`Tã`), they almost certainly also need baby wipes (`Vệ sinh`), formula (`Sữa`), and baby food (`Thực phẩm cho bé`). The `bill_id` column directly encodes these purchase patterns but we've never used it.

**Risk**: MEDIUM — co-purchase matrix can be large. Use sparse representation and top-K truncation like we do for I2I.

**Estimated impact**: +1.0-2.5% MAP

---

### Change 4: Fix Training Protocol (Retrain on ALL Data)

#### [MODIFY] [build_notebooks.py](file:///d:/CS116/ProjectNumberOne/QuocKien/Refresh/build_notebooks.py)

**What**: Currently:
- Phase A: Train on Oct history → Nov target (feature selection)
- Phase B: Train on Nov history → Dec target (final model)
- Inference: Use Dec model to predict Jan

The competition protocol (Step 4) explicitly says: *"Retrain on ALL known data (train+val+test)"*.

**New protocol**:
- Phase A: Train Oct → Nov (feature selection) — **keep unchanged**
- Phase B: Train on **Nov history → Dec target** (hyperparameter tuning + validation) — **keep unchanged**
- **Phase C (NEW)**: Retrain on **ALL data through Dec** → use Dec purchases as additional training signal. The model for Jan inference should have seen Dec patterns.
- Inference: Use Phase C model to predict Jan

**Why**: Currently the model has NEVER seen December purchasing patterns. December is the most recent month and the strongest signal for January behavior. The current approach throws away the most valuable training data.

**Risk**: LOW — standard practice in time-series competitions.

**Estimated impact**: +0.5-1.0% MAP

---

### Change 5: Richer Item Metadata Features

#### [MODIFY] [build_notebooks.py](file:///d:/CS116/ProjectNumberOne/QuocKien/Refresh/build_notebooks.py)

**What**: Load and use the full items schema:
- `category_l2` (136 values) → `category_l2_idx` feature + user-category_l2 affinity
- `category_l3` (479 values) → `category_l3_idx` feature + user-category_l3 affinity
- `manufacturer` (836 values) → `manufacturer_idx` feature + user-manufacturer affinity
- `sale_status` → `i_is_active` feature
- `size` → `size_idx` feature (e.g., diaper sizes correlate strongly with child age)

**Why**: Currently we only use `category_l1` (15 categories) and `brand`. But `category_l2` has 136 distinct values — that's 9x more granular matching. A user who buys "Sữa bột cho bé 0-6 tháng" (category_l3) is unlikely to want "Sữa bột cho trẻ 3-6 tuổi". The current model treats all "Sữa" as identical.

**Risk**: LOW — purely additive features. LightGBM will prune if not useful.

**Estimated impact**: +0.5-1.0% MAP

---

### Change 6: Discount / Promotion Features

#### [MODIFY] [build_notebooks.py](file:///d:/CS116/ProjectNumberOne/QuocKien/Refresh/build_notebooks.py)

**What**: Use the `discount` column from transactions:
- `u_avg_discount_rate`: User's average discount rate (promo hunter vs. full-price buyer)
- `u_promo_purchase_ratio`: What fraction of user's purchases had a discount
- `i_avg_discount_rate`: Item's typical discount (heavily promoted items are different)
- `i_promo_sales_ratio`: What fraction of item's sales were discounted
- `cross_promo_affinity`: u_promo × i_promo interaction

**Why**: These features already existed in your earlier `new_pir_lgbm_v3.py` pipeline but were dropped when the code was refactored. They help the model understand whether a user is a bargain hunter and whether an item is typically bought on promotion.

**Risk**: LOW — features from your own proven earlier pipeline.

**Estimated impact**: +0.3-0.7% MAP

---

### Change 7: Expand SVD Training Base & Add Item2Vec

#### [MODIFY] [build_notebooks.py](file:///d:/CS116/ProjectNumberOne/QuocKien/Refresh/build_notebooks.py)

**What**:
1. **Train SVD on ALL users** (not just 50k random sample). Use the full sparse matrix. The `arpack` solver handles this since we only need 100 components.
2. **Item2Vec embeddings**: Train Word2Vec on "purchase sequences" (items ordered by timestamp per user). This captures sequential co-occurrence patterns that SVD misses entirely.
   - Feed Item2Vec similarity scores as an additional feature: `item2vec_score`
   - Use as an additional candidate channel: `K_item2vec`

**Why**: SVD captures linear patterns in the user-item matrix. Item2Vec captures which items tend to appear in similar temporal contexts (e.g., "after buying diapers size M, users tend to buy diapers size L within 3 months"). These are complementary signals.

**Memory note**: Word2Vec from `gensim` is very lightweight. We can train it on sequences of item_ids with window=5, dim=64.

**Risk**: MEDIUM — gensim may not be pre-installed on Kaggle (needs `pip install gensim` in notebook). Alternative: implement skip-gram manually with numpy.

**Estimated impact**: +0.5-1.5% MAP

---

## Open Questions

> [!IMPORTANT]
> **Q1: Kaggle environment constraints.** Does your Kaggle notebook have internet access during runtime? If yes, we can `pip install gensim` for Item2Vec. If not, we need a pure numpy implementation.

> [!IMPORTANT]
> **Q2: Which changes to prioritize?** I recommend implementing in this order for maximum impact:
> 1. Change 1 (filter dead items) — 10 min
> 2. Change 4 (fix training protocol) — 30 min
> 3. Change 2 (event data) — 2 hours
> 4. Change 3 (co-purchase) — 1.5 hours
> 5. Change 5 (item metadata) — 45 min
> 6. Change 6 (discount features) — 30 min
> 7. Change 7 (Item2Vec) — 2 hours
>
> Should I implement all 7, or do you want to start with a subset and test?

> [!WARNING]
> **Q3: Memory budget.** The event data adds ~160MB of raw data. With all changes combined, we need to be very careful about Kaggle's 30GB RAM limit. I'll use the same aggressive memory management (lazy scans, chunk processing, gc.collect) that we've been doing. Should I target the Fast or Full notebook version?

---

## Verification Plan

### Automated Tests (Local Dec Evaluation)
Run local evaluation on December target (train on ≤Nov, predict Dec, compare to actual Dec purchases):
```bash
conda run -n science_env python local_eval.py
```
This gives us MAP@10, Precision@10, MRR before uploading to Kaggle.

### Kaggle Submission
After local validation shows improvement, generate the notebook and submit to Kaggle for the blind Jan 2026 test.

### Ablation Testing
Test each change individually to measure its contribution:
1. Baseline (current): 9.24%
2. + Filter dead items: ?%
3. + Event data: ?%
4. + Co-purchase: ?%
5. + Full retrain: ?%
6. All combined: ?%
