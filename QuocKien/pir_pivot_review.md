# PIR Pivot Review

Date: 2026-05-18

## Bottom line

The v12 system is a strong piece of engineering, but the current strategy has reached feature-engineering saturation and the reported `0.1998827542` December `Precision@10` is not a fully honest production estimate.

The next pivot should not be "EDA idea 51". It should be:

1. Fix the evaluation protocol so test candidates are never seeded with ground truth.
2. Optimize candidate recall and event-aware retrieval.
3. Add source-aware ranking and segment/slot reranking.
4. Only then continue feature engineering.

## What I reviewed

- `Personalized Item Recommendation (PIR)__-....md`
- `QuocKien/pir_pipeline_v12.ipynb`
- `QuocKien/v12_technical_architecture.md`
- `QuocKien/v11_technical_architecture.md`
- `QuocKien/recsys_v12_framework.md`
- `QuocKien/data_analysis/analytical_insights_master.md`
- `QuocKien/data_analysis/data_exploration_ideas.md`
- Scratch recall/baseline scripts under `QuocKien/scratch`
- Dataset schemas and several aggregate diagnostics over `transaction_full_2025.parquet`, `event_full_2025.parquet`, and `items.parquet`

## Result review

Your v12 score:

| Version | Hits | Precision@10 | MRR |
|---|---:|---:|---:|
| v10 | 18,320 | 0.195268 | 0.638260 |
| v11 | 18,672 | 0.199019 | 0.650454 |
| v12 | 18,753 | 0.199883 | 0.652929 |

The v11 to v12 lift is only `+81` hits and `+0.000864` Precision@10. That is a very small gain after adding several sophisticated features. This usually means the ranker is not the bottleneck anymore.

The larger issue: v7 onward appends validation/test positives into the candidate set:

```python
missed = truth.join(ds, on=['customer_id', 'item_id'], how='anti')
ds = pl.concat([ds, missed]).unique(subset=['customer_id', 'item_id'])
```

This is acceptable for ranker training if used carefully, but it is not acceptable for validation/test candidate generation. In v12, December evaluation uses `create_dataset_v12(... truth_df=month_12 ...)`, so every true December purchase missing from retrieval gets added before scoring. That makes the reported score a ranking-on-oracle-candidates metric, not an end-to-end recommendation metric.

There is also target-user sampling bias. v12 samples `40,000` users from history users, then evaluates only the sampled users who purchased in December. That denominator is `9,382` users, not all December buyers.

Full Month 12 reality:

| Month 12 group | Count |
|---|---:|
| Purchase users | 862,958 |
| Users with purchase history before Month 12 | 658,853 |
| Cold-start purchase users | 204,105 |
| Unique customer-item truth pairs | 3,432,162 |

So about `23.65%` of December buyers are cold-start by purchase history. The current v12 evaluation excludes that hard part.

## Strategy review

The 50-EDA strategy was valuable for finding signals. It gave you replenishment, local assortment, brand/category concentration, size/age proxy, momentum, and repeat-vs-one-off behavior. But as an optimization strategy it is now too indirect.

The model is now limited by these bottlenecks:

1. Candidate recall, not more tabular features.
2. Evaluation leakage, which hides the real retriever gap.
3. Missing event data, even though the task explicitly provides it.
4. Lack of source/rank features. The ranker knows item/user features, but not enough about why a candidate was retrieved or where it ranked in each retrieval channel.
5. Cold-start handling is under-tested.

Existing recall audit from `check_v10_recall.py` on 10,000 active December users:

| Source | Pair recall |
|---|---:|
| History | 0.2699 |
| Global top candidates | 0.3111 |
| SVD | 0.2523 |
| I2I | 0.2349 |
| Category top | 0.0449 |
| Combined retriever | 0.5050 |

That means roughly half of true purchased pairs are not even available to rank in that audit. If the item is not in candidates, the ranker cannot recover it.

Repeat signal is real but not enough alone:

- Month 12 repeat customer-item pairs: `927,585 / 3,432,162 = 27.03%`
- Recent-history top-10 repeat baseline on warm December users: about `0.09327` Precision@10

Global fallback is necessary for cold-start but weak by itself:

- Global top-10 from last 14 days before December on all December buyers: about `0.01690` Precision@10

## Event-data opportunity

v12 does not use `event_full_2025.parquet`. This is the most obvious pivot.

November events predicting December same user-item purchases:

| Window | Event type | Event pairs | Hit pairs | Approx conversion |
|---|---:|---:|---:|---:|
| November | view_item | 1,468,354 | 83,867 | 5.7% |
| November | add_to_cart | 443,664 | 65,180 | 14.7% |
| November | any event | 1,588,885 | 98,058 | 6.2% |
| Last 7 days of November | add_to_cart | 115,190 | 23,743 | 20.6% |
| Last 7 days of November | any event | 430,715 | 36,967 | 8.6% |

`add_to_cart` is a high-precision candidate source. Recent views are lower precision but still useful as candidates and ranking features.

## Code issues to fix

1. Split candidate generation modes:
   - `train`: may append missing positives for ranker learning.
   - `eval` and `inference`: must never append truth.

2. Evaluation target users should come from `truth_df['customer_id'].unique()` for internal validation, not `history_df`.

3. Report full, warm, and cold-start metrics separately:
   - Full Month 12 users.
   - Warm users with prior purchases.
   - Cold-start users without prior purchases.

4. Add candidate recall metrics before ranking:
   - `Recall@candidate_pool`
   - average candidates per user
   - source-level hit counts
   - missed-positive count

5. Preserve retrieval source metadata:
   - `src_history`, `src_replenishment`, `src_global`, `src_local`, `src_svd`, `src_i2i`, `src_category`, `src_event_view`, `src_event_atc`
   - per-source rank and raw score where available

6. Fix v12 feature mismatches:
   - `ui_size_age_diff` is documented as absolute difference, but code uses signed difference.
   - `ui_loc_sparsity_penalty` is computed before null `ui_loc_sales` is filled, so missing local sales may not trigger the intended zero-sales penalty.
   - Final model trains `1200` rounds even though Optuna trials often early-stop around `1-55` rounds. Use best iteration or a fold-derived round count.

## Recommended pivot

### Phase 0: Make the leaderboard honest

Build `pir_pipeline_v13_eval_clean.ipynb` or a `.py` runner that changes only evaluation first.

Required outputs for every fold:

```text
month=12
target_users=862958
warm_users=658853
cold_users=204105
avg_candidates_per_user=...
candidate_recall=...
precision@10_full=...
precision@10_warm=...
precision@10_cold=...
```

Do not optimize until this exists. Otherwise every future change will be hard to trust.

### Phase 1: Event-aware candidate generation

Add these retrieval channels:

- Recent same-item `add_to_cart`, last 1/7/14/30 days.
- Recent same-item `view_item`, last 1/7/14/30 days.
- Event co-visitation I2I: items viewed/ATCed by similar users in a short window.
- Event-to-purchase I2I: event item -> later purchased item, especially same category and same brand.
- Event-only user fallback, because users may have behavior but no purchases.

Add features:

- `ev_view_cnt_7d`, `ev_view_cnt_30d`
- `ev_atc_cnt_7d`, `ev_atc_cnt_30d`
- `ev_days_since_last_view`, `ev_days_since_last_atc`
- `ev_last_type_weight`
- `ev_atc_to_view_ratio`
- `ev_candidate_rank`

For the final January model, train on transactions/events through December and infer from all known December history/events.

### Phase 2: Segment-aware slot reranking

Instead of making LambdaRank solve every business regime in one score, reserve slots by confidence:

- Warm habitual/replenishment users: first 2-4 slots from due repeats and high-repeat consumables.
- Recent ATC users: first 1-3 slots from recent cart items unless already purchased afterward.
- Explorers/whales: allocate more slots to event, SVD/I2I, and trending items.
- Cold-start/no-history users: local/global/category popularity, and event candidates if available.

Then let LambdaRank order within or after those slot pools. Tune slot budgets on folds.

### Phase 3: Upgrade retrieval models

Use weighted implicit signals rather than only purchase SVD/cosine:

```text
purchase weight = 10
add_to_cart weight = 4-6
view_item weight = 1
recent-event multiplier = exp decay
```

Test:

- BM25-weighted item-item retrieval.
- ALS or SVD over weighted purchase + event matrix.
- Local/category constrained retrievers.
- Replenishment hazard candidates at item and category level.

The objective is to push candidate recall well above the current roughly `0.50` audit while keeping average candidate count manageable.

### Phase 4: Ranker refactor

After retrieval improves:

- Tune Optuna against actual fold `Precision@10`, not only LightGBM `ndcg@10`.
- Use source flags and source ranks as first-class features.
- Use fold-specific early stopping and reuse best iteration.
- Keep training negatives from the real candidate distribution.
- Run ablations by retrieval source and feature group.

## What to stop doing

Stop adding broad EDA-derived features without an ablation and a candidate-recall diagnosis. v12's tiny lift shows that general feature expansion is now low ROI.

The next big gains should come from:

1. Honest evaluation.
2. More true items entering the candidate pool.
3. Event-aware high-intent candidates.
4. Better cold-start fallback.
5. Source-aware reranking.

## Questions before implementation

1. For the private January submission, do we receive a fixed target customer list, or do we need to output recommendations for all known customers? All known customers 
2. Are January behavior events available at inference time, or only 2025 history/events through December? Inference time
3. What exactly does `items.sale_status` mean? December purchases include both `0` and `1`, so I would not hard-filter on it until confirmed. Maybe item sold at least once? You can use transaction to verify that

My professor also spoiled us before that event files are not that useful so keep in mind while using it

All data processing and analysing/using should use 100% polar for optimized speed and RAM full ultilization

Also new pipeline should mark with different pipeline (Maybe new_pir_pipeline) (starts with v1)
