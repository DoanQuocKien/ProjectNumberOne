# V13 Direct Upgrade Notes

## What This Version Is

`pir_pipeline_v13_direct_upgrade.py` is the direct upgrade from v12. It keeps:

- V12 two-stage design: retrieval -> feature matrix -> LightGBM LambdaRank.
- V12 core retrieval: history, replenishment, global, local, SVD, I2I, category top.
- V12 feature families: user, item, user-item, location, category/brand ids, momentum, size-age proxy.

It upgrades candidate selection with EDA-backed sources:

- Idea 13 / 42: top-3 category discovery, not only top-1 category.
- Idea 21 / 37 / 38 / 47: local-category candidates based on primary store and local assortment.
- Idea 2 / 43: preferred brand within preferred category.
- Idea 9 / 23 / 39: category-specific momentum candidates.
- Idea 6 / 17 / 44: stronger replenishment source.
- Idea 8 / 46 / 48: segment-aware source budgets. Targeted habituals receive more history/replenishment/brand candidates; active discoverers receive more local/global/momentum/CF candidates; hibernators receive broader reactivation/local/global coverage.
- Idea 6 / 17 / 44, v13.1: replenishment now backs off from user-item average gaps to item/category median repeat gaps, so one observed staple purchase can still become a due candidate if the item/category has repeat evidence.
- Source-aware features: `src_*`, `rank_*`, `retr_score`, and `retr_source_count`.
- Replenishment-aware ranker features: expected gap, due ratio, gap error, item/category repeat observations.

Events are optional via `--use-events`, but not central, because the professor warned they may be weak.

## Default Run Size

The script and notebook keep the v12-scale comparison numbers by default:

- `train_sample_users=60,000`
- `eval_sample_users=40,000`
- `n_negatives=150`
- `num_boost_round=800`
- `early_stopping_rounds=50`
- `SVD/I2I CF disabled by default`
- export is batched at `10,000` users per batch

These defaults live in `pir_pipeline_v13_direct_upgrade.py` as `DEFAULT_*` constants. The notebook imports them instead of duplicating numbers, so the Python file is the source of truth.

The RAM optimization is not smaller samples. It is Polars-first execution:

- Negative sampling is now deterministic hard-negative selection inside Polars, not a full random shuffle of all negatives.
- Each temporal fold is converted to LightGBM arrays and the Polars fold frame is freed before the next memory-heavy phase.
- Training arrays used only for validation are freed before final refit.
- Month 12 is still built only after the pre-M12 final model is trained.
- Export writes the pickle in user batches instead of materializing all users at once.

Install dependencies on the high-RAM environment:

```bash
pip install -r QuocKien/requirements.txt
```

## Train And Evaluate

### Kaggle Setup

Upload or attach the `QuocKien` folder so the notebook can see:

- `pir_pipeline_v13_direct_upgrade.ipynb`
- `pir_pipeline_v13_direct_upgrade.py`

Attach the data dataset containing:

- `transaction_full_2025.parquet`
- `event_full_2025.parquet`
- `items.parquet`

The code automatically searches `/kaggle/input/**` for those parquet filenames and writes outputs to:

```text
/kaggle/working/QuocKien/outputs
```

For a normal Kaggle run, use the notebook:

```text
QuocKien/pir_pipeline_v13_direct_upgrade.ipynb
```

The first code cell prints the resolved parquet paths and output directory.

V12-comparable evaluation mode, where missed positives may be appended to the evaluation candidate set:

```bash
python -X utf8 QuocKien/pir_pipeline_v13_direct_upgrade.py \
  --mode train-eval \
  --eval-append-missing-positives
```

Default strict end-to-end retrieval evaluation mode, where evaluation candidates must come from retrieval:

```bash
python -X utf8 QuocKien/pir_pipeline_v13_direct_upgrade.py \
  --mode train-eval
```

Same defaults on Kaggle/GPU, v12-comparable:

```bash
python -X utf8 QuocKien/pir_pipeline_v13_direct_upgrade.py \
  --mode train-eval \
  --train-sample-users 60000 \
  --eval-sample-users 40000 \
  --n-negatives 150 \
  --num-boost-round 800 \
  --early-stopping-rounds 50 \
  --eval-append-missing-positives \
  --lightgbm-device gpu
```

Larger Kaggle/GPU version with CF enabled:

```bash
python -X utf8 QuocKien/pir_pipeline_v13_direct_upgrade.py \
  --mode train-eval \
  --train-sample-users 60000 \
  --eval-sample-users 40000 \
  --n-negatives 150 \
  --num-boost-round 800 \
  --early-stopping-rounds 50 \
  --eval-append-missing-positives \
  --lightgbm-device gpu \
  --enable-cf \
  --cf-max-users 80000 \
  --cf-max-items 30000 \
  --cf-chunk-size 1500
```

## Export All Known Transaction Customers

After training creates `QuocKien/outputs/v13_direct_lgbm.txt`, run:

```bash
python -X utf8 QuocKien/pir_pipeline_v13_direct_upgrade.py \
  --mode export-all \
  --model-path QuocKien/outputs/v13_direct_lgbm.txt \
  --confirm-large-export \
  --export-batch-users 10000 \
  --output-name v13_direct_all_users_recommendations.pkl
```

This writes a standard pickle dictionary:

```python
{
    customer_id: ["item_id_1", "item_id_2", "..."],
    ...
}
```

For an export smoke test:

```bash
python -X utf8 QuocKien/pir_pipeline_v13_direct_upgrade.py \
  --mode export-all \
  --model-path QuocKien/outputs/v13_direct_lgbm.txt \
  --export-limit 1000 \
  --export-batch-users 1000 \
  --output-name v13_direct_smoke_1000.pkl
```
