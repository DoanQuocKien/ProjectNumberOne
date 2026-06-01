# PIR Baseline in `QuocKien`

This folder contains the first implementation step for the Personalized Item Recommendation project.

## What this baseline does

- Uses `transaction_full_2025.parquet` and `items.parquet` only.
- Uses Polars lazy scanning/grouping for faster full-parquet processing.
- Splits time strictly by month:
  - Train: January to September 2025
  - Validation: October 2025
  - Test: November 2025
- Builds a 2D user-item interaction matrix from purchases.
- Trains a collaborative-filtering baseline with `TruncatedSVD`.
- Evaluates `Precision@K`, `MRR`, `MAP@K`, and `IoU`.
- Optional final retraining step on all used months (Jan-Nov) before export.
- Exports a JSON submission mapping `customer_id -> [item_id, ...]`.

## Why days are not a third dimension here

For a baseline, days are better treated as time-split boundaries or future tabular features such as recency, frequency, and month-based aggregates. A 3D tensor is usually unnecessary at this stage and makes the pipeline much harder to train and debug.

## Run

```bash
python QuocKien/run_baseline.py --smoke-test
python QuocKien/run_baseline.py --top-k 10 --train-on-all-used-data
```

## Notes

- The current implementation ignores `event_full_2025.parquet` for the first baseline, as requested.
- Cold-start users fall back to popularity-based recommendations.
- You can add LightGBM or other tabular models later using the same time splits.
