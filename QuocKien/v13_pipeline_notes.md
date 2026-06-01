# V13 Polars Heuristic Notes

This earlier `pir_pipeline_v13_polars.py` file is a lightweight heuristic smoke/export baseline, not the direct v12 upgrade.

Use `pir_pipeline_v13_direct_upgrade.py` and `v13_direct_upgrade_notes.md` for the real v13 path.

## Files

- Pipeline: `QuocKien/pir_pipeline_v13_polars.py`
- Local smoke pickle: `QuocKien/outputs/v13_polars_smoke_1000.pkl`
- Local smoke metadata: `QuocKien/outputs/v13_polars_smoke_1000.metadata.json`

Ignore any large full-export `.pkl` that exists without a matching `.metadata.json`; it likely came from an interrupted run and should not be submitted.

## Local Smoke Tests

Run a small honest December evaluation:

```bash
python -X utf8 QuocKien/pir_pipeline_v13_polars.py --eval-month 12 --sample-users 5000 --use-events
```

Run a tiny pickle export:

```bash
python -X utf8 QuocKien/pir_pipeline_v13_polars.py --export-all --export-limit 1000 --use-events --output-name v13_polars_smoke_1000.pkl
```

## Full Export On High-RAM Machine

Use this for the requested all-known-customer pickle:

```bash
python -X utf8 QuocKien/pir_pipeline_v13_polars.py --export-all --confirm-large-export --use-events --output-name v13_polars_fast_all_users_recommendations.pkl
```

The output is a standard pickle dictionary:

```python
{
    customer_id: ["item_id_1", "item_id_2", "..."],
    ...
}
```

## Optional Richer Export

This adds local and category candidates. It is slower and uses more RAM:

```bash
python -X utf8 QuocKien/pir_pipeline_v13_polars.py --export-all --confirm-large-export --use-events --rich --output-name v13_polars_rich_all_users_recommendations.pkl
```

Use the fast export first unless the high-RAM environment has plenty of headroom.

## RAM Notes

The full export covers about 3.02M known transaction customers. It should not be run on the local 3-4GB free-RAM machine. The script requires `--confirm-large-export` for full export to avoid accidental local runs.

The pickle writer streams the dictionary entries to disk to avoid building a second giant Python dictionary in memory, but the Polars candidate-ranking stage is still the heavy part.
