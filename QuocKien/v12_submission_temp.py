"""Temporary December submission runner.

This script keeps the same submission structure as the normal pipeline:
customer_id -> [item_id, ...].

The only intentional change is the temporal split:
- train on months 1..11
- build/export recommendations for month 12 users

It reuses the V13 direct-upgrade helpers because they already encode the
current submission/export machinery in a standalone Python file.
"""

from __future__ import annotations

import argparse
import gc
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import optuna
import polars as pl


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pir_pipeline_v13_direct_upgrade import (  # noqa: E402
    ALL_FEATURES,
    CAT_FEATURES,
    DEFAULT_CF_CHUNK_SIZE,
    DEFAULT_CF_MAX_ITEMS,
    DEFAULT_CF_MAX_USERS,
    DEFAULT_EARLY_STOPPING_ROUNDS,
    DEFAULT_EVAL_SAMPLE_USERS,
    DEFAULT_INCLUDE_CF,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LIGHTGBM_DEVICE,
    DEFAULT_LAMBDA_L1,
    DEFAULT_LAMBDA_L2,
    DEFAULT_MAX_DEPTH,
    DEFAULT_MIN_DATA_IN_LEAF,
    DEFAULT_N_NEGATIVES,
    DEFAULT_NUM_BOOST_ROUND,
    DEFAULT_NUM_LEAVES,
    DEFAULT_TRAIN_SAMPLE_USERS,
    DEFAULT_USE_EVENTS,
    DEFAULT_EXPORT_BATCH_USERS,
    EVENT_PATH,
    ITEMS_PATH,
    SEED,
    TRANSACTION_PATH,
    candidate_diagnostics,
    create_dataset_v13,
    evaluate_scored,
    load_items,
    load_transactions,
    prep_lgb,
)


TRAIN_END_MONTH = 11
EVAL_MONTH = 12
OPTUNA_TRIALS = 1
OPTUNA_RANGES = {
    "learning_rate": (0.005, 0.09),
    "num_leaves": (63, 1023),
    "max_depth": (6, 16),
    "min_data_in_leaf": (20, 500),
    "lambda_l1": (1e-10, 50.0),
    "lambda_l2": (1e-10, 50.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Temporary December submission runner built from the v12/v13 pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=["train-eval", "export"], default="export")
    parser.add_argument("--transaction-path", type=Path, default=TRANSACTION_PATH)
    parser.add_argument("--event-path", type=Path, default=EVENT_PATH)
    parser.add_argument("--items-path", type=Path, default=ITEMS_PATH)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--model-name", default="v12_submission_temp_lgbm.txt")
    parser.add_argument("--model-path", type=Path, default=ROOT / "outputs" / "v12_submission_temp_lgbm.txt")
    parser.add_argument("--output-name", default="v12_submission_temp.pkl")
    parser.add_argument("--train-end-month", type=int, default=TRAIN_END_MONTH)
    parser.add_argument("--eval-month", type=int, default=EVAL_MONTH)
    parser.add_argument("--use-events", action="store_true", default=DEFAULT_USE_EVENTS)
    parser.add_argument("--enable-cf", action="store_true", default=DEFAULT_INCLUDE_CF)
    parser.add_argument("--no-cf", action="store_true")
    parser.add_argument("--cf-chunk-size", type=int, default=DEFAULT_CF_CHUNK_SIZE)
    parser.add_argument("--cf-max-users", type=int, default=DEFAULT_CF_MAX_USERS)
    parser.add_argument("--cf-max-items", type=int, default=DEFAULT_CF_MAX_ITEMS)
    parser.add_argument("--train-sample-users", type=int, default=DEFAULT_TRAIN_SAMPLE_USERS)
    parser.add_argument("--eval-sample-users", type=int, default=DEFAULT_EVAL_SAMPLE_USERS)
    parser.add_argument("--n-negatives", type=int, default=DEFAULT_N_NEGATIVES)
    parser.add_argument("--lightgbm-device", choices=["cpu", "gpu"], default=DEFAULT_LIGHTGBM_DEVICE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--num-leaves", type=int, default=DEFAULT_NUM_LEAVES)
    parser.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH)
    parser.add_argument("--min-data-in-leaf", type=int, default=DEFAULT_MIN_DATA_IN_LEAF)
    parser.add_argument("--lambda-l1", type=float, default=DEFAULT_LAMBDA_L1)
    parser.add_argument("--lambda-l2", type=float, default=DEFAULT_LAMBDA_L2)
    parser.add_argument("--num-boost-round", type=int, default=DEFAULT_NUM_BOOST_ROUND)
    parser.add_argument("--early-stopping-rounds", type=int, default=DEFAULT_EARLY_STOPPING_ROUNDS)
    parser.add_argument("--export-batch-users", type=int, default=DEFAULT_EXPORT_BATCH_USERS)
    return parser.parse_args()


def train_final_model(args: argparse.Namespace, df_raw: pl.DataFrame, items_df: pl.DataFrame):
    import lightgbm as lgb

    folds = [(8, 9), (9, 10), (10, 11)]
    fold_arrays = []
    fold_candidate_reports = []

    for train_end, val_month in folds:
        print(f"Building fold train<=M{train_end} -> M{val_month}")
        val_truth = df_raw.filter(pl.col("month") == val_month)
        fold_set = create_dataset_v13(
            history_df=df_raw.filter(pl.col("month") <= train_end),
            truth_df=val_truth,
            items_df=items_df,
            sample_users=args.train_sample_users,
            n_negatives=args.n_negatives,
            include_cf=args.enable_cf and not args.no_cf,
            include_events=args.use_events,
            event_path=args.event_path,
            mode="train",
            append_missing_positives=True,
            cf_chunk_size=args.cf_chunk_size,
            cf_max_users=args.cf_max_users,
            cf_max_items=args.cf_max_items,
        )
        fold_report = candidate_diagnostics(fold_set, val_truth)
        fold_report.update({"train_end_month": train_end, "validation_month": val_month})
        fold_candidate_reports.append(fold_report)
        print(json.dumps(fold_report, indent=2))
        fold_arrays.append(prep_lgb(fold_set))
        del fold_set, val_truth
        gc.collect()

    (x1, y1, g1), (x2, y2, g2), (x3, y3, g3) = fold_arrays
    x_train = np.vstack([x1, x2])
    y_train = np.concatenate([y1, y2])
    g_train = np.concatenate([g1, g2])

    cat_idx = [ALL_FEATURES.index(c) for c in CAT_FEATURES if c in ALL_FEATURES]

    def objective(trial):
        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [10],
            "learning_rate": trial.suggest_float("learning_rate", *OPTUNA_RANGES["learning_rate"], log=True),
            "num_leaves": trial.suggest_int("num_leaves", *OPTUNA_RANGES["num_leaves"]),
            "max_depth": trial.suggest_int("max_depth", *OPTUNA_RANGES["max_depth"]),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", *OPTUNA_RANGES["min_data_in_leaf"]),
            "lambda_l1": trial.suggest_float("lambda_l1", *OPTUNA_RANGES["lambda_l1"], log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", *OPTUNA_RANGES["lambda_l2"], log=True),
            "verbosity": -1,
            "random_state": SEED,
        }
        if args.lightgbm_device != "cpu":
            params.update({"device": args.lightgbm_device, "max_bin": 255})

        dtrain = lgb.Dataset(x_train, y_train, group=g_train, categorical_feature=cat_idx)
        dval = lgb.Dataset(x3, y3, group=g3, categorical_feature=cat_idx, reference=dtrain)
        model = lgb.train(
            params,
            dtrain,
            valid_sets=[dval],
            num_boost_round=args.num_boost_round,
            callbacks=[lgb.early_stopping(args.early_stopping_rounds), lgb.log_evaluation(25)],
        )
        score = model.best_score["valid_0"]["ndcg@10"]
        trial.set_user_attr("best_iteration", model.best_iteration or args.num_boost_round)
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=OPTUNA_TRIALS)
    best = study.best_trial
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [10],
        "learning_rate": best.params["learning_rate"],
        "num_leaves": best.params["num_leaves"],
        "max_depth": best.params["max_depth"],
        "min_data_in_leaf": best.params["min_data_in_leaf"],
        "lambda_l1": best.params["lambda_l1"],
        "lambda_l2": best.params["lambda_l2"],
        "verbosity": -1,
        "random_state": SEED,
    }
    if args.lightgbm_device != "cpu":
        params.update({"device": args.lightgbm_device, "max_bin": 255})

    dtrain = lgb.Dataset(x_train, y_train, group=g_train, categorical_feature=cat_idx)
    dval = lgb.Dataset(x3, y3, group=g3, categorical_feature=cat_idx, reference=dtrain)
    model = lgb.train(
        params,
        dtrain,
        valid_sets=[dval],
        num_boost_round=args.num_boost_round,
        callbacks=[lgb.early_stopping(args.early_stopping_rounds), lgb.log_evaluation(25)],
    )

    x_final = np.vstack([x1, x2, x3])
    y_final = np.concatenate([y1, y2, y3])
    g_final = np.concatenate([g1, g2, g3])
    final_rounds = model.best_iteration or args.num_boost_round
    del dtrain, dval, x_train, y_train, g_train
    gc.collect()

    final_data = lgb.Dataset(x_final, y_final, group=g_final, categorical_feature=cat_idx)
    final_model = lgb.train(params, final_data, num_boost_round=final_rounds)
    return final_model, fold_candidate_reports


def export_december_submission(
    args: argparse.Namespace,
    df_raw: pl.DataFrame,
    items_df: pl.DataFrame,
    model,
) -> Path:
    history_df = df_raw.filter(pl.col("month") <= args.train_end_month)
    eval_truth = df_raw.filter(pl.col("month") == args.eval_month)
    target_users = eval_truth.select("customer_id").unique().sort("customer_id")

    if args.eval_sample_users and target_users.height > args.eval_sample_users:
        target_users = target_users.head(args.eval_sample_users)

    submission = {}
    batch_size = args.export_batch_users
    for start in range(0, target_users.height, batch_size):
        batch_users = target_users.slice(start, batch_size)
        print(f"Export batch users {start + 1}-{start + batch_users.height} / {target_users.height}")
        export_set = create_dataset_v13(
            history_df=history_df,
            truth_df=None,
            items_df=items_df,
            sample_users=None,
            n_negatives=None,
            include_cf=args.enable_cf and not args.no_cf,
            include_events=args.use_events,
            event_path=args.event_path,
            mode="inference",
            all_history_users=True,
            target_users_override=batch_users,
            cf_chunk_size=args.cf_chunk_size,
            cf_max_users=args.cf_max_users,
            cf_max_items=args.cf_max_items,
        )
        x, _, _ = prep_lgb(export_set)
        export_set = export_set.with_columns(pl.Series("pred", model.predict(x)))
        top = (
            export_set.sort(["customer_id", "pred"], descending=[False, True])
            .group_by("customer_id", maintain_order=True)
            .head(10)
            .group_by("customer_id", maintain_order=True)
            .agg(pl.col("item_id").alias("recommendations"))
        )
        for customer_id, recommendations in top.iter_rows():
            submission[int(customer_id)] = [str(item_id) for item_id in recommendations]
        del export_set, x, top
        gc.collect()

    output_path = args.output_dir / args.output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(submission, f, protocol=4)

    metadata = {
        "train_end_month": args.train_end_month,
        "eval_month": args.eval_month,
        "users": len(submission),
        "all_values_len_10": all(len(v) == 10 for v in submission.values()),
        "key_type": type(next(iter(submission.keys()))).__name__ if submission else None,
        "value_type": type(next(iter(submission.values()))[0]).__name__ if submission else None,
    }
    output_path.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return output_path


def main() -> None:
    args = parse_args()
    df_raw = load_transactions(args.transaction_path)
    items_df = load_items(args.items_path)

    if args.mode == "train-eval":
        model, fold_reports = train_final_model(args, df_raw, items_df)
        metrics_set = create_dataset_v13(
            history_df=df_raw.filter(pl.col("month") <= args.train_end_month),
            truth_df=df_raw.filter(pl.col("month") == args.eval_month),
            items_df=items_df,
            sample_users=args.eval_sample_users,
            n_negatives=None,
            include_cf=args.enable_cf and not args.no_cf,
            include_events=args.use_events,
            event_path=args.event_path,
            mode="eval",
            append_missing_positives=False,
            cf_chunk_size=args.cf_chunk_size,
            cf_max_users=args.cf_max_users,
            cf_max_items=args.cf_max_items,
        )
        metrics_set = metrics_set.with_columns(pl.Series("pred", model.predict(prep_lgb(metrics_set)[0])))
        metrics = evaluate_scored(metrics_set, df_raw.filter(pl.col("month") == args.eval_month))
        metrics.update({"fold_candidate_reports": fold_reports})
        print(json.dumps(metrics, indent=2))
        return

    model, _ = train_final_model(args, df_raw, items_df)
    output_path = export_december_submission(args, df_raw, items_df, model)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()