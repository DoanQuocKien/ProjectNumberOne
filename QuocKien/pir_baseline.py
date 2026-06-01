from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD


TRAIN_MONTHS = {1, 2, 3, 4, 5, 6, 7, 8, 9}
VAL_MONTHS = {10}
TEST_MONTHS = {11}


@dataclass(frozen=True)
class SplitResult:
    train: pl.DataFrame
    validation: pl.DataFrame
    test: pl.DataFrame
    items: pl.DataFrame


def _detect_columns(columns: List[str]) -> Dict[str, str]:
    available = set(columns)

    def pick(*candidates: str) -> str:
        for candidate in candidates:
            if candidate in available:
                return candidate
        raise KeyError(f"Could not find any of: {', '.join(candidates)} in {sorted(available)}")

    return {
        "user": pick("customer_id", "user_id", "member_id", "buyer_id"),
        "item": pick("item_id", "product_id", "sku_id"),
        "date": pick("updated_date", "event_date", "date", "timestamp"),
        "quantity": pick("quantity", "qty", "amount"),
        "event": pick("event_type", "event", "event_name"),
    }


def _format_month(year: int, month: int) -> str:
    """Format year and month as YYYY-MM string."""
    return f"{year:04d}-{month:02d}"


def load_items(items_path: Path) -> pl.DataFrame:
    items = pl.scan_parquet(items_path).collect()
    if "item_id" not in items.columns:
        raise KeyError("items.parquet is missing item_id")
    return items


def load_splits(
    transaction_path: Path,
    items_path: Path,
    batch_size: int = 1_000_000,
    positive_event_types: Set[str] | None = None,
) -> SplitResult:
    if positive_event_types is None:
        positive_event_types = {"purchased"}

    transaction_schema = pl.scan_parquet(transaction_path).collect_schema()
    column_names = list(transaction_schema.names())
    detected = _detect_columns(column_names)

    raw = pl.read_parquet(
        transaction_path,
        columns=[
            detected["user"],
            detected["item"],
            detected["date"],
            detected["quantity"],
            detected["event"],
        ],
    )

    base_filtered = (
        raw
        .select(
            pl.col(detected["user"]).cast(pl.Int64, strict=False).alias("customer_id"),
            pl.col(detected["item"]).cast(pl.Utf8, strict=False).alias("item_id"),
            pl.col(detected["date"]).cast(pl.Datetime, strict=False).alias("event_ts"),
            pl.col(detected["quantity"]).cast(pl.Float64, strict=False).fill_null(0.0).alias("quantity"),
            pl.col(detected["event"]).cast(pl.Utf8, strict=False).str.to_lowercase().alias("event_type"),
        )
        .drop_nulls(["customer_id", "item_id", "event_ts"])
        .filter(pl.col("event_ts").dt.year() == 2025)
        .filter(
            (pl.col("quantity") > 0)
            | pl.col("event_type").is_in(sorted(positive_event_types))
        )
        .with_columns(pl.col("event_ts").dt.month().alias("month_number"))
    )

    def to_split_frame(months: Set[int]) -> pl.DataFrame:
        month_filters = [pl.col("month_number") == m for m in sorted(months)]
        combined_filter = month_filters[0] if month_filters else pl.lit(False)
        for month_filter in month_filters[1:]:
            combined_filter = combined_filter | month_filter
        
        if not month_filters:
            return pl.DataFrame({"customer_id": pl.Int64, "item_id": pl.Utf8, "weight": pl.Float32})
        
        return (
            base_filtered
            .filter(combined_filter)
            .group_by(["customer_id", "item_id"], maintain_order=False)
            .agg(pl.col("quantity").sum().cast(pl.Float32).alias("weight"))
            .select("customer_id", "item_id", "weight")
        )

    train = to_split_frame(TRAIN_MONTHS)
    validation = to_split_frame(VAL_MONTHS)
    test = to_split_frame(TEST_MONTHS)
    items = load_items(items_path)

    return SplitResult(train=train, validation=validation, test=test, items=items)


def _build_id_mappings(train: pl.DataFrame) -> Tuple[Dict[int, int], Dict[str, int], List[int], List[str]]:
    users = sorted(train.select("customer_id").unique().to_series().to_list())
    items = sorted(train.select("item_id").unique().to_series().to_list())
    user_to_index = {user_id: index for index, user_id in enumerate(users)}
    item_to_index = {item_id: index for index, item_id in enumerate(items)}
    return user_to_index, item_to_index, users, items


def build_sparse_matrix(train: pl.DataFrame) -> Tuple[csr_matrix, Dict[int, int], Dict[str, int], List[int], List[str]]:
    user_to_index, item_to_index, users, items = _build_id_mappings(train)
    
    # Map to indices using dictionary lookups (memory efficient)
    row_indices = np.array(
        [user_to_index[uid] for uid in train["customer_id"].to_list()],
        dtype=np.int32
    )
    col_indices = np.array(
        [item_to_index[iid] for iid in train["item_id"].to_list()],
        dtype=np.int32
    )
    values = np.ones(len(train), dtype=np.uint8)
    
    matrix = csr_matrix((values, (row_indices, col_indices)), shape=(len(users), len(items)))
    matrix.sum_duplicates()
    return matrix, user_to_index, item_to_index, users, items


class TruncatedSVDRecommender:
    def __init__(self, n_components: int = 64, random_state: int = 42) -> None:
        self.n_components = n_components
        self.random_state = random_state
        self.model = TruncatedSVD(n_components=n_components, random_state=random_state)
        self.matrix: csr_matrix | None = None
        self.user_to_index: Dict[int, int] = {}
        self.item_to_index: Dict[str, int] = {}
        self.index_to_item: List[str] = []
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None
        self.item_popularity: np.ndarray | None = None

    def fit(self, train: pl.DataFrame) -> "TruncatedSVDRecommender":
        matrix, user_to_index, item_to_index, _, items = build_sparse_matrix(train)
        self.matrix = matrix
        self.user_to_index = user_to_index
        self.item_to_index = item_to_index
        self.index_to_item = items
        self.model.fit(matrix)
        self.user_factors = self.model.transform(matrix)
        self.item_factors = self.model.components_.T
        self.item_popularity = np.asarray(matrix.sum(axis=0)).ravel().astype(float)
        return self

    def _fallback_ranking(self, top_k: int) -> List[str]:
        if self.item_popularity is None:
            return self.index_to_item[:top_k]
        order = np.argsort(-self.item_popularity)
        return [self.index_to_item[index] for index in order[:top_k]]

    def recommend(self, customer_id: int, seen_items: Iterable[str] | None = None, top_k: int = 10) -> List[str]:
        item_factors = self.item_factors
        user_factors = self.user_factors
        if item_factors is None or user_factors is None:
            raise RuntimeError("Model has not been fit yet")

        if customer_id not in self.user_to_index:
            return self._fallback_ranking(top_k)

        user_index = self.user_to_index[customer_id]
        user_vector = user_factors[user_index]
        scores = user_vector @ item_factors.T
        scores = np.asarray(scores).ravel()

        if seen_items is not None:
            for item_id in seen_items:
                item_index = self.item_to_index.get(item_id)
                if item_index is not None:
                    scores[item_index] = -np.inf

        ranking = np.argsort(-scores)
        recommended = [self.index_to_item[index] for index in ranking[:top_k]]
        if len(recommended) < top_k:
            fallback = self._fallback_ranking(top_k)
            for item_id in fallback:
                if item_id not in recommended:
                    recommended.append(item_id)
                if len(recommended) == top_k:
                    break
        return recommended[:top_k]


def build_truth_map(frame: pl.DataFrame) -> Dict[int, Set[str]]:
    truth: Dict[int, Set[str]] = {}
    if frame.is_empty():
        return truth
    for row in frame.iter_rows(named=True):
        cid = int(row["customer_id"])
        iid = str(row["item_id"])
        if cid not in truth:
            truth[cid] = set()
        truth[cid].add(iid)
    return truth


def build_seen_map(frame: pl.DataFrame) -> Dict[int, Set[str]]:
    seen: Dict[int, Set[str]] = {}
    if frame.is_empty():
        return seen
    for row in frame.iter_rows(named=True):
        cid = int(row["customer_id"])
        iid = str(row["item_id"])
        if cid not in seen:
            seen[cid] = set()
        seen[cid].add(iid)
    return seen


def precision_at_k(recommended: Sequence[str], truth: Set[str], top_k: int) -> float:
    if top_k == 0:
        return 0.0
    hits = sum(1 for item_id in recommended[:top_k] if item_id in truth)
    return hits / top_k


def reciprocal_rank(recommended: Sequence[str], truth: Set[str]) -> float:
    for index, item_id in enumerate(recommended, start=1):
        if item_id in truth:
            return 1.0 / index
    return 0.0


def average_precision_at_k(recommended: Sequence[str], truth: Set[str], top_k: int) -> float:
    if not truth:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for index, item_id in enumerate(recommended[:top_k], start=1):
        if item_id in truth:
            hits += 1
            precision_sum += hits / index
    denominator = min(len(truth), top_k)
    return precision_sum / denominator if denominator else 0.0


def intersection_over_union(recommended: Sequence[str], truth: Set[str], top_k: int) -> float:
    recommended_set = set(recommended[:top_k])
    intersection = len(recommended_set & truth)
    union = len(recommended_set | truth)
    return intersection / union if union else 0.0


def evaluate_recommender(
    model: TruncatedSVDRecommender,
    truth_map: Dict[int, Set[str]],
    seen_map: Dict[int, Set[str]],
    top_k: int = 10,
) -> Dict[str, float]:
    users = sorted(truth_map.keys())
    if not users:
        return {"precision@k": 0.0, "mrr": 0.0, "map@k": 0.0, "iou": 0.0, "evaluated_users": 0.0}

    precision_scores = []
    mrr_scores = []
    map_scores = []
    iou_scores = []

    for customer_id in users:
        truth = truth_map[customer_id]
        recommendations = model.recommend(customer_id, seen_items=seen_map.get(customer_id, set()), top_k=top_k)
        precision_scores.append(precision_at_k(recommendations, truth, top_k))
        mrr_scores.append(reciprocal_rank(recommendations, truth))
        map_scores.append(average_precision_at_k(recommendations, truth, top_k))
        iou_scores.append(intersection_over_union(recommendations, truth, top_k))

    return {
        "precision@k": float(np.mean(precision_scores)),
        "mrr": float(np.mean(mrr_scores)),
        "map@k": float(np.mean(map_scores)),
        "iou": float(np.mean(iou_scores)),
        "evaluated_users": float(len(users)),
    }


def popularity_baseline(train: pl.DataFrame, top_k: int = 10) -> List[str]:
    if train.is_empty():
        return []
    ranking = (
        train.group_by("item_id", maintain_order=False)
        .agg(pl.col("weight").sum().alias("total_weight"))
        .sort("total_weight", descending=True)
        .head(top_k)
        .get_column("item_id")
        .to_list()
    )
    return [str(item) for item in ranking]


def make_submission(
    model: TruncatedSVDRecommender,
    customer_ids: Iterable[int],
    seen_map: Dict[int, Set[str]],
    top_k: int = 10,
) -> Dict[str, List[str]]:
    submission: Dict[str, List[str]] = {}
    for customer_id in customer_ids:
        submission[str(customer_id)] = model.recommend(customer_id, seen_items=seen_map.get(customer_id, set()), top_k=top_k)
    return submission


def save_json_submission(submission: Dict[str, List[str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(submission, ensure_ascii=False, indent=2), encoding="utf-8")
