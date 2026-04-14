from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


NUMERIC_DTYPES = {"integer", "float", "boolean"}
TEXT_DTYPES = {"categorical", "string", "text"}


@dataclass(frozen=True)
class BasisArtifact:
    db_name: str
    A_db: torch.Tensor
    basis_ids: list[str]
    basis_types: list[str]
    basis_descs: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "db_name": self.db_name,
            "A_db": self.A_db,
            "basis_ids": self.basis_ids,
            "basis_types": self.basis_types,
            "basis_descs": self.basis_descs,
        }


@dataclass(frozen=True)
class BasisItem:
    basis_id: str
    basis_type: str
    desc: str


class DBConditionedSemanticBasisBuilder:
    """Builds a fixed database-conditioned semantic basis from schema and stats."""

    def __init__(
        self,
        tokenizer,
        input_embedding,
        output_root: str | Path = "artifacts/basis",
        batch_size: int = 256,
    ) -> None:
        self.tokenizer = tokenizer
        self.input_embedding = input_embedding
        self.output_root = Path(output_root)
        self.batch_size = batch_size
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @classmethod
    def from_model_name(
        cls,
        model_name_or_path: str,
        output_root: str | Path = "artifacts/basis",
        batch_size: int = 256,
        device: str | torch.device | None = None,
    ) -> tuple["DBConditionedSemanticBasisBuilder", Any]:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_fast=False,
            padding_side="right",
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        if device is not None:
            model = model.to(device)
        builder = cls(
            tokenizer=tokenizer,
            input_embedding=model.get_input_embeddings(),
            output_root=output_root,
            batch_size=batch_size,
        )
        return builder, model

    def build(self, db_name: str, db) -> BasisArtifact:
        items = self.enumerate_basis_items(db)
        basis_descs = [item.desc for item in items]
        basis_vectors = self.embed_descriptions(basis_descs)
        return BasisArtifact(
            db_name=db_name,
            A_db=basis_vectors,
            basis_ids=[item.basis_id for item in items],
            basis_types=[item.basis_type for item in items],
            basis_descs=basis_descs,
        )

    def build_and_save(self, db_name: str, db, output_path: str | Path | None = None) -> BasisArtifact:
        artifact = self.build(db_name, db)
        self.save(artifact, output_path=output_path)
        return artifact

    def save(self, artifact: BasisArtifact, output_path: str | Path | None = None) -> Path:
        path = Path(output_path) if output_path is not None else self.output_root / artifact.db_name / "basis.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(artifact.to_dict(), path)
        return path

    def enumerate_basis_items(self, db) -> list[BasisItem]:
        items: dict[str, BasisItem] = {}

        for table_name in sorted(db.table_dict):
            table = db.table_dict[table_name]
            self._add_item(
                items,
                BasisItem(
                    basis_id=f"table::{table_name}",
                    basis_type="table",
                    desc=f"table {table_name}",
                ),
            )

            for column_name in sorted(table.df.columns):
                self._add_item(
                    items,
                    BasisItem(
                        basis_id=f"column::{table_name}::{column_name}",
                        basis_type="column",
                        desc=f"column {table_name} {column_name}",
                    ),
                )
                if table.pkey_col == column_name:
                    self._add_item(
                        items,
                        BasisItem(
                            basis_id=f"pk::{table_name}::{column_name}",
                            basis_type="pk",
                            desc=f"primary key {table_name} {column_name}",
                        ),
                    )

            for src_col, dst_table in sorted(table.fkey_col_to_pkey_table.items()):
                dst_col = db.table_dict[dst_table].pkey_col or "unknown"
                self._add_item(
                    items,
                    BasisItem(
                        basis_id=f"fk::{table_name}::{src_col}::{dst_table}::{dst_col}",
                        basis_type="fk",
                        desc=f"foreign key from {table_name} {src_col} to {dst_table} {dst_col}",
                    ),
                )

            for stat_item in self._enumerate_stat_items(table_name, table):
                self._add_item(items, stat_item)

        for join_item in self._enumerate_join_items(db):
            self._add_item(items, join_item)

        return list(items.values())

    def embed_descriptions(self, descriptions: list[str]) -> torch.Tensor:
        vectors: list[torch.Tensor] = []
        embedding_weight = self.input_embedding.weight
        device = embedding_weight.device

        with torch.inference_mode():
            for start in range(0, len(descriptions), self.batch_size):
                batch_descs = descriptions[start : start + self.batch_size]
                encoded = self.tokenizer(
                    batch_descs,
                    add_special_tokens=False,
                    padding=True,
                    truncation=False,
                    return_tensors="pt",
                )
                input_ids = encoded["input_ids"].to(device)
                attention_mask = encoded["attention_mask"].to(device)
                token_embeds = self.input_embedding(input_ids)
                masked_embeds = token_embeds * attention_mask.unsqueeze(-1)
                denom = attention_mask.sum(dim=1, keepdim=True).clamp_min(1)
                pooled = masked_embeds.sum(dim=1) / denom
                vectors.append(pooled.float().cpu())
        return torch.cat(vectors, dim=0)

    def _enumerate_stat_items(self, table_name: str, table) -> Iterable[BasisItem]:
        df = table.df
        for column_name in sorted(df.columns):
            series = df[column_name]
            dtype_bucket = self._infer_dtype_bucket(series)
            yield BasisItem(
                basis_id=f"stat::{table_name}::{column_name}::dtype::{dtype_bucket}",
                basis_type="stat_dtype",
                desc=f"column {table_name} {column_name} has dtype {dtype_bucket}",
            )

            unique_ratio = self._safe_unique_ratio(series)
            unique_bucket = self._bucket_ratio(unique_ratio)
            yield BasisItem(
                basis_id=f"stat::{table_name}::{column_name}::unique::{unique_bucket}",
                basis_type="stat_unique",
                desc=f"column {table_name} {column_name} has unique ratio {unique_bucket}",
            )

            missing_ratio = float(series.isna().mean()) if len(series) > 0 else 0.0
            missing_bucket = self._bucket_ratio(missing_ratio)
            yield BasisItem(
                basis_id=f"stat::{table_name}::{column_name}::missing::{missing_bucket}",
                basis_type="stat_missing",
                desc=f"column {table_name} {column_name} has missing ratio {missing_bucket}",
            )

            if dtype_bucket in TEXT_DTYPES:
                category_support = int(series.nunique(dropna=True))
                category_bucket = self._bucket_category_support(category_support)
                yield BasisItem(
                    basis_id=f"stat::{table_name}::{column_name}::category_support::{category_bucket}",
                    basis_type="stat_category_support",
                    desc=f"column {table_name} {column_name} has category support {category_bucket}",
                )

            if dtype_bucket in NUMERIC_DTYPES:
                quantile_bucket = self._bucket_quantiles(series)
                yield BasisItem(
                    basis_id=f"stat::{table_name}::{column_name}::quantile::{quantile_bucket}",
                    basis_type="stat_quantile",
                    desc=f"column {table_name} {column_name} has quantile signature {quantile_bucket}",
                )

            if dtype_bucket == "time":
                time_bucket = self._bucket_time_range(series)
                yield BasisItem(
                    basis_id=f"stat::{table_name}::{column_name}::time::{time_bucket}",
                    basis_type="stat_time",
                    desc=f"column {table_name} {column_name} has time range {time_bucket}",
                )

    def _enumerate_join_items(self, db) -> Iterable[BasisItem]:
        adjacency: dict[str, set[str]] = {table_name: set() for table_name in db.table_dict}
        for src_table, table in db.table_dict.items():
            for _, dst_table in table.fkey_col_to_pkey_table.items():
                adjacency[src_table].add(dst_table)
                adjacency[dst_table].add(src_table)

        one_hop_paths: set[tuple[str, str]] = set()
        two_hop_paths: set[tuple[str, str, str]] = set()

        for src in sorted(adjacency):
            for mid in sorted(adjacency[src]):
                pair = min((src, mid), (mid, src))
                one_hop_paths.add(pair)
                for dst in sorted(adjacency[mid]):
                    if dst == src or dst == mid:
                        continue
                    path = (src, mid, dst)
                    canonical = min(path, path[::-1])
                    two_hop_paths.add(canonical)

        for table_a, table_b in sorted(one_hop_paths):
            yield BasisItem(
                basis_id=f"join::{table_a}::{table_b}",
                basis_type="join_1hop",
                desc=f"join path {table_a} to {table_b}",
            )

        for table_a, table_b, table_c in sorted(two_hop_paths):
            yield BasisItem(
                basis_id=f"join::{table_a}::{table_b}::{table_c}",
                basis_type="join_2hop",
                desc=f"join path {table_a} to {table_b} to {table_c}",
            )

    @staticmethod
    def _add_item(items: dict[str, BasisItem], item: BasisItem) -> None:
        items.setdefault(item.basis_id, item)

    @staticmethod
    def _infer_dtype_bucket(series: pd.Series) -> str:
        if pd.api.types.is_datetime64_any_dtype(series):
            return "time"
        if pd.api.types.is_bool_dtype(series):
            return "boolean"
        if pd.api.types.is_integer_dtype(series):
            return "integer"
        if pd.api.types.is_float_dtype(series):
            return "float"
        if pd.api.types.is_categorical_dtype(series):
            return "categorical"
        if pd.api.types.is_string_dtype(series):
            return "string"
        if pd.api.types.is_object_dtype(series):
            non_null = series.dropna()
            if non_null.empty:
                return "string"
            sample = non_null.astype(str).head(128)
            avg_length = sample.map(len).mean()
            if avg_length >= 32 or sample.str.contains(r"\s", regex=True).mean() >= 0.5:
                return "text"
            return "string"
        return "other"

    @staticmethod
    def _safe_unique_ratio(series: pd.Series) -> float:
        if len(series) == 0:
            return 0.0
        return float(series.nunique(dropna=True) / max(len(series), 1))

    @staticmethod
    def _bucket_ratio(value: float) -> str:
        thresholds = (
            (0.01, "very_low"),
            (0.05, "low"),
            (0.20, "mid"),
            (0.50, "high"),
        )
        for threshold, bucket in thresholds:
            if value <= threshold:
                return bucket
        return "very_high"

    @staticmethod
    def _bucket_category_support(value: int) -> str:
        if value <= 1:
            return "singleton"
        if value == 2:
            return "binary"
        if value <= 10:
            return "small"
        if value <= 100:
            return "medium"
        if value <= 1000:
            return "large"
        return "huge"

    def _bucket_quantiles(self, series: pd.Series) -> str:
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        if numeric.empty:
            return "empty"
        quantiles = numeric.quantile([0.10, 0.25, 0.50, 0.75, 0.90]).tolist()
        buckets = [self._bucket_signed_value(float(value)) for value in quantiles]
        return "__".join(
            [
                f"q10_{buckets[0]}",
                f"q25_{buckets[1]}",
                f"q50_{buckets[2]}",
                f"q75_{buckets[3]}",
                f"q90_{buckets[4]}",
            ]
        )

    @staticmethod
    def _bucket_signed_value(value: float) -> str:
        abs_value = abs(value)
        if abs_value < 1e-8:
            magnitude = "zero"
        elif abs_value < 1e-3:
            magnitude = "tiny"
        elif abs_value < 1e-1:
            magnitude = "small"
        elif abs_value < 1:
            magnitude = "subunit"
        elif abs_value < 10:
            magnitude = "unit"
        elif abs_value < 1e3:
            magnitude = "large"
        else:
            magnitude = "huge"

        if value > 0:
            sign = "pos"
        elif value < 0:
            sign = "neg"
        else:
            sign = "zero"
        return f"{sign}_{magnitude}"

    @staticmethod
    def _bucket_time_range(series: pd.Series) -> str:
        timestamps = pd.to_datetime(series, errors="coerce").dropna()
        if timestamps.empty:
            return "empty"
        duration = timestamps.max() - timestamps.min()
        days = duration.total_seconds() / 86400.0
        if days < 1:
            return "lt_1d"
        if days < 7:
            return "1d_1w"
        if days < 30:
            return "1w_1m"
        if days < 180:
            return "1m_6m"
        if days < 365 * 2:
            return "6m_2y"
        return "gt_2y"
