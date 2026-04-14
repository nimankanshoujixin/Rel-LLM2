from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch

from basis.builder import DBConditionedSemanticBasisBuilder


class FakeTokenizer:
    def __init__(self) -> None:
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self._vocab = {self.pad_token: 0}

    def _encode_text(self, text: str) -> list[int]:
        token_ids = []
        for token in text.split():
            if token not in self._vocab:
                self._vocab[token] = len(self._vocab)
            token_ids.append(self._vocab[token])
        return token_ids or [self._vocab[self.pad_token]]

    def __call__(
        self,
        texts,
        add_special_tokens: bool = False,
        padding: bool = False,
        truncation: bool = False,
        return_tensors: str | None = None,
    ):
        del add_special_tokens, truncation
        if isinstance(texts, str):
            texts = [texts]
        encoded = [self._encode_text(text) for text in texts]
        if not padding:
            return {"input_ids": encoded}

        max_len = max(len(ids) for ids in encoded)
        padded = []
        masks = []
        for ids in encoded:
            pad_len = max_len - len(ids)
            padded.append(ids + [self._vocab[self.pad_token]] * pad_len)
            masks.append([1] * len(ids) + [0] * pad_len)

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(padded, dtype=torch.long),
                "attention_mask": torch.tensor(masks, dtype=torch.long),
            }
        return {"input_ids": padded, "attention_mask": masks}


class FakeInputEmbedding(torch.nn.Module):
    def __init__(self, vocab_size: int = 4096, dim: int = 8) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(vocab_size, dim), requires_grad=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.embedding(input_ids, self.weight)


def make_fake_db():
    users = SimpleNamespace(
        df=pd.DataFrame(
            {
                "user_id": [0, 1, 2],
                "country": ["cn", "us", "cn"],
                "tags": [["a", "b"], ["b"], ["c"]],
                "is_active": [True, False, True],
                "event_time": pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-05"]),
            }
        ),
        fkey_col_to_pkey_table={},
        pkey_col="user_id",
        time_col="event_time",
    )
    orders = SimpleNamespace(
        df=pd.DataFrame(
            {
                "order_id": [0, 1, 2, 3],
                "user_id": [0, 1, 1, 2],
                "amount": [12.5, 8.0, 13.0, 20.0],
                "items": [torch.tensor([1, 2]).numpy(), torch.tensor([2]).numpy(), torch.tensor([3, 4]).numpy(), torch.tensor([5]).numpy()],
                "created_at": pd.to_datetime(["2024-01-02", "2024-01-04", "2024-01-06", "2024-01-08"]),
            }
        ),
        fkey_col_to_pkey_table={"user_id": "users"},
        pkey_col="order_id",
        time_col="created_at",
    )
    return SimpleNamespace(table_dict={"users": users, "orders": orders})


def test_basis_builder_builds_and_saves_artifact(tmp_path: Path):
    builder = DBConditionedSemanticBasisBuilder(
        tokenizer=FakeTokenizer(),
        input_embedding=FakeInputEmbedding(dim=8),
        output_root=tmp_path,
        batch_size=16,
    )
    db = make_fake_db()

    artifact = builder.build("fake-db", db)

    assert artifact.db_name == "fake-db"
    assert artifact.A_db.ndim == 2
    assert artifact.A_db.shape[1] == 8
    assert artifact.A_db.shape[0] == len(artifact.basis_ids)
    assert len(artifact.basis_ids) == len(artifact.basis_types) == len(artifact.basis_descs)
    assert "table::orders" in artifact.basis_ids
    assert "column::users::country" in artifact.basis_ids
    assert "pk::users::user_id" in artifact.basis_ids
    assert "fk::orders::user_id::users::user_id" in artifact.basis_ids
    assert "join::orders::users" in artifact.basis_ids
    assert any(basis_id.startswith("stat::orders::amount::quantile::") for basis_id in artifact.basis_ids)
    assert any(basis_id.startswith("stat::users::country::category_support::") for basis_id in artifact.basis_ids)

    output_path = builder.save(artifact)
    assert output_path == tmp_path / "fake-db" / "basis.pt"
    saved = torch.load(output_path)
    assert saved["db_name"] == "fake-db"
    assert saved["A_db"].shape == artifact.A_db.shape
    assert saved["basis_ids"] == artifact.basis_ids
    assert saved["basis_types"] == artifact.basis_types
    assert saved["basis_descs"] == artifact.basis_descs
