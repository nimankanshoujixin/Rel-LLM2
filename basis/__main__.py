import argparse

import torch

from basis.builder import DBConditionedSemanticBasisBuilder
from relbench.datasets import get_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a DB-conditioned semantic basis artifact.")
    parser.add_argument("--dataset", type=str, required=True, help="RelBench dataset name.")
    parser.add_argument("--model-type", type=str, required=True, help="LLM name or local path.")
    parser.add_argument(
        "--output-root",
        type=str,
        default="artifacts/basis",
        help="Directory root used to save basis artifacts.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size used when embedding basis descriptions.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the dataset through RelBench if needed.",
    )
    parser.add_argument(
        "--db-scope",
        type=str,
        default="train_visible",
        choices=["train_visible", "upto_test", "full"],
        help=(
            "Database scope used to build the basis. "
            "'train_visible' uses the full historical database visible at training time, "
            "truncated at dataset.val_timestamp. "
            "'upto_test' uses data up to dataset.test_timestamp. "
            "'full' uses the complete database including future rows."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional device for the LLM embedding matrix, e.g. cuda:0 or cpu.",
    )
    return parser.parse_args()


def resolve_db(dataset, db_scope: str):
    if db_scope == "train_visible":
        return dataset.get_db(upto_test_timestamp=False).upto(dataset.val_timestamp)
    if db_scope == "upto_test":
        return dataset.get_db(upto_test_timestamp=True)
    if db_scope == "full":
        return dataset.get_db(upto_test_timestamp=False)
    raise ValueError(f"Unsupported db scope: {db_scope}")


def main() -> None:
    args = parse_args()
    dataset = get_dataset(args.dataset, download=args.download)
    db = resolve_db(dataset, args.db_scope)

    device = args.device
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    builder, model = DBConditionedSemanticBasisBuilder.from_model_name(
        model_name_or_path=args.model_type,
        output_root=args.output_root,
        batch_size=args.batch_size,
        device=device,
    )
    output_path = builder.save(builder.build(args.dataset, db))
    del model
    print(f"Saved semantic basis to {output_path}")


if __name__ == "__main__":
    main()
