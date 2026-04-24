"""Training entry for first runnable SNF-MV full model with consistency reasoning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from src.datasets.weibo_dataset import WeiboStructuralDataset
from src.evaluators.metrics import binary_classification_metrics
from src.models.full_model import FullModel
from src.utils.seed import set_seed


def _build_dataloader(
    dataset: WeiboStructuralDataset,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=WeiboStructuralDataset.collate_fn,
    )


def train_one_epoch(
    model: FullModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    seen = 0

    for batch in dataloader:
        texts = [str(text or "") for text in batch["text"]]
        labels = torch.tensor(batch["overall_label"], dtype=torch.long, device=device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(texts=texts)

        # First iteration: optimize only overall fake/real objective.
        # Future extension: add auxiliary losses for view labels and
        # pairwise consistency/conflict supervision when reliable labels exist.
        loss = criterion(outputs["logits"], labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += float(loss.item()) * batch_size
        seen += batch_size

    return running_loss / max(1, seen)


@torch.no_grad()
def evaluate(model: FullModel, dataloader: DataLoader, device: torch.device) -> dict[str, float | int]:
    model.eval()

    y_true: list[int] = []
    y_pred: list[int] = []

    for batch in dataloader:
        texts = [str(text or "") for text in batch["text"]]
        labels = [int(x) for x in batch["overall_label"]]

        outputs = model(texts=texts)
        preds = outputs["predictions"].detach().cpu().tolist()

        y_true.extend(labels)
        y_pred.extend(int(p) for p in preds)

    return binary_classification_metrics(y_true=y_true, y_pred=y_pred)


def _split_train_eval(
    dataset: WeiboStructuralDataset,
    eval_ratio: float,
    seed: int,
) -> tuple[WeiboStructuralDataset, WeiboStructuralDataset]:
    eval_size = max(1, int(len(dataset) * eval_ratio))
    train_size = len(dataset) - eval_size
    if train_size <= 0:
        raise ValueError("Not enough samples to split train/eval. Provide separate eval JSONL or more data.")

    train_subset, eval_subset = random_split(
        dataset,
        lengths=[train_size, eval_size],
        generator=torch.Generator().manual_seed(seed),
    )

    train_records = [train_subset.dataset[idx] for idx in train_subset.indices]
    eval_records = [eval_subset.dataset[idx] for idx in eval_subset.indices]
    return WeiboStructuralDataset(train_records), WeiboStructuralDataset(eval_records)


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    train_dataset_full = WeiboStructuralDataset.from_jsonl(args.train_jsonl)

    if args.eval_jsonl:
        train_dataset = train_dataset_full
        eval_dataset = WeiboStructuralDataset.from_jsonl(args.eval_jsonl)
    else:
        train_dataset, eval_dataset = _split_train_eval(train_dataset_full, args.eval_ratio, args.seed)

    train_loader = _build_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_loader = _build_dataloader(eval_dataset, batch_size=args.batch_size, shuffle=False)

    model = FullModel(
        vocab_size=args.vocab_size,
        input_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        view_dim=args.view_dim,
        consistency_dim=args.consistency_dim,
        dropout=args.dropout,
        num_classes=2,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_metrics: dict[str, float | int] | None = None
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        metrics = evaluate(model, eval_loader, device)
        metrics["train_loss"] = train_loss
        metrics["epoch"] = epoch

        if best_metrics is None or float(metrics["accuracy"]) >= float(best_metrics["accuracy"]):
            best_metrics = dict(metrics)
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        print(
            "[epoch={epoch}] loss={loss:.4f} acc={acc:.4f} f1_fake={f1:.4f}".format(
                epoch=epoch,
                loss=train_loss,
                acc=float(metrics["accuracy"]),
                f1=float(metrics["f1_fake"]),
            )
        )

    assert best_metrics is not None
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = out_dir / "full_model_best.pt"
    torch.save(
        {
            "model_state_dict": best_state,
            "config": model.config.__dict__,
            "best_metrics": best_metrics,
            "train_args": vars(args),
        },
        ckpt_path,
    )

    metrics_path = out_dir / "full_metrics.json"
    metrics_path.write_text(json.dumps(best_metrics, indent=2), encoding="utf-8")

    return {
        "best_metrics": best_metrics,
        "checkpoint": str(ckpt_path),
        "metrics_json": str(metrics_path),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train SNF-MV full model with consistency reasoning on Weibo JSONL")
    parser.add_argument("--train-jsonl", required=True, help="Path to training JSONL file")
    parser.add_argument("--eval-jsonl", default=None, help="Optional path to eval JSONL file")
    parser.add_argument("--output-dir", default="outputs/full_stage1", help="Output directory")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--view-dim", type=int, default=64)
    parser.add_argument("--consistency-dim", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=50000)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--eval-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    outputs = run_training(args)
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
