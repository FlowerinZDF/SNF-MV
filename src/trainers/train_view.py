"""Training entry for first runnable Weibo view-aware fake/real model stage."""

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
from src.models.view_model import ViewModel
from src.models.view_extractor import VIEW_NAMES
from src.trainers.multimodal import build_model_inputs
from src.utils.seed import set_seed

VIEW_LABEL_FIELDS = {
    "subject": "subject_label",
    "event": "event_label",
    "scene": "scene_label",
    "time": "time_label",
}


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


def _prepare_optional_view_targets(
    batch: dict[str, list[Any]],
    *,
    device: torch.device,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Prepare per-view targets with validity masks.

    Returns mapping: view -> (labels, valid_mask), where labels are `long`
    tensors on `device` and `valid_mask` marks samples with usable labels.
    Missing/null/non-castable labels are masked out.
    """
    batch_size = len(batch.get("overall_label", []))
    targets: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    for view in VIEW_NAMES:
        field = VIEW_LABEL_FIELDS[view]
        raw_values = batch.get(field, [None] * batch_size)
        if len(raw_values) != batch_size:
            raw_values = list(raw_values)[:batch_size] + [None] * max(0, batch_size - len(raw_values))

        labels = torch.zeros(batch_size, dtype=torch.long, device=device)
        valid_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        for idx, value in enumerate(raw_values):
            if value is None:
                continue
            try:
                labels[idx] = int(value)
                valid_mask[idx] = True
            except (TypeError, ValueError):
                continue

        targets[view] = (labels, valid_mask)

    return targets


def _compute_aux_view_loss(
    outputs: dict[str, Any],
    view_targets: dict[str, tuple[torch.Tensor, torch.Tensor]],
    *,
    criterion: nn.Module,
) -> torch.Tensor:
    """Compute mean auxiliary loss over available per-view labels."""
    per_view_logits = outputs.get("per_view_logits", {})
    if not isinstance(per_view_logits, dict):
        raise ValueError("View model output must include per_view_logits for view supervision.")

    losses: list[torch.Tensor] = []
    for view in VIEW_NAMES:
        labels, valid_mask = view_targets[view]
        if not torch.any(valid_mask):
            continue
        view_logits = per_view_logits[view][valid_mask]
        view_labels = labels[valid_mask]
        losses.append(criterion(view_logits, view_labels))

    if not losses:
        return outputs["logits"].new_zeros(())
    return torch.stack(losses).mean()


def train_one_epoch(
    model: ViewModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    *,
    image_key: str | None,
    enable_images: bool,
    aux_view_loss_weight: float = 0.0,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_aux_loss = 0.0
    seen = 0

    for batch in dataloader:
        labels = torch.tensor(batch["overall_label"], dtype=torch.long, device=device)
        view_targets = _prepare_optional_view_targets(batch, device=device)
        model_inputs = build_model_inputs(
            model,
            batch,
            device=device,
            image_key=image_key,
            enable_images=enable_images,
        )

        optimizer.zero_grad(set_to_none=True)
        outputs = model(**model_inputs)

        overall_loss = criterion(outputs["logits"], labels)
        aux_loss = _compute_aux_view_loss(outputs, view_targets, criterion=criterion)
        loss = overall_loss + aux_view_loss_weight * aux_loss
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += float(loss.item()) * batch_size
        running_aux_loss += float(aux_loss.item()) * batch_size
        seen += batch_size

    return running_loss / max(1, seen), running_aux_loss / max(1, seen)


@torch.no_grad()
def evaluate(
    model: ViewModel,
    dataloader: DataLoader,
    device: torch.device,
    *,
    image_key: str | None,
    enable_images: bool,
) -> dict[str, float | int]:
    model.eval()

    y_true: list[int] = []
    y_pred: list[int] = []

    for batch in dataloader:
        labels = [int(x) for x in batch["overall_label"]]
        model_inputs = build_model_inputs(
            model,
            batch,
            device=device,
            image_key=image_key,
            enable_images=enable_images,
        )

        outputs = model(**model_inputs)
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
    if not hasattr(args, "enable_images"):
        args.enable_images = not bool(getattr(args, "disable_images", False))
    if not hasattr(args, "image_key"):
        args.image_key = None
    if not hasattr(args, "aux_view_loss_weight"):
        args.aux_view_loss_weight = 0.0

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    dataset_kwargs = {"load_image_tensors": args.enable_images}
    try:
        train_dataset_full = WeiboStructuralDataset.from_jsonl(args.train_jsonl, **dataset_kwargs)
    except TypeError:
        train_dataset_full = WeiboStructuralDataset.from_jsonl(args.train_jsonl)

    if args.eval_jsonl:
        train_dataset = train_dataset_full
        try:
            eval_dataset = WeiboStructuralDataset.from_jsonl(args.eval_jsonl, **dataset_kwargs)
        except TypeError:
            eval_dataset = WeiboStructuralDataset.from_jsonl(args.eval_jsonl)
    else:
        train_dataset, eval_dataset = _split_train_eval(train_dataset_full, args.eval_ratio, args.seed)

    train_loader = _build_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_loader = _build_dataloader(eval_dataset, batch_size=args.batch_size, shuffle=False)

    model = ViewModel(
        vocab_size=args.vocab_size,
        input_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        view_dim=args.view_dim,
        dropout=args.dropout,
        num_classes=2,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_metrics: dict[str, float | int] | None = None
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss, train_aux_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            image_key=args.image_key,
            enable_images=args.enable_images,
            aux_view_loss_weight=args.aux_view_loss_weight,
        )
        metrics = evaluate(
            model,
            eval_loader,
            device,
            image_key=args.image_key,
            enable_images=args.enable_images,
        )
        metrics["train_loss"] = train_loss
        metrics["train_aux_view_loss"] = train_aux_loss
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

    ckpt_path = out_dir / "view_model_best.pt"
    torch.save(
        {
            "model_state_dict": best_state,
            "config": model.config.__dict__,
            "best_metrics": best_metrics,
            "train_args": vars(args),
        },
        ckpt_path,
    )

    metrics_path = out_dir / "view_metrics.json"
    metrics_path.write_text(json.dumps(best_metrics, indent=2), encoding="utf-8")

    return {
        "best_metrics": best_metrics,
        "checkpoint": str(ckpt_path),
        "metrics_json": str(metrics_path),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train first-stage view-aware fake/real model on Weibo JSONL")
    parser.add_argument("--train-jsonl", required=True, help="Path to training JSONL file")
    parser.add_argument("--eval-jsonl", default=None, help="Optional path to eval JSONL file")
    parser.add_argument("--output-dir", default="outputs/view_stage1", help="Output directory")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--view-dim", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=50000)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--aux-view-loss-weight",
        type=float,
        default=0.0,
        help="Weight for optional auxiliary per-view supervision loss (0 keeps overall-only training).",
    )
    parser.add_argument("--eval-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--disable-images",
        action="store_true",
        help="Disable image tensor usage even when image tensors are available in the dataset.",
    )
    parser.add_argument(
        "--image-key",
        default=None,
        help="Optional batch key for image tensors (auto-detect when omitted).",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.enable_images = not args.disable_images
    outputs = run_training(args)
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
