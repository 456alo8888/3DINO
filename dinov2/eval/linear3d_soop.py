# Author: GitHub Copilot

import argparse
import json
import logging
import math
import os
import sys
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from tqdm.auto import tqdm

import dinov2.distributed as distributed
from dinov2.data import SamplerType, make_classification_dataset_3d, make_data_loader
from dinov2.data.transforms import make_classification_transform_3d
from dinov2.eval.linear3d import create_linear_input
from dinov2.eval.setup import get_args_parser as get_setup_args_parser
from dinov2.eval.setup import setup_and_build_model_3d
from dinov2.eval.utils import ModelWithIntermediateLayers


logger = logging.getLogger("dinov2")
OUTCOME_REGRESSION_TARGETS = {"nihss", "gs_rankin_6isdeath", "gs_rankin+6isdeath"}


def get_args_parser(
    description: Optional[str] = None,
    parents: Optional[List[argparse.ArgumentParser]] = None,
    add_help: bool = True,
):
    parents = parents or []
    setup_args_parser = get_setup_args_parser(parents=parents, add_help=False)
    parents = [setup_args_parser]
    parser = argparse.ArgumentParser(
        description=description,
        parents=parents,
        add_help=add_help,
    )
    parser.add_argument("--dataset-name", type=str, default="SOOP")
    parser.add_argument("--base-data-dir", type=str, required=True)
    parser.add_argument("--cache-dir", type=str, required=True)
    parser.add_argument("--target-col", type=str, default="gs_rankin_6isdeath")
    parser.add_argument("--task-type", type=str, choices=["binary", "multiclass", "regression"], default="regression")
    parser.add_argument("--allow-classification-outcome", action="store_true")
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--drop-missing-labels", action="store_true")
    parser.add_argument("--use-tabular", action="store_true")
    parser.add_argument("--dataset-percent", type=int, default=100)
    parser.add_argument("--dataset-seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--epoch-length", type=int, default=125)
    parser.add_argument("--eval-period-iterations", type=int, default=125)
    parser.add_argument("--save-checkpoint-frequency", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--optimizer", type=str, choices=["auto", "sgd", "adamw"], default="auto")
    parser.add_argument("--sgd-momentum", type=float, default=0.9)
    parser.add_argument("--adamw-beta1", type=float, default=0.9)
    parser.add_argument("--adamw-beta2", type=float, default=0.999)
    parser.add_argument("--adamw-eps", type=float, default=1e-8)
    parser.add_argument("--normalize-features", action="store_true")
    parser.add_argument("--grad-clip-norm", type=float, default=0.0)
    parser.add_argument("--diagnostics-period", type=int, default=10)
    parser.add_argument("--n-last-blocks", type=int, default=4)
    parser.add_argument("--use-avgpool", action="store_true")
    parser.add_argument("--no-resume", action="store_true")

    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="3dino-soop-outcome")
    parser.add_argument("--wandb-entity", type=str, default="")
    parser.add_argument("--wandb-run-name", type=str, default="")
    parser.add_argument("--wandb-tags", type=str, default="")
    parser.add_argument("--wandb-mode", type=str, choices=["online", "offline", "disabled"], default="online")
    return parser


class SOOPLinearHead(nn.Module):
    def __init__(self, img_dim: int, tabular_dim: int, out_dim: int, use_tabular: bool):
        super().__init__()
        self.use_tabular = use_tabular
        in_dim = img_dim + (tabular_dim if use_tabular else 0)
        self.linear = nn.Linear(in_dim, out_dim)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, img_feat: torch.Tensor, tabular: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.use_tabular:
            if tabular is None:
                raise ValueError("Tabular tensor is required when use_tabular=True")
            feat = torch.cat([img_feat, tabular.float()], dim=-1)
        else:
            feat = img_feat
        return self.linear(feat)


@dataclass
class EvalStats:
    metric_name: str
    metric_value: float
    sample_count: int
    pred_records: List[dict]
    metrics: Dict[str, float]


def _normalize_features(features: torch.Tensor, enabled: bool) -> torch.Tensor:
    if not enabled:
        return features
    return nn.functional.normalize(features, dim=-1, p=2)


def _resolve_optimizer_name(task_type: str, optimizer: str) -> str:
    if optimizer != "auto":
        return optimizer
    return "adamw" if task_type == "regression" else "sgd"


def _build_optimizer(
    head: nn.Module,
    task_type: str,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float,
    sgd_momentum: float,
    adamw_betas: Tuple[float, float],
    adamw_eps: float,
):
    resolved = _resolve_optimizer_name(task_type, optimizer_name)
    if resolved == "adamw":
        optimizer = torch.optim.AdamW(
            head.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=adamw_betas,
            eps=adamw_eps,
        )
    else:
        optimizer = torch.optim.SGD(
            head.parameters(),
            lr=learning_rate,
            momentum=sgd_momentum,
            weight_decay=weight_decay,
        )
    return optimizer, resolved


def _compute_grad_norm(parameters) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad_norm = float(parameter.grad.detach().float().norm().item())
        total += grad_norm * grad_norm
    return math.sqrt(total)


def _filter_valid_batch(batch: dict, drop_missing_labels: bool) -> dict:
    if "label_mask" not in batch:
        return batch

    if drop_missing_labels:
        valid = batch["label_mask"] > 0
    else:
        valid = torch.ones_like(batch["label_mask"]).bool()

    if valid.sum() == 0:
        return {"empty": True}

    filtered = {
        "image": batch["image"][valid],
        "label": batch["label"][valid],
    }
    if "tabular" in batch:
        filtered["tabular"] = batch["tabular"][valid]
    if "subject_id" in batch:
        subject_ids = batch["subject_id"]
        if isinstance(subject_ids, list):
            valid_idx = valid.nonzero(as_tuple=False).flatten().tolist()
            filtered["subject_id"] = [subject_ids[i] for i in valid_idx]
    return filtered


def _compute_loss(task_type: str, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if task_type == "regression":
        return nn.functional.mse_loss(logits.squeeze(-1), labels.float())
    return nn.functional.cross_entropy(logits, labels.long())


def _prepare_classification_labels(labels: torch.Tensor, task_type: str, num_classes: int) -> torch.Tensor:
    if task_type == "regression":
        return labels.float()

    labels = labels.float()
    if task_type == "binary":
        return (labels > 0).long()

    labels = torch.round(labels).long()
    labels = torch.clamp(labels, min=0, max=max(0, num_classes - 1))
    return labels


def _compute_regression_metrics(preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-8) -> Dict[str, float]:
    preds = preds.float()
    targets = targets.float()

    error = preds - targets
    mse = torch.mean(error ** 2)
    rmse = torch.sqrt(torch.clamp(mse, min=0.0))
    mae = torch.mean(torch.abs(error))
    mape = torch.mean(torch.abs(error) / torch.clamp(torch.abs(targets), min=eps)) * 100.0

    ss_res = torch.sum(error ** 2)
    target_mean = torch.mean(targets)
    ss_tot = torch.sum((targets - target_mean) ** 2)
    r2 = 1.0 - (ss_res / torch.clamp(ss_tot, min=eps))

    return {
        "mse": float(mse.item()),
        "rmse": float(rmse.item()),
        "mae": float(mae.item()),
        "mape": float(mape.item()),
        "r2": float(r2.item()),
    }


def evaluate_model(
    feature_model,
    head,
    data_loader,
    task_type: str,
    drop_missing_labels: bool,
    n_last_blocks: int,
    use_avgpool: bool,
    device,
    num_classes: int,
    normalize_features: bool,
):
    feature_model.eval()
    head.eval()

    all_preds = []
    all_targets = []
    pred_records = []
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for batch in data_loader:
            filtered = _filter_valid_batch(batch, drop_missing_labels=drop_missing_labels)
            if "empty" in filtered:
                continue

            images = filtered["image"].to(device, non_blocking=True)
            labels = filtered["label"].to(device, non_blocking=True)
            labels = _prepare_classification_labels(labels, task_type, num_classes)
            tabular = filtered.get("tabular")
            if tabular is not None:
                tabular = tabular.to(device, non_blocking=True)

            token_features = feature_model(images)
            img_feat = create_linear_input(token_features, n_last_blocks, use_avgpool)
            img_feat = _normalize_features(img_feat, normalize_features)
            logits = head(img_feat, tabular)

            batch_loss = _compute_loss(task_type, logits, labels)

            if task_type == "regression":
                preds = logits.squeeze(-1)
            else:
                preds = torch.argmax(logits, dim=1)

            batch_count = int(labels.numel())
            total_loss += float(batch_loss.item()) * batch_count
            total_count += batch_count

            all_preds.append(preds.detach().cpu())
            all_targets.append(labels.detach().cpu())

            if "subject_id" in filtered:
                for sid, pred, target in zip(filtered["subject_id"], preds.detach().cpu(), labels.detach().cpu()):
                    pred_records.append({
                        "subject_id": sid,
                        "pred": float(pred.item()),
                        "target": float(target.item()),
                    })

    if not all_preds:
        empty_metrics = {"loss": 0.0}
        if task_type == "regression":
            empty_metrics.update({"mse": 0.0, "rmse": 0.0, "mae": 0.0, "mape": 0.0, "r2": 0.0})
            return EvalStats("mse", 0.0, 0, pred_records, empty_metrics)
        empty_metrics.update({"accuracy": 0.0})
        return EvalStats("accuracy", 0.0, 0, pred_records, empty_metrics)

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    avg_loss = total_loss / max(1, total_count)

    if task_type == "regression":
        metrics = _compute_regression_metrics(preds, targets)
        metrics["loss"] = float(avg_loss)
        metric_name = "mse"
        metric_value = metrics[metric_name]
    else:
        acc = (preds == targets.long()).float().mean().item()
        metrics = {"accuracy": float(acc), "loss": float(avg_loss)}
        metric_name = "accuracy"
        metric_value = metrics[metric_name]

    return EvalStats(metric_name, metric_value, int(targets.numel()), pred_records, metrics)


def _build_wandb_payload(prefix: str, metrics: Dict[str, float], samples: int, iteration: Optional[int] = None) -> Dict[str, float]:
    payload: Dict[str, float] = {f"{prefix}/{k}": float(v) for k, v in metrics.items()}
    payload[f"{prefix}/samples"] = float(samples)
    if iteration is not None:
        payload["iteration"] = float(iteration)
    return payload


def run_eval_soop(
    model,
    autocast_dtype,
    output_dir,
    dataset_name,
    dataset_percent,
    base_data_dir,
    batch_size,
    data_cache_path,
    image_size,
    epochs,
    epoch_length,
    num_workers,
    save_checkpoint_frequency,
    eval_period_iterations,
    learning_rate,
    weight_decay,
    target_col,
    task_type,
    num_classes,
    drop_missing_labels,
    use_tabular,
    dataset_seed,
    optimizer_name,
    sgd_momentum,
    adamw_beta1,
    adamw_beta2,
    adamw_eps,
    normalize_features,
    grad_clip_norm,
    diagnostics_period,
    strict_pretrained,
    n_last_blocks,
    use_avgpool,
    wandb_run=None,
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda", torch.cuda.current_device())

    train_transform, val_transform = make_classification_transform_3d(dataset_name, image_size, min_int=-1.0)
    train_dataset, val_dataset, test_dataset, loader_num_classes = make_classification_dataset_3d(
        dataset_name=dataset_name,
        dataset_percent=dataset_percent,
        base_directory=base_data_dir,
        train_transforms=train_transform,
        val_transforms=val_transform,
        cache_path=data_cache_path,
        dataset_seed=dataset_seed,
        target_col=target_col,
        include_tabular=use_tabular,
        drop_missing_labels=False,
    )

    if task_type != "regression" and num_classes <= 1:
        num_classes = loader_num_classes

    for parameter in model.parameters():
        parameter.requires_grad = False

    backbone_trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if backbone_trainable_params != 0:
        raise RuntimeError(
            f"Backbone must be frozen in linear SOOP eval. Found {backbone_trainable_params} trainable params."
        )

    autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=autocast_dtype)
    feature_model = ModelWithIntermediateLayers(model, n_last_blocks, autocast_ctx)

    sample = train_dataset[0]
    sample_features = feature_model(sample["image"].unsqueeze(0).to(device))
    img_dim = create_linear_input(sample_features, n_last_blocks, use_avgpool).shape[-1]
    tabular_dim = int(sample["tabular"].numel()) if use_tabular and "tabular" in sample else 0

    out_dim = 1 if task_type == "regression" else num_classes
    head = SOOPLinearHead(img_dim=img_dim, tabular_dim=tabular_dim, out_dim=out_dim, use_tabular=use_tabular).to(device)

    optimizer, resolved_optimizer = _build_optimizer(
        head=head,
        task_type=task_type,
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        sgd_momentum=sgd_momentum,
        adamw_betas=(adamw_beta1, adamw_beta2),
        adamw_eps=adamw_eps,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * epoch_length, eta_min=0)

    head_trainable_params = sum(parameter.numel() for parameter in head.parameters() if parameter.requires_grad)
    if distributed.is_main_process():
        logger.info(
            f"optimizer={resolved_optimizer} lr={learning_rate} wd={weight_decay} "
            f"normalize_features={normalize_features} grad_clip_norm={grad_clip_norm} "
            f"strict_pretrained={strict_pretrained} backbone_trainable={backbone_trainable_params} "
            f"head_trainable={head_trainable_params}"
        )

    train_loader = make_data_loader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        seed=0,
        sampler_type=SamplerType.SHARDED_INFINITE,
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )
    val_loader = make_data_loader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=False,
        sampler_type=SamplerType.DISTRIBUTED,
        drop_last=False,
        persistent_workers=False,
    )
    test_loader = make_data_loader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=False,
        sampler_type=SamplerType.DISTRIBUTED,
        drop_last=False,
        persistent_workers=False,
    )

    train_iter = iter(train_loader)
    max_iter = epochs * epoch_length
    best_val = math.inf if task_type == "regression" else -math.inf
    best_path = os.path.join(output_dir, "best_val_soop.pth")

    train_loss_history: List[float] = []

    if distributed.is_main_process():
        iter_range = tqdm(range(max_iter), desc="train", dynamic_ncols=True)
    else:
        iter_range = range(max_iter)

    for iteration in iter_range:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        filtered = _filter_valid_batch(batch, drop_missing_labels=drop_missing_labels)
        if "empty" in filtered:
            continue

        images = filtered["image"].to(device, non_blocking=True)
        labels = filtered["label"].to(device, non_blocking=True)
        labels = _prepare_classification_labels(labels, task_type, num_classes)
        tabular = filtered.get("tabular")
        if tabular is not None:
            tabular = tabular.to(device, non_blocking=True)

        token_features = feature_model(images)
        img_feat = create_linear_input(token_features, n_last_blocks, use_avgpool)
        img_feat = _normalize_features(img_feat, normalize_features)
        logits = head(img_feat, tabular)

        loss = _compute_loss(task_type, logits, labels)
        train_loss_history.append(float(loss.item()))

        optimizer.zero_grad()
        loss.backward()

        grad_norm = _compute_grad_norm(head.parameters())
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(head.parameters(), grad_clip_norm)

        optimizer.step()
        scheduler.step()

        if iteration % max(1, diagnostics_period) == 0 and distributed.is_main_process():
            feature_absmax = float(img_feat.detach().abs().max().item())
            feature_norm_mean = float(img_feat.detach().norm(dim=-1).mean().item())
            pred_absmax = float(logits.detach().abs().max().item())
            logger.info(
                f"iter={iteration} loss={loss.item():.6f} lr={optimizer.param_groups[0]['lr']:.6g} "
                f"grad_norm={grad_norm:.6f} feature_absmax={feature_absmax:.6f} "
                f"feature_norm_mean={feature_norm_mean:.6f} pred_absmax={pred_absmax:.6f}"
            )
            if hasattr(iter_range, "set_postfix"):
                iter_range.set_postfix(
                    loss=f"{loss.item():.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                    grad=f"{grad_norm:.2e}",
                )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "iteration": float(iteration),
                        "train/loss": float(loss.item()),
                        "train/lr": float(optimizer.param_groups[0]["lr"]),
                        "train/grad_norm": float(grad_norm),
                        "train/feature_absmax": feature_absmax,
                        "train/feature_norm_mean": feature_norm_mean,
                        "train/pred_absmax": pred_absmax,
                    }
                )

        should_eval = (iteration + 1) % eval_period_iterations == 0 or (iteration + 1) == max_iter
        if should_eval:
            val_stats = evaluate_model(
                feature_model=feature_model,
                head=head,
                data_loader=val_loader,
                task_type=task_type,
                drop_missing_labels=drop_missing_labels,
                n_last_blocks=n_last_blocks,
                use_avgpool=use_avgpool,
                device=device,
                num_classes=num_classes,
                normalize_features=normalize_features,
            )
            if distributed.is_main_process():
                logger.info(f"val_metrics={json.dumps(val_stats.metrics, sort_keys=True)} samples={val_stats.sample_count}")
                if wandb_run is not None:
                    wandb_run.log(_build_wandb_payload("val", val_stats.metrics, val_stats.sample_count, iteration + 1))

            is_better = val_stats.metric_value < best_val if task_type == "regression" else val_stats.metric_value > best_val
            if is_better:
                best_val = val_stats.metric_value
                if distributed.is_main_process():
                    torch.save({"head": head.state_dict(), "iter": iteration}, best_path)

        if (iteration + 1) % (save_checkpoint_frequency * epoch_length) == 0 and distributed.is_main_process():
            ckpt_path = os.path.join(output_dir, f"checkpoint_iter_{iteration + 1}.pth")
            torch.save({"head": head.state_dict(), "iter": iteration}, ckpt_path)

    if os.path.exists(best_path):
        payload = torch.load(best_path, map_location="cpu")
        head.load_state_dict(payload["head"])

    val_stats = evaluate_model(
        feature_model=feature_model,
        head=head,
        data_loader=val_loader,
        task_type=task_type,
        drop_missing_labels=drop_missing_labels,
        n_last_blocks=n_last_blocks,
        use_avgpool=use_avgpool,
        device=device,
        num_classes=num_classes,
        normalize_features=normalize_features,
    )
    test_stats = evaluate_model(
        feature_model=feature_model,
        head=head,
        data_loader=test_loader,
        task_type=task_type,
        drop_missing_labels=drop_missing_labels,
        n_last_blocks=n_last_blocks,
        use_avgpool=use_avgpool,
        device=device,
        num_classes=num_classes,
        normalize_features=normalize_features,
    )

    train_loss_mean = float(sum(train_loss_history) / max(1, len(train_loss_history)))
    train_loss_last = float(train_loss_history[-1]) if train_loss_history else 0.0

    mode_name = "image_tabular" if use_tabular else "image_only"
    selection_metric = "mse" if task_type == "regression" else "accuracy"
    results = {
        "dataset": dataset_name,
        "target_col": target_col,
        "task_type": task_type,
        "mode": mode_name,
        "tabular_dim": tabular_dim,
        "selection_metric": selection_metric,
        "training_config": {
            "optimizer": resolved_optimizer,
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "sgd_momentum": float(sgd_momentum),
            "adamw_beta1": float(adamw_beta1),
            "adamw_beta2": float(adamw_beta2),
            "adamw_eps": float(adamw_eps),
            "normalize_features": bool(normalize_features),
            "grad_clip_norm": float(grad_clip_norm),
            "diagnostics_period": int(diagnostics_period),
            "strict_pretrained": bool(strict_pretrained),
        },
        "train_metrics": {
            "loss_last": train_loss_last,
            "loss_mean": train_loss_mean,
            "iterations": int(len(train_loss_history)),
        },
        "val_metric": {
            "name": val_stats.metric_name,
            "value": val_stats.metric_value,
            "samples": val_stats.sample_count,
            "metrics": val_stats.metrics,
        },
        "test_metric": {
            "name": test_stats.metric_name,
            "value": test_stats.metric_value,
            "samples": test_stats.sample_count,
            "metrics": test_stats.metrics,
        },
        "test_pred_dict": test_stats.pred_records,
    }

    result_path = os.path.join(output_dir, "results_eval_linear_soop.json")
    if distributed.is_main_process():
        with open(result_path, "w") as f:
            json.dump(results, f, indent=2)

        if wandb_run is not None:
            wandb_run.log(_build_wandb_payload("test", test_stats.metrics, test_stats.sample_count, max_iter))
            wandb_run.log({"train/loss_mean": train_loss_mean, "train/loss_last": train_loss_last})

    logger.info(f"SOOP evaluation finished. Results written to: {result_path}")
    return results


def _build_wandb_tags(raw: str) -> List[str]:
    if not raw:
        return []
    return [tag.strip() for tag in raw.split(",") if tag.strip()]


def _maybe_init_wandb(args) -> Optional[Any]:
    if not args.use_wandb:
        return None
    if not distributed.is_main_process():
        return None

    try:
        import wandb
    except ImportError as exc:
        raise ImportError("wandb is required when --use-wandb is enabled. Please install wandb.") from exc

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_run_name or None,
        mode=args.wandb_mode,
        tags=_build_wandb_tags(args.wandb_tags),
        config={
            "dataset_name": args.dataset_name,
            "target_col": args.target_col,
            "task_type": args.task_type,
            "use_tabular": args.use_tabular,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "epoch_length": args.epoch_length,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "optimizer": _resolve_optimizer_name(args.task_type, args.optimizer),
            "normalize_features": args.normalize_features,
            "grad_clip_norm": args.grad_clip_norm,
            "diagnostics_period": args.diagnostics_period,
            "strict_pretrained": args.strict_pretrained,
        },
    )
    return run


def main(args):
    if args.target_col in OUTCOME_REGRESSION_TARGETS and args.task_type != "regression" and not args.allow_classification_outcome:
        raise ValueError(
            "Outcome targets nihss/gs_rankin_6isdeath are configured as regression-first. "
            "Use --task-type regression or pass --allow-classification-outcome to override."
        )

    wandb_run = _maybe_init_wandb(args)
    try:
        model, autocast_dtype = setup_and_build_model_3d(args)
        run_eval_soop(
            model=model,
            autocast_dtype=autocast_dtype,
            output_dir=args.output_dir,
            dataset_name=args.dataset_name,
            dataset_percent=args.dataset_percent,
            base_data_dir=args.base_data_dir,
            batch_size=args.batch_size,
            data_cache_path=args.cache_dir,
            image_size=args.image_size,
            epochs=args.epochs,
            epoch_length=args.epoch_length,
            num_workers=args.num_workers,
            save_checkpoint_frequency=args.save_checkpoint_frequency,
            eval_period_iterations=args.eval_period_iterations,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            target_col=args.target_col,
            task_type=args.task_type,
            num_classes=args.num_classes,
            drop_missing_labels=args.drop_missing_labels,
            use_tabular=args.use_tabular,
            dataset_seed=args.dataset_seed,
            optimizer_name=args.optimizer,
            sgd_momentum=args.sgd_momentum,
            adamw_beta1=args.adamw_beta1,
            adamw_beta2=args.adamw_beta2,
            adamw_eps=args.adamw_eps,
            normalize_features=args.normalize_features,
            grad_clip_norm=args.grad_clip_norm,
            diagnostics_period=args.diagnostics_period,
            strict_pretrained=args.strict_pretrained,
            n_last_blocks=args.n_last_blocks,
            use_avgpool=args.use_avgpool,
            wandb_run=wandb_run,
        )
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    return 0


if __name__ == "__main__":
    description = "DINOv2 3d linear evaluation for SOOP stroke outcomes"
    args_parser = get_args_parser(description=description)
    args = args_parser.parse_args()
    sys.exit(main(args))
