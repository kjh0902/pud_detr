#!/usr/bin/env python3
"""Train and evaluate the historical PUD-DETR implementation.

This script consolidates the standalone experiment files in
``pu_pascal_07_tuning`` and ``pu_pascal_07_tuning_with_rewrite`` into one
reproducible entry point.  One invocation represents one method, one training
annotation file, and one random seed.

The implementation intentionally preserves the historical PU target mask:
every zero entry in the one-hot class target contributes to the unlabeled
negative-risk term.  ``weight_p`` is the scaling factor alpha described in the
paper.  The positive-as-negative focal term uses the corrected modulation from
``pu_pascal_07_tuning_with_rewrite/train.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from scripts.gpu_selection import (
    DEFAULT_DETERMINISTIC,
    concrete_cuda_index,
    configure_cuda_visibility,
    lightning_deterministic_setting,
    normalize_device_argument,
)

# Lightning imports torchmetrics, which imports matplotlib.  Keep its cache out
# of a potentially read-only home directory in managed or containerized runs.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/pud_detr_matplotlib")

# A concrete --device value is applied before importing torch or Lightning.
# CUDA remaps that one physical GPU to logical cuda:0 inside this process.
configure_cuda_visibility(sys.argv[1:])

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torchvision.ops.boxes import box_area
from tqdm.auto import tqdm
from transformers import (
    DeformableDetrConfig,
    DeformableDetrForObjectDetection,
    DeformableDetrImageProcessor,
)


COCO_METRIC_NAMES = (
    "AP",
    "AP50",
    "AP75",
    "AP_small",
    "AP_medium",
    "AP_large",
    "AR1",
    "AR10",
    "AR100",
    "AR_small",
    "AR_medium",
    "AR_large",
)


@dataclass(frozen=True)
class AnnotationSummary:
    path: str
    image_count: int
    annotation_count: int
    category_ids: tuple[int, ...]
    image_files: frozenset[str]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a PN Deformable DETR baseline or PUD-DETR and evaluate the "
            "best-validation checkpoint with COCO metrics."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    experiment = parser.add_argument_group("experiment")
    experiment.add_argument("--method", choices=("pn", "pud"), default="pud")
    experiment.add_argument(
        "--weight-p",
        type=float,
        default=5.0,
        help="PU positive-risk scale; this is alpha in the paper.",
    )
    experiment.add_argument(
        "--reduction",
        choices=("global", "query_wise", "element_wise"),
        default="global",
        help="Non-negative correction granularity used only for method=pud.",
    )
    experiment.add_argument("--drop-ratio", type=float, default=None)
    experiment.add_argument("--nominal-drop-probability", type=float, default=None)
    experiment.add_argument("--seed", type=int, default=42)
    experiment.add_argument("--experiment-name", required=True)
    experiment.add_argument("--output-dir", type=Path, default=Path("outputs"))
    experiment.add_argument(
        "--skip-test",
        action="store_true",
        help="Stop after validation-based checkpoint selection (for tuning).",
    )

    data = parser.add_argument_group("data")
    data.add_argument("--train-json", type=Path, required=True)
    data.add_argument("--val-json", type=Path, required=True)
    data.add_argument("--test-json", type=Path)
    data.add_argument("--trainval-image-dir", type=Path, required=True)
    data.add_argument("--test-image-dir", type=Path)
    data.add_argument("--num-workers", type=int, default=6)
    data.add_argument("--prefetch-factor", type=int, default=1)

    model = parser.add_argument_group("model")
    model.add_argument(
        "--hf-checkpoint", default="SenseTime/deformable-detr"
    )
    model.add_argument(
        "--hf-revision",
        default=None,
        help="Pinned Hugging Face model revision for reproducible loading.",
    )
    model.add_argument("--local-files-only", action="store_true")
    model.add_argument("--focal-alpha", type=float, default=0.25)
    model.add_argument("--focal-gamma", type=float, default=2.0)

    training = parser.add_argument_group("training")
    training.add_argument("--epochs", type=int, default=20)
    training.add_argument("--batch-size", type=int, default=4)
    training.add_argument("--lr", type=float, default=1e-4)
    training.add_argument("--lr-backbone", type=float, default=1e-5)
    training.add_argument("--weight-decay", type=float, default=1e-4)
    training.add_argument("--gradient-clip", type=float, default=0.1)
    training.add_argument("--warmup-steps", type=int, default=2_000)
    training.add_argument(
        "--device",
        default="auto",
        help=(
            "auto, cpu, cuda, or a physical GPU ID such as 0 or 1. "
            "The legacy cuda:<ID> form is also accepted."
        ),
    )
    training.add_argument(
        "--precision",
        default="32-true",
        help="PyTorch Lightning precision setting.",
    )
    training.add_argument("--log-every-n-steps", type=int, default=50)
    training.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_DETERMINISTIC,
        help=(
            "Enable best-effort deterministic algorithms. CUDA operations "
            "without deterministic kernels emit warnings instead of stopping "
            "training; disabled by default for Deformable DETR."
        ),
    )

    args = parser.parse_args(argv)
    validate_args(parser, args)
    return args


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not args.skip_test and (args.test_json is None or args.test_image_dir is None):
        parser.error(
            "--test-json and --test-image-dir are required unless --skip-test is set"
        )
    if args.weight_p <= 0:
        parser.error("--weight-p must be positive")
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    if args.num_workers > 0 and args.prefetch_factor <= 0:
        parser.error("--prefetch-factor must be positive when workers are used")
    if args.warmup_steps < 0:
        parser.error("--warmup-steps cannot be negative")
    if args.lr <= 0 or args.lr_backbone <= 0:
        parser.error("learning rates must be positive")
    if not 0 <= args.focal_alpha <= 1:
        parser.error("--focal-alpha must be in [0, 1]")
    if args.focal_gamma < 0:
        parser.error("--focal-gamma cannot be negative")
    try:
        args.device = normalize_device_argument(args.device)
    except ValueError as exc:
        parser.error(str(exc))


def resolved_path(path: Path) -> Path:
    return path.expanduser().resolve()


def set_reproducibility(seed: int, deterministic: bool) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_and_validate_annotation(
    path: Path, split_name: str
) -> tuple[AnnotationSummary, dict[int, str], dict[str, int]]:
    path = resolved_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{split_name} annotation does not exist: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    for key in ("images", "annotations", "categories"):
        if key not in data or not isinstance(data[key], list):
            raise ValueError(f"{path}: missing or invalid COCO key {key!r}")

    image_ids = [int(image["id"]) for image in data["images"]]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError(f"{path}: image IDs must be globally unique")

    annotation_ids = [int(annotation["id"]) for annotation in data["annotations"]]
    if len(annotation_ids) != len(set(annotation_ids)):
        duplicates = len(annotation_ids) - len(set(annotation_ids))
        raise ValueError(
            f"{path}: {duplicates} duplicate annotation IDs; pycocotools would "
            "overwrite targets"
        )

    image_id_set = set(image_ids)
    invalid_references = [
        int(annotation["image_id"])
        for annotation in data["annotations"]
        if int(annotation["image_id"]) not in image_id_set
    ]
    if invalid_references:
        raise ValueError(
            f"{path}: annotations reference {len(invalid_references)} missing images"
        )

    categories = sorted(data["categories"], key=lambda category: int(category["id"]))
    category_ids = tuple(int(category["id"]) for category in categories)
    expected_ids = tuple(range(len(categories)))
    if category_ids != expected_ids:
        raise ValueError(
            f"{path}: category IDs must be contiguous 0..C-1; found {category_ids}"
        )

    id2label = {int(category["id"]): str(category["name"]) for category in categories}
    label2id = {name: category_id for category_id, name in id2label.items()}
    if len(label2id) != len(id2label):
        raise ValueError(f"{path}: category names must be unique")

    summary = AnnotationSummary(
        path=str(path),
        image_count=len(data["images"]),
        annotation_count=len(data["annotations"]),
        category_ids=category_ids,
        image_files=frozenset(str(image["file_name"]) for image in data["images"]),
    )
    return summary, id2label, label2id


def validate_dataset_inputs(
    args: argparse.Namespace,
) -> tuple[dict[str, AnnotationSummary], dict[int, str], dict[str, int]]:
    trainval_image_dir = resolved_path(args.trainval_image_dir)
    if not trainval_image_dir.is_dir():
        raise NotADirectoryError(
            f"train/validation image directory does not exist: {trainval_image_dir}"
        )
    if not args.skip_test:
        test_image_dir = resolved_path(args.test_image_dir)
        if not test_image_dir.is_dir():
            raise NotADirectoryError(
                f"test image directory does not exist: {test_image_dir}"
            )

    summaries: dict[str, AnnotationSummary] = {}
    label_maps: dict[str, tuple[dict[int, str], dict[str, int]]] = {}
    annotation_splits = [
        ("train", args.train_json),
        ("validation", args.val_json),
    ]
    if not args.skip_test:
        annotation_splits.append(("test", args.test_json))
    for split_name, path in annotation_splits:
        summary, id2label, label2id = load_and_validate_annotation(path, split_name)
        summaries[split_name] = summary
        label_maps[split_name] = (id2label, label2id)

    train_id2label, train_label2id = label_maps["train"]
    for split_name, _ in annotation_splits[1:]:
        if label_maps[split_name][0] != train_id2label:
            raise ValueError(
                f"{split_name} categories differ from the training categories"
            )

    train_val_overlap = summaries["train"].image_files & summaries[
        "validation"
    ].image_files
    if train_val_overlap:
        examples = ", ".join(sorted(train_val_overlap)[:5])
        raise ValueError(
            f"train/validation leakage: {len(train_val_overlap)} overlapping image "
            f"files (examples: {examples})"
        )

    print("Dataset summary")
    for split_name, summary in summaries.items():
        print(
            f"  {split_name:10s} images={summary.image_count:5d} "
            f"annotations={summary.annotation_count:6d} json={summary.path}"
        )
    return summaries, train_id2label, train_label2id


class ProcessedCocoDetection(torchvision.datasets.CocoDetection):
    def __init__(
        self,
        image_dir: Path,
        annotation_file: Path,
        processor: DeformableDetrImageProcessor,
    ) -> None:
        super().__init__(str(image_dir), str(annotation_file))
        self.processor = processor

    def __getitem__(self, index: int) -> tuple[Tensor, dict[str, Tensor]]:
        image, target = super().__getitem__(index)
        image_id = self.ids[index]
        annotation = {"image_id": image_id, "annotations": target}
        encoding = self.processor(
            images=image,
            annotations=annotation,
            return_tensors="pt",
        )
        return encoding["pixel_values"].squeeze(0), encoding["labels"][0]


class CocoCollator:
    def __init__(self, processor: DeformableDetrImageProcessor) -> None:
        self.processor = processor

    def __call__(
        self, batch: Sequence[tuple[Tensor, dict[str, Tensor]]]
    ) -> dict[str, Any]:
        pixel_values, labels = zip(*batch)
        encoding = self.processor.pad(list(pixel_values), return_tensors="pt")
        return {
            "pixel_values": encoding["pixel_values"],
            "pixel_mask": encoding["pixel_mask"],
            "labels": list(labels),
        }


def make_data_loader(
    dataset: ProcessedCocoDetection,
    collator: CocoCollator,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    shuffle: bool,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "collate_fn": collator,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "shuffle": shuffle,
        "pin_memory": pin_memory,
        "worker_init_fn": seed_worker,
        "generator": generator,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
        kwargs["persistent_workers"] = True
    return DataLoader(**kwargs)


def box_cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    center_x, center_y, width, height = boxes.unbind(-1)
    return torch.stack(
        (
            center_x - 0.5 * width,
            center_y - 0.5 * height,
            center_x + 0.5 * width,
            center_y + 0.5 * height,
        ),
        dim=-1,
    )


def box_xyxy_to_xywh(boxes: Tensor) -> Tensor:
    xmin, ymin, xmax, ymax = boxes.unbind(-1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=-1)


def box_iou(boxes1: Tensor, boxes2: Tensor) -> tuple[Tensor, Tensor]:
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    left_top = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    right_bottom = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    width_height = (right_bottom - left_top).clamp(min=0)
    intersection = width_height[:, :, 0] * width_height[:, :, 1]
    union = area1[:, None] + area2 - intersection
    return intersection / union.clamp(min=torch.finfo(union.dtype).eps), union


def generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    if not (boxes1[:, 2:] >= boxes1[:, :2]).all():
        raise ValueError("boxes1 contains invalid xyxy boxes")
    if not (boxes2[:, 2:] >= boxes2[:, :2]).all():
        raise ValueError("boxes2 contains invalid xyxy boxes")
    iou, union = box_iou(boxes1, boxes2)
    left_top = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    right_bottom = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    width_height = (right_bottom - left_top).clamp(min=0)
    enclosing_area = width_height[:, :, 0] * width_height[:, :, 1]
    return iou - (enclosing_area - union) / enclosing_area.clamp(
        min=torch.finfo(enclosing_area.dtype).eps
    )


class HungarianMatcher(nn.Module):
    def __init__(
        self,
        cost_class: float = 1.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
    ) -> None:
        super().__init__()
        if cost_class == 0 and cost_bbox == 0 and cost_giou == 0:
            raise ValueError("at least one matching cost must be non-zero")
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(
        self, outputs: dict[str, Tensor], targets: Sequence[dict[str, Tensor]]
    ) -> list[tuple[Tensor, Tensor]]:
        batch_size, num_queries = outputs["pred_logits"].shape[:2]
        sizes = [len(target["boxes"]) for target in targets]
        if sum(sizes) == 0:
            return [
                (
                    torch.empty(0, dtype=torch.int64),
                    torch.empty(0, dtype=torch.int64),
                )
                for _ in targets
            ]

        output_probability = outputs["pred_logits"].flatten(0, 1).sigmoid()
        output_boxes = outputs["pred_boxes"].flatten(0, 1)
        target_ids = torch.cat([target["labels"] for target in targets])
        target_boxes = torch.cat([target["boxes"] for target in targets])

        alpha = 0.25
        gamma = 2.0
        negative_class_cost = (1 - alpha) * (output_probability**gamma) * (
            -(1 - output_probability + 1e-8).log()
        )
        positive_class_cost = alpha * ((1 - output_probability) ** gamma) * (
            -(output_probability + 1e-8).log()
        )
        class_cost = (
            positive_class_cost[:, target_ids] - negative_class_cost[:, target_ids]
        )
        bbox_cost = torch.cdist(output_boxes, target_boxes, p=1)
        giou_cost = -generalized_box_iou(
            box_cxcywh_to_xyxy(output_boxes), box_cxcywh_to_xyxy(target_boxes)
        )
        cost_matrix = (
            self.cost_bbox * bbox_cost
            + self.cost_class * class_cost
            + self.cost_giou * giou_cost
        )
        cost_matrix = cost_matrix.view(batch_size, num_queries, -1).cpu()

        assignments = [
            linear_sum_assignment(cost[index])
            for index, cost in enumerate(cost_matrix.split(sizes, dim=-1))
        ]
        return [
            (
                torch.as_tensor(source, dtype=torch.int64),
                torch.as_tensor(target, dtype=torch.int64),
            )
            for source, target in assignments
        ]


def sigmoid_focal_loss_tensor(
    inputs: Tensor,
    targets: Tensor,
    alpha: float,
    gamma: float,
) -> Tensor:
    probability = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = probability * targets + (1 - probability) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return alpha_t * loss


def pn_sigmoid_focal_loss(
    inputs: Tensor,
    targets: Tensor,
    num_boxes: Tensor,
    alpha: float,
    gamma: float,
) -> Tensor:
    loss = sigmoid_focal_loss_tensor(inputs, targets, alpha, gamma)
    return loss.mean(dim=1).sum() / num_boxes


def pu_sigmoid_focal_loss(
    inputs: Tensor,
    targets: Tensor,
    num_boxes: Tensor,
    unlabeled_weight: Tensor,
    weight_p: float,
    reduction: str,
    alpha: float,
    gamma: float,
) -> Tensor:
    """Historical nnPU focal loss with the corrected R_p^- modulation.

    The historical implementation treats every zero one-hot class position as
    part of the unlabeled negative-risk term.  This is kept deliberately for
    compatibility with the experiment scripts being consolidated.
    """

    positive_mask = targets.eq(1)
    positive_focal = sigmoid_focal_loss_tensor(inputs, targets, alpha, gamma)

    probability = inputs.sigmoid()
    background_targets = torch.zeros_like(targets)
    background_ce = F.binary_cross_entropy_with_logits(
        inputs, background_targets, reduction="none"
    )
    # Correct focal modulation when a labeled positive is assumed negative.
    background_focal = (1 - alpha) * background_ce * (probability**gamma)

    positive_risk = torch.where(
        positive_mask,
        positive_focal * weight_p,
        torch.zeros_like(positive_focal),
    )
    negative_risk = torch.where(
        positive_mask,
        -background_focal * weight_p,
        background_focal * unlabeled_weight,
    )

    positive_scalar = positive_risk.mean(dim=1).sum() / num_boxes
    negative_scalar = reduce_non_negative_risk(
        negative_risk, num_boxes, reduction
    )
    return positive_scalar + negative_scalar


def reduce_non_negative_risk(
    negative_risk: Tensor, num_boxes: Tensor, reduction: str
) -> Tensor:
    """Apply nnPU correction at element, query/class, or image granularity."""

    if reduction == "element_wise":
        return negative_risk.clamp(min=0).mean(dim=1).sum() / num_boxes
    if reduction == "query_wise":
        per_query_class_risk = negative_risk.mean(dim=1)
        return per_query_class_risk.clamp(min=0).sum() / num_boxes
    if reduction == "global":
        per_image_risk = negative_risk.mean(dim=1).sum(dim=1)
        return per_image_risk.clamp(min=0).sum() / num_boxes
    raise ValueError(f"unsupported PU reduction: {reduction}")


class DetectionCriterion(nn.Module):
    def __init__(
        self,
        num_classes: int,
        matcher: HungarianMatcher,
        weight_dict: dict[str, float],
        method: str,
        weight_p: float,
        reduction: str,
        focal_alpha: float,
        focal_gamma: float,
        auxiliary_loss: bool,
        decoder_layers: int,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.method = method
        self.weight_p = weight_p
        self.reduction = reduction
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.auxiliary_loss = auxiliary_loss
        self.decoder_layers = decoder_layers

    @staticmethod
    def _source_permutation_indices(
        indices: Sequence[tuple[Tensor, Tensor]], device: torch.device
    ) -> tuple[Tensor, Tensor]:
        batch_parts = [
            torch.full_like(source, batch_index, device=device)
            for batch_index, (source, _) in enumerate(indices)
        ]
        source_parts = [source.to(device) for source, _ in indices]
        return torch.cat(batch_parts), torch.cat(source_parts)

    def loss_labels(
        self,
        outputs: dict[str, Tensor],
        targets: Sequence[dict[str, Tensor]],
        indices: Sequence[tuple[Tensor, Tensor]],
        num_boxes: Tensor,
    ) -> dict[str, Tensor]:
        source_logits = outputs["pred_logits"]
        batch_size, num_queries = source_logits.shape[:2]
        source_index = self._source_permutation_indices(indices, source_logits.device)

        matched_labels = [
            target["labels"][target_index.to(target["labels"].device)]
            for target, (_, target_index) in zip(targets, indices)
        ]
        target_classes_o = torch.cat(matched_labels).to(source_logits.device)
        target_classes = torch.full(
            source_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=source_logits.device,
        )
        target_classes[source_index] = target_classes_o

        target_onehot = torch.zeros(
            (
                source_logits.shape[0],
                source_logits.shape[1],
                source_logits.shape[2] + 1,
            ),
            dtype=source_logits.dtype,
            device=source_logits.device,
        )
        target_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_onehot = target_onehot[:, :, :-1]

        if self.method == "pn":
            classification_loss = pn_sigmoid_focal_loss(
                source_logits,
                target_onehot,
                num_boxes,
                self.focal_alpha,
                self.focal_gamma,
            )
        else:
            matched_query_count = target_classes_o.numel()
            unmatched_query_count = max(
                batch_size * num_queries - matched_query_count, 1
            )
            unlabeled_weight = source_logits.new_tensor(
                (batch_size * num_queries) / unmatched_query_count
            )
            classification_loss = pu_sigmoid_focal_loss(
                source_logits,
                target_onehot,
                num_boxes,
                unlabeled_weight,
                self.weight_p,
                self.reduction,
                self.focal_alpha,
                self.focal_gamma,
            )

        # Preserve the Deformable DETR implementation's query scaling.
        return {"loss_ce": classification_loss * num_queries}

    def loss_boxes(
        self,
        outputs: dict[str, Tensor],
        targets: Sequence[dict[str, Tensor]],
        indices: Sequence[tuple[Tensor, Tensor]],
        num_boxes: Tensor,
    ) -> dict[str, Tensor]:
        if sum(len(target["boxes"]) for target in targets) == 0:
            zero = outputs["pred_boxes"].sum() * 0
            return {"loss_bbox": zero, "loss_giou": zero}

        source_index = self._source_permutation_indices(
            indices, outputs["pred_boxes"].device
        )
        source_boxes = outputs["pred_boxes"][source_index]
        target_boxes = torch.cat(
            [
                target["boxes"][target_index.to(target["boxes"].device)]
                for target, (_, target_index) in zip(targets, indices)
            ],
            dim=0,
        ).to(source_boxes.device)

        bbox_loss = F.l1_loss(source_boxes, target_boxes, reduction="none")
        giou_loss = 1 - torch.diag(
            generalized_box_iou(
                box_cxcywh_to_xyxy(source_boxes), box_cxcywh_to_xyxy(target_boxes)
            )
        )
        return {
            "loss_bbox": bbox_loss.sum() / num_boxes,
            "loss_giou": giou_loss.sum() / num_boxes,
        }

    def _compute_losses(
        self,
        outputs: dict[str, Tensor],
        targets: Sequence[dict[str, Tensor]],
        num_boxes: Tensor,
    ) -> dict[str, Tensor]:
        indices = self.matcher(outputs, targets)
        losses = self.loss_labels(outputs, targets, indices, num_boxes)
        losses.update(self.loss_boxes(outputs, targets, indices, num_boxes))
        return losses

    def forward(
        self, outputs: dict[str, Any], targets: Sequence[dict[str, Tensor]]
    ) -> dict[str, Tensor]:
        outputs_without_auxiliary = {
            key: value for key, value in outputs.items() if key != "aux_outputs"
        }
        raw_num_boxes = sum(len(target["labels"]) for target in targets)
        num_boxes = outputs["pred_logits"].new_tensor(float(max(raw_num_boxes, 1)))
        losses = self._compute_losses(outputs_without_auxiliary, targets, num_boxes)

        if self.auxiliary_loss and "aux_outputs" in outputs:
            for layer_index, auxiliary_outputs in enumerate(outputs["aux_outputs"]):
                auxiliary_losses = self._compute_losses(
                    auxiliary_outputs, targets, num_boxes
                )
                losses.update(
                    {
                        f"{name}_{layer_index}": value
                        for name, value in auxiliary_losses.items()
                    }
                )
        return losses


def inverse_sigmoid(value: Tensor, epsilon: float = 1e-5) -> Tensor:
    value = value.clamp(min=0, max=1)
    return torch.log(value.clamp(min=epsilon) / (1 - value).clamp(min=epsilon))


def predictions_to_coco(
    processor: DeformableDetrImageProcessor,
    outputs: Any,
    labels: Sequence[dict[str, Tensor]],
) -> list[dict[str, Any]]:
    original_sizes = torch.stack([label["orig_size"] for label in labels], dim=0)
    predictions = processor.post_process_object_detection(
        outputs, threshold=0.0, target_sizes=original_sizes
    )
    coco_predictions: list[dict[str, Any]] = []
    for label, prediction in zip(labels, predictions):
        image_id = int(label["image_id"].item())
        boxes = box_xyxy_to_xywh(prediction["boxes"]).tolist()
        for category_id, score, box in zip(
            prediction["labels"].tolist(),
            prediction["scores"].tolist(),
            boxes,
        ):
            coco_predictions.append(
                {
                    "image_id": image_id,
                    "category_id": int(category_id),
                    "bbox": box,
                    "score": float(score),
                }
            )
    return coco_predictions


def run_coco_evaluation(
    ground_truth: COCO, predictions: list[dict[str, Any]]
) -> dict[str, float]:
    if not predictions:
        raise RuntimeError("COCO evaluation received no predictions")
    detections = ground_truth.loadRes(predictions)
    evaluator = COCOeval(ground_truth, detections, "bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return {
        name: float(value)
        for name, value in zip(COCO_METRIC_NAMES, evaluator.stats)
    }


class DeformableDetrExperiment(pl.LightningModule):
    def __init__(
        self,
        processor: DeformableDetrImageProcessor,
        id2label: dict[int, str],
        label2id: dict[str, int],
        validation_json: Path,
        method: str,
        weight_p: float,
        reduction: str,
        hf_checkpoint: str,
        hf_revision: str | None,
        local_files_only: bool,
        lr: float,
        lr_backbone: float,
        weight_decay: float,
        warmup_steps: int,
        focal_alpha: float,
        focal_gamma: float,
        auxiliary_loss: bool = False,
    ) -> None:
        super().__init__()
        self.processor = processor
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.auxiliary_loss = auxiliary_loss

        pretrained_kwargs: dict[str, Any] = {
            "revision": hf_revision,
            "local_files_only": local_files_only,
        }
        config = DeformableDetrConfig.from_pretrained(
            hf_checkpoint,
            num_labels=len(id2label),
            id2label=id2label,
            label2id=label2id,
            auxiliary_loss=auxiliary_loss,
            **pretrained_kwargs,
        )
        # The full Deformable DETR checkpoint already contains backbone weights.
        # Prevent timm from attempting a second, unpinned backbone download while
        # the full checkpoint is being reconstructed (especially in offline runs).
        config.use_pretrained_backbone = False
        self.model = DeformableDetrForObjectDetection.from_pretrained(
            hf_checkpoint,
            config=config,
            ignore_mismatched_sizes=True,
            **pretrained_kwargs,
        )

        weight_dict: dict[str, float] = {
            "loss_ce": 1.0,
            "loss_bbox": float(self.model.config.bbox_loss_coefficient),
            "loss_giou": float(self.model.config.giou_loss_coefficient),
        }
        if auxiliary_loss:
            for layer_index in range(self.model.config.decoder_layers - 1):
                weight_dict.update(
                    {
                        f"loss_ce_{layer_index}": 1.0,
                        f"loss_bbox_{layer_index}": float(
                            self.model.config.bbox_loss_coefficient
                        ),
                        f"loss_giou_{layer_index}": float(
                            self.model.config.giou_loss_coefficient
                        ),
                    }
                )
        self.weight_dict = weight_dict
        self.criterion = DetectionCriterion(
            num_classes=len(id2label),
            matcher=HungarianMatcher(
                cost_class=float(self.model.config.class_cost),
                cost_bbox=float(self.model.config.bbox_cost),
                cost_giou=float(self.model.config.giou_cost),
            ),
            weight_dict=weight_dict,
            method=method,
            weight_p=weight_p,
            reduction=reduction,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            auxiliary_loss=auxiliary_loss,
            decoder_layers=self.model.config.decoder_layers,
        )
        self.validation_ground_truth = COCO(str(validation_json))
        self.validation_predictions: list[dict[str, Any]] = []
        self.last_validation_metrics: dict[str, float] = {}

    def forward(self, pixel_values: Tensor, pixel_mask: Tensor) -> Any:
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    def _custom_outputs(self, model_outputs: Any) -> dict[str, Any]:
        hidden_states = model_outputs.intermediate_hidden_states
        initial_reference = model_outputs.init_reference_points
        intermediate_references = model_outputs.intermediate_reference_points
        output_classes: list[Tensor] = []
        output_coordinates: list[Tensor] = []

        for level in range(hidden_states.shape[1]):
            reference = (
                initial_reference
                if level == 0
                else intermediate_references[:, level - 1]
            )
            reference = inverse_sigmoid(reference)
            output_class = self.model.class_embed[level](hidden_states[:, level])
            delta_box = self.model.bbox_embed[level](hidden_states[:, level])
            if reference.shape[-1] == 4:
                coordinate_logits = delta_box + reference
            elif reference.shape[-1] == 2:
                coordinate_logits = torch.cat(
                    (delta_box[..., :2] + reference, delta_box[..., 2:]), dim=-1
                )
            else:
                raise ValueError(
                    f"unexpected reference point dimension: {reference.shape[-1]}"
                )
            output_classes.append(output_class)
            output_coordinates.append(coordinate_logits.sigmoid())

        stacked_classes = torch.stack(output_classes, dim=1)
        stacked_coordinates = torch.stack(output_coordinates, dim=1)
        outputs: dict[str, Any] = {
            "pred_logits": stacked_classes[:, -1],
            "pred_boxes": stacked_coordinates[:, -1],
        }
        if self.auxiliary_loss:
            outputs["aux_outputs"] = [
                {"pred_logits": logits, "pred_boxes": boxes}
                for logits, boxes in zip(
                    stacked_classes[:, :-1].unbind(dim=1),
                    stacked_coordinates[:, :-1].unbind(dim=1),
                )
            ]
        return outputs

    def _common_step(
        self, batch: dict[str, Any]
    ) -> tuple[Tensor, dict[str, Tensor], Any, list[dict[str, Tensor]]]:
        pixel_values = batch["pixel_values"].to(self.device)
        pixel_mask = batch["pixel_mask"].to(self.device)
        labels = [
            {name: value.to(self.device) for name, value in label.items()}
            for label in batch["labels"]
        ]
        targets = [
            {"labels": label["class_labels"], "boxes": label["boxes"]}
            for label in labels
        ]
        model_outputs = self.model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            output_hidden_states=True,
        )
        custom_outputs = self._custom_outputs(model_outputs)
        loss_dict = self.criterion(custom_outputs, targets)
        loss = sum(
            self.weight_dict[name] * value
            for name, value in loss_dict.items()
            if name in self.weight_dict
        )
        return loss, loss_dict, model_outputs, labels

    def training_step(self, batch: dict[str, Any], batch_index: int) -> Tensor:
        del batch_index
        loss, loss_dict, _, _ = self._common_step(batch)
        batch_size = batch["pixel_values"].shape[0]
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        for name, value in loss_dict.items():
            self.log(
                f"train_{name}",
                value,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )
        return loss

    def on_validation_epoch_start(self) -> None:
        self.validation_predictions.clear()

    def validation_step(self, batch: dict[str, Any], batch_index: int) -> Tensor:
        del batch_index
        loss, loss_dict, model_outputs, labels = self._common_step(batch)
        batch_size = batch["pixel_values"].shape[0]
        self.log(
            "validation_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        for name, value in loss_dict.items():
            self.log(
                f"validation_{name}",
                value,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )
        self.validation_predictions.extend(
            predictions_to_coco(self.processor, model_outputs, labels)
        )
        return loss

    def on_validation_epoch_end(self) -> None:
        metrics = run_coco_evaluation(
            self.validation_ground_truth, self.validation_predictions
        )
        self.last_validation_metrics = metrics
        self.log("validation_AP", metrics["AP"], prog_bar=True)
        for name, value in metrics.items():
            if name != "AP":
                self.log(f"validation_{name}", value)
        self.validation_predictions.clear()

    def configure_optimizers(self) -> dict[str, Any]:
        parameter_groups = [
            {
                "params": [
                    parameter
                    for name, parameter in self.named_parameters()
                    if "backbone" not in name and parameter.requires_grad
                ]
            },
            {
                "params": [
                    parameter
                    for name, parameter in self.named_parameters()
                    if "backbone" in name and parameter.requires_grad
                ],
                "lr": self.lr_backbone,
            },
        ]
        optimizer = torch.optim.AdamW(
            parameter_groups, lr=self.lr, weight_decay=self.weight_decay
        )

        total_steps = int(self.trainer.estimated_stepping_batches)
        if total_steps <= 1:
            return {"optimizer": optimizer}
        warmup_steps = min(self.warmup_steps, total_steps - 1)
        if warmup_steps == 0:
            scheduler: Any = CosineAnnealingLR(optimizer, T_max=total_steps)
        else:
            warmup = LinearLR(
                optimizer,
                start_factor=1e-8,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            cosine = CosineAnnealingLR(
                optimizer,
                T_max=max(total_steps - warmup_steps, 1),
                eta_min=0.0,
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_steps],
            )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


@torch.no_grad()
def evaluate_model(
    model: DeformableDetrExperiment,
    data_loader: DataLoader,
    processor: DeformableDetrImageProcessor,
    annotation_json: Path,
) -> dict[str, float]:
    model.eval()
    predictions: list[dict[str, Any]] = []
    for batch in tqdm(data_loader, desc="test"):
        pixel_values = batch["pixel_values"].to(model.device)
        pixel_mask = batch["pixel_mask"].to(model.device)
        labels = [
            {name: value.to(model.device) for name, value in label.items()}
            for label in batch["labels"]
        ]
        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        predictions.extend(predictions_to_coco(processor, outputs, labels))
    return run_coco_evaluation(COCO(str(annotation_json)), predictions)


def resolve_lightning_device(device: str) -> tuple[str, int | list[int]]:
    device = normalize_device_argument(device)
    if device == "auto":
        return ("gpu", 1) if torch.cuda.is_available() else ("cpu", 1)
    if device == "cpu":
        return "cpu", 1
    if not torch.cuda.is_available():
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<not set>")
        raise RuntimeError(
            f"CUDA device requested but CUDA is unavailable: {device}; "
            f"CUDA_VISIBLE_DEVICES={visible}"
        )
    if device == "cuda":
        return "gpu", 1

    index = int(device)
    if os.environ.get("CUDA_VISIBLE_DEVICES") == str(index):
        if torch.cuda.device_count() != 1:
            raise RuntimeError(
                "A concrete GPU ID must expose exactly one logical CUDA device; "
                f"CUDA_VISIBLE_DEVICES={index}, count={torch.cuda.device_count()}"
            )
        return "gpu", 1
    if index < 0 or index >= torch.cuda.device_count():
        raise RuntimeError(
            f"CUDA device index {index} is unavailable; count={torch.cuda.device_count()}"
        )
    return "gpu", [index]


def make_unique_run_dir(base: Path, experiment_name: str, seed: int) -> Path:
    experiment_dir = resolved_path(base) / experiment_name
    candidate = experiment_dir / f"seed_{seed}"
    suffix = 1
    while candidate.exists():
        candidate = experiment_dir / f"seed_{seed}_run_{suffix:03d}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(resolved_path(value))
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(data), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_metrics_csv(
    path: Path,
    best_validation_ap: float,
    test_metrics: dict[str, float] | None,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("split", "metric", "value"))
        writer.writerow(("validation", "AP", best_validation_ap))
        if test_metrics is not None:
            for name, value in test_metrics.items():
                writer.writerow(("test", name, value))


def package_versions() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "pytorch_lightning": pl.__version__,
        "transformers": __import__("transformers").__version__,
        "numpy": np.__version__,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    set_reproducibility(args.seed, args.deterministic)
    summaries, id2label, label2id = validate_dataset_inputs(args)
    accelerator, devices = resolve_lightning_device(args.device)
    run_dir = make_unique_run_dir(args.output_dir, args.experiment_name, args.seed)

    print(f"Run directory: {run_dir}")
    print(
        f"Method={args.method} weight_p(alpha)={args.weight_p} "
        f"reduction={args.reduction} seed={args.seed}"
    )
    if accelerator == "gpu":
        print(
            f"CUDA selection: requested={args.device} "
            f"physical_gpu={concrete_cuda_index(args.device)} "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<not set>')} "
            f"logical_device=cuda:0"
        )

    processor_kwargs: dict[str, Any] = {
        "revision": args.hf_revision,
        "local_files_only": args.local_files_only,
    }
    processor = DeformableDetrImageProcessor.from_pretrained(
        args.hf_checkpoint, **processor_kwargs
    )

    trainval_image_dir = resolved_path(args.trainval_image_dir)
    train_json = resolved_path(args.train_json)
    validation_json = resolved_path(args.val_json)
    test_image_dir = (
        None if args.skip_test else resolved_path(args.test_image_dir)
    )
    test_json = None if args.skip_test else resolved_path(args.test_json)

    train_dataset = ProcessedCocoDetection(
        trainval_image_dir, train_json, processor
    )
    validation_dataset = ProcessedCocoDetection(
        trainval_image_dir, validation_json, processor
    )
    test_dataset = (
        None
        if args.skip_test
        else ProcessedCocoDetection(test_image_dir, test_json, processor)
    )
    collator = CocoCollator(processor)
    pin_memory = accelerator == "gpu"
    train_loader = make_data_loader(
        train_dataset,
        collator,
        args.batch_size,
        args.num_workers,
        args.prefetch_factor,
        True,
        args.seed,
        pin_memory,
    )
    validation_loader = make_data_loader(
        validation_dataset,
        collator,
        args.batch_size,
        args.num_workers,
        args.prefetch_factor,
        False,
        args.seed + 1,
        pin_memory,
    )
    test_loader = None
    if test_dataset is not None:
        test_loader = make_data_loader(
            test_dataset,
            collator,
            args.batch_size,
            args.num_workers,
            args.prefetch_factor,
            False,
            args.seed + 2,
            pin_memory,
        )

    model = DeformableDetrExperiment(
        processor=processor,
        id2label=id2label,
        label2id=label2id,
        validation_json=validation_json,
        method=args.method,
        weight_p=args.weight_p,
        reduction=args.reduction,
        hf_checkpoint=args.hf_checkpoint,
        hf_revision=args.hf_revision,
        local_files_only=args.local_files_only,
        lr=args.lr,
        lr_backbone=args.lr_backbone,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        auxiliary_loss=False,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=run_dir / "checkpoints",
        monitor="validation_AP",
        mode="max",
        save_top_k=1,
        save_last=True,
        filename="best-epoch{epoch:02d}-validation_AP{validation_AP:.4f}",
        auto_insert_metric_name=False,
    )
    logger = CSVLogger(save_dir=str(run_dir), name="lightning_logs")
    deterministic_setting = lightning_deterministic_setting(args.deterministic)
    trainer = Trainer(
        max_epochs=args.epochs,
        gradient_clip_val=args.gradient_clip,
        accelerator=accelerator,
        devices=devices,
        precision=args.precision,
        deterministic=deterministic_setting,
        logger=logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=args.log_every_n_steps,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=1,
    )

    configuration: dict[str, Any] = {
        "arguments": vars(args),
        "resolved": {
            "run_dir": run_dir,
            "accelerator": accelerator,
            "devices": devices,
            "physical_gpu": concrete_cuda_index(args.device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "lightning_deterministic": deterministic_setting,
            "train_json": train_json,
            "validation_json": validation_json,
            "test_json": test_json,
            "trainval_image_dir": trainval_image_dir,
            "test_image_dir": test_image_dir,
        },
        "dataset": {
            split: {
                "path": summary.path,
                "images": summary.image_count,
                "annotations": summary.annotation_count,
                "category_ids": summary.category_ids,
            }
            for split, summary in summaries.items()
        },
        "versions": package_versions(),
        "implementation": {
            "weight_p_is_paper_alpha": True,
            "pu_mask": "historical_all_zero_onehot_positions",
            "positive_as_negative_modulation": "probability**gamma",
            "auxiliary_loss": False,
            "test_checkpoint": "best_validation_AP",
        },
    }
    write_json(run_dir / "config.json", configuration)

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=validation_loader,
    )

    best_checkpoint = checkpoint_callback.best_model_path
    if not best_checkpoint:
        raise RuntimeError("training finished without a best-validation checkpoint")
    best_validation_ap = float(checkpoint_callback.best_model_score.item())
    print(f"Best checkpoint: {best_checkpoint}")
    print(f"Best validation AP: {best_validation_ap:.6f}")

    # This checkpoint was created by the same run and contains Lightning
    # metadata in addition to tensors, so load it explicitly as a trusted full
    # checkpoint rather than relying on PyTorch's weights-only default.
    checkpoint = torch.load(
        best_checkpoint, map_location="cpu", weights_only=False
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    if accelerator == "gpu":
        if isinstance(devices, list):
            evaluation_device = torch.device(f"cuda:{devices[0]}")
        else:
            evaluation_device = torch.device("cuda:0")
    else:
        evaluation_device = torch.device("cpu")
    model.to(evaluation_device)

    test_metrics: dict[str, float] | None = None
    if not args.skip_test:
        assert test_loader is not None and test_json is not None
        test_metrics = evaluate_model(model, test_loader, processor, test_json)

    configuration["result"] = {
        "best_checkpoint": best_checkpoint,
        "best_validation_ap": best_validation_ap,
        "test_metrics": test_metrics,
    }
    write_json(run_dir / "config.json", configuration)
    write_metrics_csv(
        run_dir / "metrics.csv", best_validation_ap, test_metrics
    )
    print(f"Metrics written to {run_dir / 'metrics.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
