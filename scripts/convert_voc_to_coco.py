#!/usr/bin/env python3
"""Convert PASCAL VOC 2007 XML annotations to PUD-DETR COCO JSON files.

The conversion intentionally mirrors the annotation representation used by
jiseokson/PUD-DETR while keeping annotation IDs unique for pycocotools and the
validation performed by ``train_pud_detr.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


VOC_CATEGORIES = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)
CATEGORY_TO_ID = {name: category_id for category_id, name in enumerate(VOC_CATEGORIES)}
DROP_DIRECTORY_PATTERN = re.compile(
    r"^Annotations_drop_ratio_(?P<ratio>\d+(?:\.\d+)?)(?:_seed_.+)?$"
)
STATISTICS_FILENAME = "drop_statistics.json"


@dataclass(frozen=True)
class ConversionJob:
    split_name: str
    image_ids: tuple[str, ...]
    annotations_dir: Path
    output_path: Path
    dropped: bool = False


def categories() -> list[dict[str, Any]]:
    return [
        {"supercategory": "none", "id": category_id, "name": name}
        for category_id, name in enumerate(VOC_CATEGORIES)
    ]


def read_split(path: Path) -> tuple[str, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"split file does not exist: {path}")
    image_ids = tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not image_ids:
        raise ValueError(f"split file is empty: {path}")
    if len(image_ids) != len(set(image_ids)):
        raise ValueError(f"split file contains duplicate image IDs: {path}")
    for image_id in image_ids:
        if not image_id.isdigit():
            raise ValueError(f"image ID must be numeric for PUD-DETR compatibility: {image_id!r}")
    return image_ids


def require_text(parent: ET.Element, path: str, source: Path) -> str:
    value = parent.findtext(path)
    if value is None or not value.strip():
        raise ValueError(f"{source}: missing XML value {path!r}")
    return value.strip()


def parse_integer(parent: ET.Element, path: str, source: Path) -> int:
    value = require_text(parent, path, source)
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{source}: {path!r} must be an integer, found {value!r}") from exc


def convert_split(
    image_ids: Sequence[str], annotations_dir: Path, split_name: str, dropped: bool
) -> dict[str, Any]:
    if not annotations_dir.is_dir():
        raise NotADirectoryError(f"annotation directory does not exist: {annotations_dir}")

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    annotation_id = 1

    for image_id_text in image_ids:
        xml_path = annotations_dir / f"{image_id_text}.xml"
        if not xml_path.is_file():
            raise FileNotFoundError(f"{split_name}: annotation XML does not exist: {xml_path}")
        if xml_path.stat().st_size == 0:
            raise ValueError(f"{split_name}: annotation XML is empty: {xml_path}")
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError as exc:
            raise ValueError(f"{split_name}: invalid XML {xml_path}: {exc}") from exc

        filename = require_text(root, "filename", xml_path)
        expected_filename = f"{image_id_text}.jpg"
        if filename != expected_filename:
            raise ValueError(
                f"{xml_path}: filename is {filename!r}, expected {expected_filename!r}"
            )
        width = parse_integer(root, "size/width", xml_path)
        height = parse_integer(root, "size/height", xml_path)
        if width <= 0 or height <= 0:
            raise ValueError(f"{xml_path}: image dimensions must be positive")

        image_id = int(image_id_text)
        images.append(
            {
                "file_name": filename,
                "height": height,
                "width": width,
                "id": image_id,
            }
        )

        for obj in root.findall("object"):
            class_name = require_text(obj, "name", xml_path)
            if class_name not in CATEGORY_TO_ID:
                raise ValueError(f"{xml_path}: unknown VOC category {class_name!r}")

            xmin = parse_integer(obj, "bndbox/xmin", xml_path)
            ymin = parse_integer(obj, "bndbox/ymin", xml_path)
            xmax = parse_integer(obj, "bndbox/xmax", xml_path)
            ymax = parse_integer(obj, "bndbox/ymax", xml_path)
            if not (1 <= xmin <= xmax <= width and 1 <= ymin <= ymax <= height):
                raise ValueError(
                    f"{xml_path}: invalid bbox {(xmin, ymin, xmax, ymax)} "
                    f"for image size {(width, height)}"
                )

            # VOC coordinates are 1-based and inclusive. PUD-DETR converts
            # them to zero-based COCO xywh while retaining inclusive extents.
            x = xmin - 1
            y = ymin - 1
            box_width = xmax - xmin + 1
            box_height = ymax - ymin + 1
            difficult_text = obj.findtext("difficult", default="0").strip()
            try:
                difficult = int(difficult_text)
            except ValueError as exc:
                raise ValueError(
                    f"{xml_path}: 'difficult' must be an integer, found {difficult_text!r}"
                ) from exc
            if difficult not in (0, 1):
                raise ValueError(f"{xml_path}: 'difficult' must be 0 or 1")

            annotations.append(
                {
                    "segmentation": [
                        [
                            x,
                            y,
                            x,
                            y + box_height,
                            x + box_width,
                            y + box_height,
                            x + box_width,
                            y,
                        ]
                    ],
                    "area": box_width * box_height,
                    "iscrowd": 0,
                    "image_id": image_id,
                    "bbox": [x, y, box_width, box_height],
                    "category_id": CATEGORY_TO_ID[class_name],
                    "id": annotation_id,
                    "ignore": difficult,
                }
            )
            annotation_id += 1

    if dropped:
        return {
            "info": {},
            "licenses": [],
            "images": images,
            "annotations": annotations,
            "categories": categories(),
        }
    if split_name == "test":
        return {
            "images": images,
            "type": "instances",
            "annotations": annotations,
            "categories": categories(),
            "info": {},
            "licenses": [],
        }
    return {
        "type": "instances",
        "categories": categories(),
        "info": {},
        "licenses": [],
        "images": images,
        "annotations": annotations,
    }


def ratio_label_from_directory(path: Path) -> str:
    statistics_path = path / STATISTICS_FILENAME
    if statistics_path.is_file():
        try:
            statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
            ratio = float(statistics["requested"]["drop_ratio"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid drop statistics file: {statistics_path}") from exc
        if not 0.0 <= ratio <= 1.0:
            raise ValueError(f"drop ratio in {statistics_path} must be in [0, 1]")
        return format(ratio, ".6g")

    match = DROP_DIRECTORY_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(
            f"cannot infer drop ratio from {path}; provide a {STATISTICS_FILENAME} "
            "or use an Annotations_drop_ratio_<RATIO> directory name"
        )
    return format(float(match.group("ratio")), ".6g")


def discover_drop_directories(voc_root: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in voc_root.iterdir()
            if path.is_dir()
            and path.name.startswith("Annotations_drop_")
            and (
                (path / STATISTICS_FILENAME).is_file()
                or DROP_DIRECTORY_PATTERN.fullmatch(path.name)
            )
        ),
        key=lambda path: path.name,
    )


def build_jobs(args: argparse.Namespace) -> list[ConversionJob]:
    voc_root = args.voc_root.expanduser().resolve()
    annotations_dir = (
        args.annotations_dir.expanduser().resolve()
        if args.annotations_dir is not None
        else voc_root / "Annotations"
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else voc_root / "coco_annotations"
    )
    split_dir = voc_root / "ImageSets" / "Main"
    splits = {
        name: read_split(split_dir / f"{name}.txt") for name in ("train", "val", "test")
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = set(splits[left]) & set(splits[right])
        if overlap:
            examples = ", ".join(sorted(overlap)[:5])
            raise ValueError(
                f"{left}.txt and {right}.txt overlap by {len(overlap)} images "
                f"(examples: {examples})"
            )

    jobs = [
        ConversionJob(name, splits[name], annotations_dir, output_dir / f"pascal_{name}.json")
        for name in ("train", "val", "test")
    ]

    if not args.skip_drop_annotations:
        drop_directories = (
            [path.expanduser().resolve() for path in args.drop_annotations_dir]
            if args.drop_annotations_dir
            else discover_drop_directories(voc_root)
        )
        seen_ratios: dict[str, Path] = {}
        for drop_dir in drop_directories:
            ratio_label = ratio_label_from_directory(drop_dir)
            if ratio_label in seen_ratios:
                raise ValueError(
                    f"multiple drop annotation directories use ratio {ratio_label}: "
                    f"{seen_ratios[ratio_label]} and {drop_dir}"
                )
            seen_ratios[ratio_label] = drop_dir
            jobs.append(
                ConversionJob(
                    "train",
                    splits["train"],
                    drop_dir,
                    output_dir / f"pascal_train_drop_{ratio_label}.json",
                    dropped=True,
                )
            )
    return jobs


def write_json_atomic(path: Path, data: dict[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"output already exists (use --overwrite): {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", text=True
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(data, handle, ensure_ascii=False, separators=(",", ":"))
            handle.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def run(args: argparse.Namespace) -> list[Path]:
    jobs = build_jobs(args)
    existing = [job.output_path for job in jobs if job.output_path.exists()]
    if existing and not args.overwrite:
        examples = ", ".join(str(path) for path in existing[:3])
        raise FileExistsError(f"output already exists (use --overwrite): {examples}")

    converted: list[tuple[ConversionJob, dict[str, Any]]] = []
    for job in jobs:
        data = convert_split(
            job.image_ids, job.annotations_dir, job.split_name, job.dropped
        )
        converted.append((job, data))

    output_paths: list[Path] = []
    for job, data in converted:
        write_json_atomic(job.output_path, data, args.overwrite)
        output_paths.append(job.output_path)
        print(
            f"wrote {job.output_path} "
            f"(images={len(data['images'])}, annotations={len(data['annotations'])})"
        )
    return output_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert VOC2007 XML and bbox-drop XML to PUD-DETR COCO JSON."
    )
    parser.add_argument(
        "--voc-root",
        type=Path,
        required=True,
        help="VOC2007 root containing Annotations, JPEGImages, and ImageSets/Main",
    )
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        help="base XML directory (default: <VOC_ROOT>/Annotations)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="JSON destination (default: <VOC_ROOT>/coco_annotations)",
    )
    parser.add_argument(
        "--drop-annotations-dir",
        action="append",
        type=Path,
        default=[],
        help=(
            "bbox-drop XML directory; repeat for multiple ratios. If omitted, "
            "Annotations_drop_ratio_* directories under VOC_ROOT are discovered."
        ),
    )
    parser.add_argument(
        "--skip-drop-annotations",
        action="store_true",
        help="generate only pascal_train.json, pascal_val.json, and pascal_test.json",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="replace existing output JSON files"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        run(args)
    except (FileNotFoundError, FileExistsError, NotADirectoryError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()
