#!/usr/bin/env python3
"""Create class-balanced, instance-dropped PASCAL VOC annotations.

Only annotations listed in ImageSets/Main/train.txt are modified. Every source
annotation is first copied to a separate output directory, so validation and
test annotations remain byte-for-byte identical to the originals.
"""

from __future__ import annotations

import argparse
import math
import random
import shutil
import tempfile
import uuid
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, List, Mapping, Optional, Sequence, Set, Tuple


@dataclass
class Annotation:
    image_id: str
    tree: ET.ElementTree
    objects: List[ET.Element]
    classes: List[str]


class Edge:
    __slots__ = ("to", "reverse", "capacity", "initial_capacity")

    def __init__(self, to: int, reverse: int, capacity: int) -> None:
        self.to = to
        self.reverse = reverse
        self.capacity = capacity
        self.initial_capacity = capacity


class Dinic:
    """Small integer max-flow implementation used for constrained sampling."""

    def __init__(self, node_count: int) -> None:
        self.graph: List[List[Edge]] = [[] for _ in range(node_count)]

    def add_edge(self, source: int, target: int, capacity: int) -> Edge:
        forward = Edge(target, len(self.graph[target]), capacity)
        backward = Edge(source, len(self.graph[source]), 0)
        self.graph[source].append(forward)
        self.graph[target].append(backward)
        return forward

    def max_flow(self, source: int, sink: int) -> int:
        total_flow = 0
        node_count = len(self.graph)

        while True:
            level = [-1] * node_count
            level[source] = 0
            queue = deque([source])
            while queue:
                node = queue.popleft()
                for edge in self.graph[node]:
                    if edge.capacity > 0 and level[edge.to] < 0:
                        level[edge.to] = level[node] + 1
                        queue.append(edge.to)
            if level[sink] < 0:
                return total_flow

            next_edge = [0] * node_count

            def send(node: int, amount: int) -> int:
                if node == sink:
                    return amount
                while next_edge[node] < len(self.graph[node]):
                    edge = self.graph[node][next_edge[node]]
                    if edge.capacity > 0 and level[node] + 1 == level[edge.to]:
                        pushed = send(edge.to, min(amount, edge.capacity))
                        if pushed:
                            edge.capacity -= pushed
                            reverse = self.graph[edge.to][edge.reverse]
                            reverse.capacity += pushed
                            return pushed
                    next_edge[node] += 1
                return 0

            while True:
                pushed = send(source, 1 << 60)
                if not pushed:
                    break
                total_flow += pushed


def read_split(path: Path) -> List[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Split file not found: {path}")
    image_ids = [Path(line.strip().split()[0]).stem for line in path.read_text().splitlines() if line.strip()]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError(f"Split contains duplicate image IDs: {path}")
    return image_ids


def load_train_annotations(annotation_dir: Path, image_ids: Sequence[str]) -> Dict[str, Annotation]:
    annotations: Dict[str, Annotation] = {}
    for image_id in image_ids:
        path = annotation_dir / f"{image_id}.xml"
        if not path.is_file():
            raise FileNotFoundError(f"Annotation listed by train.txt is missing: {path}")
        try:
            tree = ET.parse(path)
        except ET.ParseError as exc:
            raise ValueError(f"Invalid XML annotation: {path}: {exc}") from exc

        objects = list(tree.getroot().findall("object"))
        if not objects:
            raise ValueError(f"Training image has no bounding boxes: {path}")

        classes: List[str] = []
        for index, obj in enumerate(objects):
            name = obj.findtext("name")
            bbox = obj.find("bndbox")
            if name is None or not name.strip():
                raise ValueError(f"Object {index} has no class name: {path}")
            if bbox is None:
                raise ValueError(f"Object {index} has no bndbox: {path}")
            classes.append(name.strip())

        annotations[image_id] = Annotation(image_id, tree, objects, classes)
    return annotations


def quotas_for_ratio(class_totals: Mapping[str, int], ratio: float) -> Dict[str, int]:
    return {
        class_name: int(math.floor(total * ratio + 1e-12))
        for class_name, total in class_totals.items()
    }


def make_flow_plan(
    quotas: Mapping[str, int],
    pair_instances: Mapping[Tuple[str, str], Sequence[int]],
    image_capacities: Mapping[str, int],
    seed: int,
) -> Tuple[int, Dict[Tuple[str, str], int]]:
    rng = random.Random(seed)
    classes = list(quotas)
    images = list(image_capacities)
    rng.shuffle(classes)
    rng.shuffle(images)

    source = 0
    class_node = {name: index + 1 for index, name in enumerate(classes)}
    image_offset = 1 + len(classes)
    image_node = {name: image_offset + index for index, name in enumerate(images)}
    sink = image_offset + len(images)
    network = Dinic(sink + 1)

    for class_name in classes:
        network.add_edge(source, class_node[class_name], quotas[class_name])

    tracked_edges: Dict[Tuple[str, str], Edge] = {}
    pairs_by_class: DefaultDict[str, List[Tuple[str, str]]] = defaultdict(list)
    for pair in pair_instances:
        pairs_by_class[pair[0]].append(pair)
    for class_name in classes:
        pairs = pairs_by_class[class_name]
        rng.shuffle(pairs)
        for pair in pairs:
            _, image_id = pair
            tracked_edges[pair] = network.add_edge(
                class_node[class_name], image_node[image_id], len(pair_instances[pair])
            )

    for image_id in images:
        network.add_edge(image_node[image_id], sink, image_capacities[image_id])

    flow = network.max_flow(source, sink)
    plan = {
        pair: edge.initial_capacity - edge.capacity
        for pair, edge in tracked_edges.items()
        if edge.initial_capacity != edge.capacity
    }
    return flow, plan


def find_balanced_quotas(
    class_totals: Mapping[str, int],
    requested_ratio: float,
    pair_instances: Mapping[Tuple[str, str], Sequence[int]],
    image_capacities: Mapping[str, int],
    seed: int,
) -> Tuple[Dict[str, int], float, bool]:
    requested_quotas = quotas_for_ratio(class_totals, requested_ratio)
    requested_flow, _ = make_flow_plan(
        requested_quotas, pair_instances, image_capacities, seed
    )
    if requested_flow == sum(requested_quotas.values()):
        return requested_quotas, requested_ratio, True

    candidate_rates = {0.0}
    for class_name, total in class_totals.items():
        for count in range(1, requested_quotas[class_name] + 1):
            candidate_rates.add(count / total)
    rates = sorted(rate for rate in candidate_rates if rate <= requested_ratio + 1e-12)

    low, high = 0, len(rates) - 1
    best_rate = 0.0
    best_quotas = quotas_for_ratio(class_totals, 0.0)
    while low <= high:
        middle = (low + high) // 2
        ratio = rates[middle]
        quotas = quotas_for_ratio(class_totals, ratio)
        flow, _ = make_flow_plan(quotas, pair_instances, image_capacities, seed)
        if flow == sum(quotas.values()):
            best_rate = ratio
            best_quotas = quotas
            low = middle + 1
        else:
            high = middle - 1

    return best_quotas, best_rate, False


def validate_split_isolation(voc_root: Path, train_ids: Set[str]) -> Tuple[int, int]:
    main_dir = voc_root / "ImageSets" / "Main"
    counts = []
    for split_name in ("val", "test"):
        split_path = main_dir / f"{split_name}.txt"
        if not split_path.exists():
            counts.append(0)
            continue
        ids = set(read_split(split_path))
        overlap = train_ids & ids
        if overlap:
            examples = ", ".join(sorted(overlap)[:5])
            raise ValueError(
                f"train.txt overlaps {split_name}.txt ({len(overlap)} IDs; e.g. {examples}). "
                "Refusing to modify validation/test annotations."
            )
        counts.append(len(ids))
    return counts[0], counts[1]


def copy_and_write_annotations(
    source_dir: Path,
    output_dir: Path,
    annotations: Mapping[str, Annotation],
    dropped_indices: Mapping[str, Set[int]],
    overwrite: bool,
) -> None:
    source_resolved = source_dir.resolve()
    output_resolved = output_dir.resolve()
    if source_resolved == output_resolved:
        raise ValueError("Output directory must not be the original Annotations directory")
    if source_resolved in output_resolved.parents:
        raise ValueError("Output directory must not be inside the original Annotations directory")
    if output_resolved in source_resolved.parents:
        raise ValueError("Output directory must not contain the original Annotations directory")
    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. Use --overwrite to replace it."
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=str(output_dir.parent))
    )
    shutil.rmtree(temporary)
    backup: Optional[Path] = None
    try:
        shutil.copytree(source_dir, temporary, copy_function=shutil.copy2)
        for image_id, indices in dropped_indices.items():
            if not indices:
                continue
            annotation = annotations[image_id]
            root = annotation.tree.getroot()
            for index in sorted(indices, reverse=True):
                root.remove(annotation.objects[index])
            if not root.findall("object"):
                raise AssertionError(f"Dropping removed every object from {image_id}")
            annotation.tree.write(
                temporary / f"{image_id}.xml", encoding="utf-8", xml_declaration=False
            )

        if output_dir.exists():
            backup = output_dir.with_name(f".{output_dir.name}.backup-{uuid.uuid4().hex}")
            output_dir.rename(backup)
        try:
            temporary.rename(output_dir)
        except Exception:
            if backup is not None:
                backup.rename(output_dir)
            raise
        if backup is not None:
            shutil.rmtree(backup)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def print_statistics(
    output_dir: Path,
    seed: int,
    requested_ratio: float,
    effective_ratio: float,
    requested_feasible: bool,
    class_totals: Mapping[str, int],
    requested_quotas: Mapping[str, int],
    planned_quotas: Mapping[str, int],
    dropped_counts: Mapping[str, int],
    train_image_count: int,
    modified_image_count: int,
    val_image_count: int,
    test_image_count: int,
) -> None:
    total_boxes = sum(class_totals.values())
    total_dropped = sum(dropped_counts.values())
    print("\n=== PASCAL VOC instance dropping statistics ===")
    print(f"output directory       : {output_dir}")
    print(f"seed                   : {seed}")
    print(f"requested drop ratio   : {requested_ratio:.6f}")
    print(f"balanced target ratio  : {effective_ratio:.6f}")
    print(f"requested ratio feasible: {'yes' if requested_feasible else 'no (limited by min-1 constraint)'}")
    print(f"train images           : {train_image_count}")
    print(f"modified train images  : {modified_image_count}")
    print(f"unchanged val images   : {val_image_count}")
    print(f"unchanged test images  : {test_image_count}")
    print(f"train boxes before     : {total_boxes}")
    print(f"train boxes dropped    : {total_dropped}")
    print(f"actual overall ratio   : {total_dropped / total_boxes:.6f}")
    print(f"train boxes remaining  : {total_boxes - total_dropped}")
    print()
    header = f"{'class':<18} {'before':>8} {'requested':>10} {'target':>8} {'dropped':>8} {'actual%':>9} {'remain':>8}"
    print(header)
    print("-" * len(header))
    for class_name in sorted(class_totals):
        before = class_totals[class_name]
        dropped = dropped_counts.get(class_name, 0)
        print(
            f"{class_name:<18} {before:>8} {requested_quotas[class_name]:>10} "
            f"{planned_quotas[class_name]:>8} {dropped:>8} "
            f"{100.0 * dropped / before:>8.3f}% {before - dropped:>8}"
        )


def run(args: argparse.Namespace) -> Path:
    if not 0.0 <= args.drop_ratio <= 1.0:
        raise ValueError("--drop-ratio must be between 0 and 1 inclusive")

    voc_root = args.voc_root.resolve()
    source_dir = voc_root / "Annotations"
    train_split = voc_root / "ImageSets" / "Main" / "train.txt"
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Annotations directory not found: {source_dir}")

    ratio_label = format(args.drop_ratio, ".6g")
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else voc_root / f"Annotations_drop_ratio_{ratio_label}_seed_{args.seed}"
    )

    train_ids = read_split(train_split)
    val_count, test_count = validate_split_isolation(voc_root, set(train_ids))
    annotations = load_train_annotations(source_dir, train_ids)

    class_totals: Counter[str] = Counter()
    pair_instances: DefaultDict[Tuple[str, str], List[int]] = defaultdict(list)
    image_capacities: Dict[str, int] = {}
    for image_id, annotation in annotations.items():
        image_capacities[image_id] = len(annotation.objects) - 1
        for index, class_name in enumerate(annotation.classes):
            class_totals[class_name] += 1
            pair_instances[(class_name, image_id)].append(index)

    requested_quotas = quotas_for_ratio(class_totals, args.drop_ratio)
    planned_quotas, effective_ratio, requested_feasible = find_balanced_quotas(
        class_totals,
        args.drop_ratio,
        pair_instances,
        image_capacities,
        args.seed,
    )
    flow, pair_drop_counts = make_flow_plan(
        planned_quotas, pair_instances, image_capacities, args.seed
    )
    if flow != sum(planned_quotas.values()):
        raise AssertionError("Internal error: final class quotas are not feasible")

    rng = random.Random(args.seed)
    dropped_indices: DefaultDict[str, Set[int]] = defaultdict(set)
    dropped_counts: Counter[str] = Counter()
    for pair in sorted(pair_drop_counts):
        count = pair_drop_counts[pair]
        class_name, image_id = pair
        selected = rng.sample(list(pair_instances[pair]), count)
        dropped_indices[image_id].update(selected)
        dropped_counts[class_name] += count

    for class_name, quota in planned_quotas.items():
        if dropped_counts[class_name] != quota:
            raise AssertionError(f"Internal error: quota mismatch for {class_name}")
    for image_id, indices in dropped_indices.items():
        if len(annotations[image_id].objects) - len(indices) < 1:
            raise AssertionError(f"Internal error: no objects would remain in {image_id}")

    copy_and_write_annotations(
        source_dir, output_dir, annotations, dropped_indices, args.overwrite
    )
    print_statistics(
        output_dir,
        args.seed,
        args.drop_ratio,
        effective_ratio,
        requested_feasible,
        class_totals,
        requested_quotas,
        planned_quotas,
        dropped_counts,
        len(train_ids),
        len(dropped_indices),
        val_count,
        test_count,
    )
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Randomly drop class-balanced object instances from PASCAL VOC train "
            "annotations while retaining at least one object per image."
        )
    )
    parser.add_argument(
        "--voc-root",
        type=Path,
        default=Path("datasets/VOC2007"),
        help="VOC2007 root containing Annotations, JPEGImages, and ImageSets (default: datasets/VOC2007)",
    )
    parser.add_argument(
        "--drop-ratio",
        type=float,
        required=True,
        help="Requested per-class instance drop ratio in [0, 1]",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output annotation directory (default: <VOC_ROOT>/Annotations_drop_ratio_<RATIO>_seed_<SEED>)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory; the source Annotations directory is always protected",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        run(args)
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()
