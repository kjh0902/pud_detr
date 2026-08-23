#!/usr/bin/env python3
"""Create priority-constrained, instance-dropped PASCAL VOC annotations.

Only annotations listed in ImageSets/Main/train.txt are modified. Every source
annotation is first copied to a separate output directory, so validation and
test annotations remain byte-for-byte identical to the originals.
"""

from __future__ import annotations

import argparse
import heapq
import random
import shutil
import tempfile
import uuid
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
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
    __slots__ = ("to", "reverse", "capacity", "initial_capacity", "cost")

    def __init__(self, to: int, reverse: int, capacity: int, cost: int) -> None:
        self.to = to
        self.reverse = reverse
        self.capacity = capacity
        self.initial_capacity = capacity
        self.cost = cost


class MinCostFlow:
    """Successive-shortest-path flow with integer costs and capacities."""

    def __init__(self, node_count: int) -> None:
        self.graph: List[List[Edge]] = [[] for _ in range(node_count)]

    def add_edge(self, source: int, target: int, capacity: int, cost: int = 0) -> Edge:
        forward = Edge(target, len(self.graph[target]), capacity, cost)
        backward = Edge(source, len(self.graph[source]), 0, -cost)
        self.graph[source].append(forward)
        self.graph[target].append(backward)
        return forward

    def send(self, source: int, sink: int, requested_flow: int) -> int:
        node_count = len(self.graph)
        potential = [0] * node_count
        total_flow = 0
        infinity = 1 << 62

        while total_flow < requested_flow:
            distance = [infinity] * node_count
            previous_node = [-1] * node_count
            previous_edge = [-1] * node_count
            distance[source] = 0
            queue: List[Tuple[int, int]] = [(0, source)]

            while queue:
                current_distance, node = heapq.heappop(queue)
                if current_distance != distance[node]:
                    continue
                for edge_index, edge in enumerate(self.graph[node]):
                    if edge.capacity <= 0:
                        continue
                    reduced_cost = edge.cost + potential[node] - potential[edge.to]
                    candidate = current_distance + reduced_cost
                    if candidate < distance[edge.to]:
                        distance[edge.to] = candidate
                        previous_node[edge.to] = node
                        previous_edge[edge.to] = edge_index
                        heapq.heappush(queue, (candidate, edge.to))

            if distance[sink] == infinity:
                break
            for node, value in enumerate(distance):
                if value != infinity:
                    potential[node] += value

            pushed = requested_flow - total_flow
            node = sink
            while node != source:
                parent = previous_node[node]
                edge = self.graph[parent][previous_edge[node]]
                pushed = min(pushed, edge.capacity)
                node = parent
            node = sink
            while node != source:
                parent = previous_node[node]
                edge = self.graph[parent][previous_edge[node]]
                edge.capacity -= pushed
                self.graph[node][edge.reverse].capacity += pushed
                node = parent
            total_flow += pushed

        return total_flow


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


def make_priority_flow_plan(
    class_totals: Mapping[str, int],
    requested_ratio: float,
    requested_drop_count: int,
    pair_instances: Mapping[Tuple[str, str], Sequence[int]],
    image_capacities: Mapping[str, int],
    seed: int,
) -> Tuple[int, Dict[Tuple[str, str], int]]:
    """Meet the global count first, then minimize squared class-rate error."""
    rng = random.Random(seed)
    classes = list(class_totals)
    images = list(image_capacities)
    rng.shuffle(classes)
    rng.shuffle(images)

    source = 0
    class_node = {name: index + 1 for index, name in enumerate(classes)}
    image_offset = 1 + len(classes)
    image_node = {name: image_offset + index for index, name in enumerate(images)}
    sink = image_offset + len(images)
    network = MinCostFlow(sink + 1)

    # Each unit edge carries the marginal increase in squared deviation from
    # the requested class rate. A common offset makes all initial costs
    # non-negative without changing the optimum for a fixed total flow.
    cost_scale = 1_000_000_000
    cost_offset = cost_scale + 1
    for class_name in classes:
        total = class_totals[class_name]
        for dropped_count in range(1, total + 1):
            previous_error = ((dropped_count - 1) / total - requested_ratio) ** 2
            next_error = (dropped_count / total - requested_ratio) ** 2
            marginal_cost = round((next_error - previous_error) * cost_scale)
            network.add_edge(
                source, class_node[class_name], 1, marginal_cost + cost_offset
            )

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

    flow = network.send(source, sink, requested_drop_count)
    plan = {
        pair: edge.initial_capacity - edge.capacity
        for pair, edge in tracked_edges.items()
        if edge.initial_capacity != edge.capacity
    }
    return flow, plan


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
    requested_drop_count: int,
    requested_feasible: bool,
    class_totals: Mapping[str, int],
    dropped_counts: Mapping[str, int],
    maximum_droppable: int,
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
    print(f"requested drop boxes   : {requested_drop_count}")
    print(f"requested ratio feasible: {'yes' if requested_feasible else 'no (limited by min-1 constraint)'}")
    print(f"maximum droppable boxes: {maximum_droppable}")
    print(f"train images           : {train_image_count}")
    print(f"modified train images  : {modified_image_count}")
    print(f"unchanged val images   : {val_image_count}")
    print(f"unchanged test images  : {test_image_count}")
    print(f"train boxes before     : {total_boxes}")
    print(f"train boxes dropped    : {total_dropped}")
    print(f"actual overall ratio   : {total_dropped / total_boxes:.6f}")
    print(f"train boxes remaining  : {total_boxes - total_dropped}")
    print()
    header = f"{'class':<18} {'before':>8} {'ideal':>10} {'dropped':>8} {'actual%':>9} {'delta(pp)':>10} {'remain':>8}"
    print(header)
    print("-" * len(header))
    for class_name in sorted(class_totals):
        before = class_totals[class_name]
        dropped = dropped_counts.get(class_name, 0)
        actual_percent = 100.0 * dropped / before
        ideal = before * requested_ratio
        print(
            f"{class_name:<18} {before:>8} {ideal:>10.2f} {dropped:>8} "
            f"{actual_percent:>8.3f}% {actual_percent - 100.0 * requested_ratio:>+9.3f} "
            f"{before - dropped:>8}"
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

    total_boxes = sum(class_totals.values())
    requested_drop_count = int(total_boxes * args.drop_ratio + 0.5)
    maximum_droppable = sum(image_capacities.values())
    flow, pair_drop_counts = make_priority_flow_plan(
        class_totals,
        args.drop_ratio,
        requested_drop_count,
        pair_instances,
        image_capacities,
        args.seed,
    )
    expected_flow = min(requested_drop_count, maximum_droppable)
    if flow != expected_flow:
        raise AssertionError(
            f"Internal error: expected {expected_flow} droppable boxes, planned {flow}"
        )
    requested_feasible = flow == requested_drop_count

    rng = random.Random(args.seed)
    dropped_indices: DefaultDict[str, Set[int]] = defaultdict(set)
    dropped_counts: Counter[str] = Counter()
    for pair in sorted(pair_drop_counts):
        count = pair_drop_counts[pair]
        class_name, image_id = pair
        selected = rng.sample(list(pair_instances[pair]), count)
        dropped_indices[image_id].update(selected)
        dropped_counts[class_name] += count

    if sum(dropped_counts.values()) != flow:
        raise AssertionError("Internal error: total drop count does not match flow")
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
        requested_drop_count,
        requested_feasible,
        class_totals,
        dropped_counts,
        maximum_droppable,
        len(train_ids),
        len(dropped_indices),
        val_count,
        test_count,
    )
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Randomly drop object instances from PASCAL VOC train annotations, "
            "prioritizing one object per image, the global ratio, then class balance."
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
        help="Requested global train-instance drop ratio in [0, 1]",
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
