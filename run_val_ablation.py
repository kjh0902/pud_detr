#!/usr/bin/env python3
"""Run the PUD-DETR weight/reduction validation ablation.

Each Cartesian-product experiment trains for exactly 20 epochs, validates at
the end of every epoch through ``train_pud_detr.py``, and skips test-set
loading and evaluation. The best validation AP from each run is collected in
one CSV file.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Sequence


REDUCTIONS = ("global", "query_wise", "element_wise")
CONTROLLED_TRAIN_OPTIONS = (
    "--method",
    "--weight-p",
    "--reduction",
    "--epochs",
    "--experiment-name",
    "--output-dir",
    "--skip-test",
    "--seed",
    "--seeds",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep weight_p and non-negative reduction using best validation AP. "
            "Pass ordinary train_pud_detr.py arguments after '--'."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--weight-p-values",
        type=float,
        nargs="+",
        required=True,
        help="Positive-risk weights to sweep.",
    )
    parser.add_argument(
        "--reductions",
        choices=REDUCTIONS,
        nargs="+",
        default=list(REDUCTIONS),
        help="Non-negative correction reductions to sweep.",
    )
    parser.add_argument(
        "--experiment-prefix", default="nnpu_val_ablation"
    )
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Run each ablation condition once with this seed.",
    )
    seed_group.add_argument(
        "--seeds",
        type=int,
        nargs=3,
        metavar=("SEED1", "SEED2", "SEED3"),
        help="Run each ablation condition independently with three seeds.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/val_ablation"),
        help="Root passed to each training run.",
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=Path("val_ablation_results.csv"),
    )
    parser.add_argument(
        "--train-script",
        type=Path,
        default=Path(__file__).with_name("train_pud_detr.py"),
    )
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Run the remaining combinations after a failed experiment.",
    )
    parser.add_argument(
        "train_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to train_pud_detr.py after '--'.",
    )
    args = parser.parse_args(argv)

    if any(weight <= 0 for weight in args.weight_p_values):
        parser.error("all --weight-p-values must be positive")
    if args.seeds is not None and len(set(args.seeds)) != 3:
        parser.error("--seeds requires three distinct seed values")
    if args.train_args[:1] == ["--"]:
        args.train_args = args.train_args[1:]
    for token in args.train_args:
        if any(
            token == option or token.startswith(f"{option}=")
            for option in CONTROLLED_TRAIN_OPTIONS
        ):
            parser.error(
                f"{token.split('=', 1)[0]} is controlled by the ablation runner"
            )
    return args


def weight_slug(weight: float) -> str:
    return format(weight, "g").replace("-", "m").replace(".", "p")


def read_best_validation_ap(metrics_path: Path) -> float:
    with metrics_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] == "validation" and row["metric"] == "AP":
                return float(row["value"])
    raise ValueError(f"validation AP is missing from {metrics_path}")


def write_results(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    fieldnames = (
        "weight_p",
        "reduction",
        "seed",
        "epochs",
        "best_validation_ap",
        "validation_ap_mean",
        "validation_ap_std",
        "status",
        "run_dir",
    )
    with temporary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def find_new_metrics(
    experiment_dir: Path,
    previous_metrics: set[Path],
    expected_count: int,
) -> list[Path]:
    current_metrics = set(experiment_dir.glob("**/metrics.csv"))
    new_metrics = current_metrics - previous_metrics
    if len(new_metrics) != expected_count:
        raise RuntimeError(
            f"expected {expected_count} new metrics.csv files below "
            f"{experiment_dir}, "
            f"found {len(new_metrics)}"
        )
    return sorted(new_metrics)


def metrics_by_seed(
    metrics_paths: Sequence[Path], seeds: Sequence[int]
) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for metrics_path in metrics_paths:
        run_dir_name = metrics_path.parent.name
        matches = [
            seed
            for seed in seeds
            if run_dir_name == f"seed_{seed}"
            or run_dir_name.startswith(f"seed_{seed}_run_")
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"cannot associate {metrics_path} with exactly one seed"
            )
        result[matches[0]] = metrics_path
    if set(result) != set(seeds):
        raise RuntimeError("new metrics files do not cover all requested seeds")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    train_script = args.train_script.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    results_csv = args.results_csv.expanduser().resolve()
    if not train_script.is_file():
        raise FileNotFoundError(f"training script does not exist: {train_script}")
    seeds = args.seeds if args.seeds is not None else [args.seed]
    seed_arguments = (
        ["--seeds", *(str(seed) for seed in seeds)]
        if args.seeds is not None
        else ["--seed", str(args.seed)]
    )

    rows: list[dict[str, object]] = []
    had_failure = False
    for weight_p in args.weight_p_values:
        for reduction in args.reductions:
            experiment_name = (
                f"{args.experiment_prefix}_weight_p_{weight_slug(weight_p)}_"
                f"{reduction}"
            )
            experiment_dir = output_dir / experiment_name
            previous_metrics = set(experiment_dir.glob("**/metrics.csv"))
            command = [
                args.python_executable,
                str(train_script),
                "--method",
                "pud",
                "--weight-p",
                format(weight_p, "g"),
                "--reduction",
                reduction,
                "--epochs",
                "20",
                "--experiment-name",
                experiment_name,
                "--output-dir",
                str(output_dir),
                "--skip-test",
                *seed_arguments,
                *args.train_args,
            ]
            print(
                f"\n=== weight_p={weight_p:g} reduction={reduction} "
                f"epochs=20 seeds={','.join(str(seed) for seed in seeds)} ===",
                flush=True,
            )
            completed = subprocess.run(command, check=False)
            condition_rows: list[dict[str, object]] = []
            if completed.returncode == 0:
                new_metrics = find_new_metrics(
                    experiment_dir, previous_metrics, len(seeds)
                )
                seed_metrics = metrics_by_seed(new_metrics, seeds)
                validation_aps = [
                    read_best_validation_ap(seed_metrics[seed]) for seed in seeds
                ]
                validation_ap_mean = statistics.fmean(validation_aps)
                validation_ap_std = statistics.pstdev(validation_aps)
                for seed, validation_ap in zip(seeds, validation_aps):
                    metrics_path = seed_metrics[seed]
                    condition_rows.append(
                        {
                            "weight_p": format(weight_p, "g"),
                            "reduction": reduction,
                            "seed": seed,
                            "epochs": 20,
                            "best_validation_ap": validation_ap,
                            "validation_ap_mean": validation_ap_mean,
                            "validation_ap_std": validation_ap_std,
                            "status": "completed",
                            "run_dir": str(metrics_path.parent),
                        }
                    )
            else:
                had_failure = True
                for seed in seeds:
                    condition_rows.append(
                        {
                            "weight_p": format(weight_p, "g"),
                            "reduction": reduction,
                            "seed": seed,
                            "epochs": 20,
                            "best_validation_ap": "",
                            "validation_ap_mean": "",
                            "validation_ap_std": "",
                            "status": f"failed_exit_{completed.returncode}",
                            "run_dir": "",
                        }
                    )
            rows.extend(condition_rows)
            write_results(results_csv, rows)
            print(f"Ablation results updated: {results_csv}", flush=True)

            if completed.returncode != 0 and not args.continue_on_error:
                return completed.returncode or 1

    return 1 if had_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
