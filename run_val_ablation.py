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
        "epochs",
        "best_validation_ap",
        "status",
        "run_dir",
    )
    with temporary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def find_new_metrics(
    experiment_dir: Path, previous_metrics: set[Path]
) -> Path:
    current_metrics = set(experiment_dir.glob("**/metrics.csv"))
    new_metrics = current_metrics - previous_metrics
    if len(new_metrics) != 1:
        raise RuntimeError(
            f"expected one new metrics.csv below {experiment_dir}, "
            f"found {len(new_metrics)}"
        )
    return new_metrics.pop()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    train_script = args.train_script.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    results_csv = args.results_csv.expanduser().resolve()
    if not train_script.is_file():
        raise FileNotFoundError(f"training script does not exist: {train_script}")

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
                *args.train_args,
            ]
            print(
                f"\n=== weight_p={weight_p:g} reduction={reduction} "
                "epochs=20 ===",
                flush=True,
            )
            completed = subprocess.run(command, check=False)
            row: dict[str, object] = {
                "weight_p": format(weight_p, "g"),
                "reduction": reduction,
                "epochs": 20,
                "best_validation_ap": "",
                "status": "failed",
                "run_dir": "",
            }
            if completed.returncode == 0:
                metrics_path = find_new_metrics(experiment_dir, previous_metrics)
                row.update(
                    best_validation_ap=read_best_validation_ap(metrics_path),
                    status="completed",
                    run_dir=str(metrics_path.parent),
                )
            else:
                had_failure = True
                row["status"] = f"failed_exit_{completed.returncode}"
            rows.append(row)
            write_results(results_csv, rows)
            print(f"Ablation results updated: {results_csv}", flush=True)

            if completed.returncode != 0 and not args.continue_on_error:
                return completed.returncode or 1

    return 1 if had_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
