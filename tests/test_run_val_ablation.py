import csv
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import run_val_ablation


class RunValAblationTest(unittest.TestCase):
    @staticmethod
    def write_fake_train(path: Path) -> None:
        path.write_text(
            textwrap.dedent(
                """
                import argparse
                import csv
                from pathlib import Path

                parser = argparse.ArgumentParser()
                parser.add_argument("--method", required=True)
                parser.add_argument("--weight-p", required=True)
                parser.add_argument("--reduction", required=True)
                parser.add_argument("--epochs", required=True)
                parser.add_argument("--experiment-name", required=True)
                parser.add_argument("--output-dir", type=Path, required=True)
                parser.add_argument("--skip-test", action="store_true")
                seed_group = parser.add_mutually_exclusive_group(required=True)
                seed_group.add_argument("--seed", type=int)
                seed_group.add_argument("--seeds", type=int, nargs=3)
                args, _ = parser.parse_known_args()
                assert args.method == "pud"
                assert args.epochs == "20"
                assert args.skip_test
                seeds = args.seeds if args.seeds is not None else [args.seed]
                for seed in seeds:
                    run_dir = args.output_dir / args.experiment_name / f"seed_{seed}"
                    run_dir.mkdir(parents=True)
                    validation_ap = float(args.weight_p) + seed / 1000
                    with (run_dir / "metrics.csv").open("w", newline="") as handle:
                        writer = csv.writer(handle)
                        writer.writerow(("split", "metric", "value"))
                        writer.writerow(("validation", "AP", validation_ap))
                """
            ),
            encoding="utf-8",
        )

    def test_runner_forces_validation_only_twenty_epoch_sweep(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fake_train = root / "fake_train.py"
            self.write_fake_train(fake_train)
            output_dir = root / "outputs"
            results_csv = root / "results.csv"
            return_code = run_val_ablation.main(
                [
                    "--weight-p-values",
                    "1.5",
                    "--reductions",
                    "global",
                    "element_wise",
                    "--train-script",
                    str(fake_train),
                    "--python-executable",
                    sys.executable,
                    "--output-dir",
                    str(output_dir),
                    "--results-csv",
                    str(results_csv),
                    "--",
                    "--train-json",
                    "train.json",
                ]
            )

            self.assertEqual(return_code, 0)
            with results_csv.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(
                {row["reduction"] for row in rows}, {"global", "element_wise"}
            )
            self.assertTrue(all(row["epochs"] == "20" for row in rows))
            self.assertTrue(all(row["seed"] == "42" for row in rows))
            self.assertTrue(all(row["status"] == "completed" for row in rows))
            self.assertTrue(
                all(row["best_validation_ap"] == "1.542" for row in rows)
            )
            self.assertTrue(
                all(row["validation_ap_mean"] == "1.542" for row in rows)
            )
            self.assertTrue(all(row["validation_ap_std"] == "0.0" for row in rows))

    def test_three_seed_mode_records_independent_runs_and_statistics(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fake_train = root / "fake_train.py"
            self.write_fake_train(fake_train)
            results_csv = root / "results.csv"

            return_code = run_val_ablation.main(
                [
                    "--weight-p-values",
                    "2",
                    "--reductions",
                    "query_wise",
                    "--seeds",
                    "7",
                    "11",
                    "19",
                    "--train-script",
                    str(fake_train),
                    "--python-executable",
                    sys.executable,
                    "--output-dir",
                    str(root / "outputs"),
                    "--results-csv",
                    str(results_csv),
                    "--",
                    "--train-json",
                    "train.json",
                ]
            )

            self.assertEqual(return_code, 0)
            with results_csv.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["seed"] for row in rows], ["7", "11", "19"])
            self.assertEqual(
                [row["best_validation_ap"] for row in rows],
                ["2.007", "2.011", "2.019"],
            )
            self.assertTrue(
                all(
                    abs(float(row["validation_ap_mean"]) - 2.0123333333333333)
                    < 1e-12
                    for row in rows
                )
            )
            self.assertTrue(all(Path(row["run_dir"]).is_dir() for row in rows))

    def test_controlled_training_options_cannot_be_overridden(self):
        with self.assertRaises(SystemExit):
            run_val_ablation.parse_args(
                ["--weight-p-values", "1", "--", "--epochs", "3"]
            )

    def test_three_seed_mode_requires_distinct_seeds(self):
        with self.assertRaises(SystemExit):
            run_val_ablation.parse_args(
                ["--weight-p-values", "1", "--seeds", "7", "7", "19"]
            )


if __name__ == "__main__":
    unittest.main()
