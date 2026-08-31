import csv
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import run_val_ablation


class RunValAblationTest(unittest.TestCase):
    def test_runner_forces_validation_only_twenty_epoch_sweep(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fake_train = root / "fake_train.py"
            fake_train.write_text(
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
                    args, _ = parser.parse_known_args()
                    assert args.method == "pud"
                    assert args.epochs == "20"
                    assert args.skip_test
                    run_dir = args.output_dir / args.experiment_name / "seed_42"
                    run_dir.mkdir(parents=True)
                    with (run_dir / "metrics.csv").open("w", newline="") as handle:
                        writer = csv.writer(handle)
                        writer.writerow(("split", "metric", "value"))
                        writer.writerow(("validation", "AP", args.weight_p))
                    """
                ),
                encoding="utf-8",
            )
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
            self.assertTrue(all(row["status"] == "completed" for row in rows))
            self.assertTrue(
                all(row["best_validation_ap"] == "1.5" for row in rows)
            )

    def test_controlled_training_options_cannot_be_overridden(self):
        with self.assertRaises(SystemExit):
            run_val_ablation.parse_args(
                ["--weight-p-values", "1", "--", "--epochs", "3"]
            )


if __name__ == "__main__":
    unittest.main()
