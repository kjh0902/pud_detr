import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock


try:
    import torch
    import train_pud_detr

    from train_pud_detr import (
        DetectionCriterion,
        parse_args,
        pu_sigmoid_focal_loss,
        reduce_non_negative_risk,
        sigmoid_focal_loss_tensor,
    )
except ModuleNotFoundError as exc:
    raise unittest.SkipTest(
        f"training dependencies are unavailable: {exc.name}"
    ) from exc


class NnPuLossTest(unittest.TestCase):
    def test_matched_non_target_class_remains_in_unlabeled_risk(self):
        criterion = DetectionCriterion(
            num_classes=2,
            matcher=None,
            weight_dict={"loss_ce": 1.0},
            method="pud",
            weight_p=5.0,
            reduction="element_wise",
            focal_alpha=0.25,
            focal_gamma=2.0,
            auxiliary_loss=False,
            decoder_layers=1,
        )
        targets = [{"labels": torch.tensor([0]), "boxes": torch.zeros((1, 4))}]
        indices = [(torch.tensor([0]), torch.tensor([0]))]
        num_boxes = torch.tensor(1.0)
        first = {
            "pred_logits": torch.tensor([[[0.2, -10.0], [0.3, -0.4]]]),
            "pred_boxes": torch.zeros((1, 2, 4)),
        }
        second = {name: value.clone() for name, value in first.items()}
        second["pred_logits"][0, 0, 1] = 10.0

        first_loss = criterion.loss_labels(first, targets, indices, num_boxes)[
            "loss_ce"
        ]
        second_loss = criterion.loss_labels(second, targets, indices, num_boxes)[
            "loss_ce"
        ]
        self.assertFalse(torch.allclose(first_loss, second_loss))

    def test_historical_all_zero_target_mask_matches_manual_risk(self):
        inputs = torch.tensor([[[0.2, -0.7], [0.3, -0.4]]])
        targets = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
        num_boxes = torch.tensor(1.0)
        alpha = 0.25
        gamma = 2.0
        weight_p = 5.0
        unlabeled_weight = torch.tensor(2.0)

        probability = inputs.sigmoid()
        background_ce = torch.nn.functional.binary_cross_entropy_with_logits(
            inputs, torch.zeros_like(targets), reduction="none"
        )
        background_focal = (1 - alpha) * background_ce * probability.pow(gamma)
        positive_focal = sigmoid_focal_loss_tensor(inputs, targets, alpha, gamma)
        positive_mask = targets.eq(1)
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
        expected = positive_risk.mean(dim=1).sum() / num_boxes
        expected += (
            negative_risk.mean(dim=1).sum(dim=1).clamp(min=0).sum() / num_boxes
        )

        actual = pu_sigmoid_focal_loss(
            inputs,
            targets,
            num_boxes,
            unlabeled_weight,
            weight_p,
            "global",
            alpha,
            gamma,
        )
        self.assertTrue(torch.allclose(actual, expected))

    def test_cli_exposes_three_reductions_and_skip_test_needs_no_test_paths(self):
        required = [
            "--experiment-name",
            "test",
            "--train-json",
            "train.json",
            "--val-json",
            "val.json",
            "--trainval-image-dir",
            "train-images",
            "--skip-test",
        ]
        single_seed_args = parse_args(required)
        self.assertEqual(single_seed_args.reduction, "global")
        self.assertEqual(single_seed_args.seed, 42)
        self.assertIsNone(single_seed_args.seeds)
        multi_seed_args = parse_args([*required, "--seeds", "7", "11", "19"])
        self.assertEqual(multi_seed_args.seeds, [7, 11, 19])
        for reduction in ("global", "query_wise", "element_wise"):
            self.assertEqual(
                parse_args([*required, "--reduction", reduction]).reduction,
                reduction,
            )
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args([*required, "--reduction", "reduce_clamp"])

    def test_reduction_orders_follow_requested_axes(self):
        risk = torch.tensor(
            [
                [[4.0, -1.0], [-2.0, -1.0]],
                [[-4.0, 1.0], [-2.0, 1.0]],
            ]
        )
        num_boxes = torch.tensor(2.0)

        expected = {
            "element_wise": risk.clamp(min=0).mean(dim=1).sum() / num_boxes,
            "query_wise": risk.mean(dim=1).clamp(min=0).sum() / num_boxes,
            "global": (
                risk.mean(dim=1).sum(dim=1).clamp(min=0).sum() / num_boxes
            ),
        }
        for reduction, value in expected.items():
            self.assertTrue(
                torch.allclose(
                    reduce_non_negative_risk(risk, num_boxes, reduction), value
                )
            )

    def test_global_clamps_each_image_before_batch_reduction(self):
        risk = torch.tensor(
            [
                [[3.0, -1.0], [3.0, -1.0]],
                [[-4.0, 0.0], [-4.0, 0.0]],
            ]
        )
        num_boxes = torch.tensor(2.0)

        actual = reduce_non_negative_risk(risk, num_boxes, "global")
        self.assertTrue(torch.allclose(actual, torch.tensor(1.0)))
        self.assertFalse(
            torch.allclose(actual, risk.mean(dim=1).sum().clamp(min=0) / num_boxes)
        )

    def test_training_entry_point_runs_three_seeds_independently(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)

            def fake_run(args):
                return {
                    "seed": args.seed,
                    "best_validation_ap": args.seed / 1000,
                    "run_dir": str(output_dir / f"seed_{args.seed}"),
                    "best_checkpoint": f"checkpoint_{args.seed}",
                    "test_metrics": None,
                }

            with mock.patch.object(
                train_pud_detr, "run_single_seed", side_effect=fake_run
            ) as run_single_seed, mock.patch.object(
                train_pud_detr.torch.cuda, "is_available", return_value=False
            ):
                return_code = train_pud_detr.main(
                    [
                        "--experiment-name",
                        "multi",
                        "--output-dir",
                        str(output_dir),
                        "--train-json",
                        "train.json",
                        "--val-json",
                        "val.json",
                        "--trainval-image-dir",
                        "images",
                        "--skip-test",
                        "--seeds",
                        "7",
                        "11",
                        "19",
                    ]
                )

            self.assertEqual(return_code, 0)
            self.assertEqual(
                [call.args[0].seed for call in run_single_seed.call_args_list],
                [7, 11, 19],
            )
            self.assertTrue(
                (output_dir / "multi" / "multi_seed_results.csv").is_file()
            )


if __name__ == "__main__":
    unittest.main()
