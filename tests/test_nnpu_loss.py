import contextlib
import io
import unittest


try:
    import torch

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
    def test_matched_non_target_class_is_excluded_from_unlabeled_risk(self):
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

        self.assertTrue(
            torch.allclose(
                criterion.loss_labels(first, targets, indices, num_boxes)["loss_ce"],
                criterion.loss_labels(second, targets, indices, num_boxes)["loss_ce"],
            )
        )

    def test_positive_correction_and_unmatched_terms_match_manual_risk(self):
        inputs = torch.tensor([[[0.2, -0.7], [0.3, -0.4]]])
        targets = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
        unlabeled_mask = torch.tensor(
            [[[False, False], [True, True]]], dtype=torch.bool
        )
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
        positive_focal = sigmoid_focal_loss_tensor(
            inputs, targets, alpha, gamma
        )

        positive_risk = torch.zeros_like(inputs)
        positive_risk[0, 0, 0] = positive_focal[0, 0, 0] * weight_p
        negative_risk = torch.zeros_like(inputs)
        negative_risk[0, 0, 0] = -background_focal[0, 0, 0] * weight_p
        negative_risk[0, 1, :] = background_focal[0, 1, :] * unlabeled_weight
        expected = positive_risk.mean(dim=1).sum() / num_boxes
        expected += (
            negative_risk.mean(dim=1).sum(dim=1).clamp(min=0).sum() / num_boxes
        )

        actual = pu_sigmoid_focal_loss(
            inputs,
            targets,
            num_boxes,
            unlabeled_mask,
            unlabeled_weight,
            weight_p,
            "global",
            alpha,
            gamma,
        )
        self.assertTrue(torch.allclose(actual, expected))
        negative_without_positive_correction = torch.where(
            unlabeled_mask,
            background_focal * unlabeled_weight,
            torch.zeros_like(background_focal),
        )
        without_positive_correction = positive_risk.mean(dim=1).sum() / num_boxes
        without_positive_correction += (
            negative_without_positive_correction.mean(dim=1)
            .sum(dim=1)
            .clamp(min=0)
            .sum()
            / num_boxes
        )
        self.assertFalse(torch.allclose(actual, without_positive_correction))

    def test_cli_exposes_only_the_three_paper_reductions(self):
        required = [
            "--experiment-name",
            "test",
            "--train-json",
            "train.json",
            "--val-json",
            "val.json",
            "--test-json",
            "test.json",
            "--trainval-image-dir",
            "train-images",
            "--test-image-dir",
            "test-images",
        ]
        self.assertEqual(parse_args(required).reduction, "global")
        for reduction in ("global", "query_wise", "element_wise"):
            self.assertEqual(
                parse_args([*required, "--reduction", reduction]).reduction,
                reduction,
            )
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args([*required, "--reduction", "reduce_clamp"])

    def test_reduction_orders_follow_the_requested_axes(self):
        risk = torch.tensor(
            [
                [[4.0, -1.0], [-2.0, -1.0]],
                [[-4.0, 1.0], [-2.0, 1.0]],
            ]
        )
        num_boxes = torch.tensor(2.0)

        element_wise = risk.clamp(min=0).mean(dim=1).sum() / num_boxes
        query_wise = risk.mean(dim=1).clamp(min=0).sum() / num_boxes
        global_risk = risk.mean(dim=1).sum(dim=1).clamp(min=0).sum() / num_boxes

        self.assertTrue(
            torch.allclose(
                reduce_non_negative_risk(risk, num_boxes, "element_wise"),
                element_wise,
            )
        )
        self.assertTrue(
            torch.allclose(
                reduce_non_negative_risk(risk, num_boxes, "query_wise"),
                query_wise,
            )
        )
        self.assertTrue(
            torch.allclose(
                reduce_non_negative_risk(risk, num_boxes, "global"), global_risk
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

    def test_loss_labels_restores_query_scaling_for_pn_and_pud(self):
        outputs = {
            "pred_logits": torch.zeros((1, 2, 1)),
            "pred_boxes": torch.zeros((1, 2, 4)),
        }
        targets = [{"labels": torch.tensor([0]), "boxes": torch.zeros((1, 4))}]
        indices = [(torch.tensor([0]), torch.tensor([0]))]
        num_boxes = torch.tensor(1.0)

        for method in ("pn", "pud"):
            criterion = DetectionCriterion(
                num_classes=1,
                matcher=None,
                weight_dict={"loss_ce": 1.0},
                method=method,
                weight_p=5.0,
                reduction="global",
                focal_alpha=0.25,
                focal_gamma=2.0,
                auxiliary_loss=False,
                decoder_layers=1,
            )
            actual = criterion.loss_labels(outputs, targets, indices, num_boxes)[
                "loss_ce"
            ]

            target_onehot = torch.tensor([[[1.0], [0.0]]])
            if method == "pn":
                expected = sigmoid_focal_loss_tensor(
                    outputs["pred_logits"], target_onehot, 0.25, 2.0
                ).mean(dim=1).sum() / num_boxes * outputs["pred_logits"].shape[1]
            else:
                expected = pu_sigmoid_focal_loss(
                    outputs["pred_logits"],
                    target_onehot,
                    num_boxes,
                    torch.tensor([[[False], [True]]]),
                    torch.tensor(2.0),
                    5.0,
                    "global",
                    0.25,
                    2.0,
                ) * outputs["pred_logits"].shape[1]
            self.assertTrue(torch.allclose(actual, expected))


if __name__ == "__main__":
    unittest.main()
