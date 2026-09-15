"""Test the actual loss/CLI definitions without loading model or dataset packages."""

import argparse
import ast
import contextlib
import io
import unittest
from pathlib import Path

from scripts.gpu_selection import DEFAULT_DETERMINISTIC, normalize_device_argument

try:
    import torch
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("PyTorch is required for loss tests") from exc


def load_definitions(filename):
    path = Path(__file__).resolve().parents[1] / filename
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {
        "sigmoid_focal_loss_tensor", "pn_sigmoid_focal_loss",
        "DetectionCriterion", "parse_args", "validate_args",
    }
    tree.body = [
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        *(node for node in tree.body if getattr(node, "name", None) in names),
    ]
    ast.fix_missing_locations(tree)
    namespace = dict(
        torch=torch, F=torch.nn.functional, nn=torch.nn, argparse=argparse,
        Path=Path, DEFAULT_DETERMINISTIC=DEFAULT_DETERMINISTIC,
        normalize_device_argument=normalize_device_argument,
    )
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace


class NegativeAblationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = load_definitions("train_pud_detr.py")
        cls.ablation = load_definitions("train_pud_detr_negative_ablation.py")

    def test_loss_and_gradient_for_all_requested_weights(self):
        logits = torch.tensor([[[0.2, -0.7], [0.3, -0.4]]], requires_grad=True)
        targets = torch.tensor([[[1., 0.], [0., 0.]]])
        boxes = torch.tensor(1.)
        baseline = self.original["pn_sigmoid_focal_loss"](logits, targets, boxes, .25, 2.)
        base_grad = torch.autograd.grad(baseline, logits)[0]
        focal = self.original["sigmoid_focal_loss_tensor"](logits, targets, .25, 2.)
        for weight in (1., .75, .5, .25, 0.):
            with self.subTest(weight=weight):
                actual = self.ablation["pn_sigmoid_focal_loss"](
                    logits, targets, boxes, .25, 2., negative_weight=weight,
                )
                expected = (focal[targets == 1].sum() + weight * focal[targets == 0].sum()) / 2
                torch.testing.assert_close(actual, expected)
                gradient = torch.autograd.grad(actual, logits)[0]
                torch.testing.assert_close(gradient[targets == 1], base_grad[targets == 1])
                torch.testing.assert_close(gradient[targets == 0], weight * base_grad[targets == 0])
                if weight == 1.:
                    self.assertTrue(torch.equal(actual, baseline))
                    self.assertTrue(torch.equal(gradient, base_grad))
                if weight == 0.:
                    self.assertEqual(torch.count_nonzero(gradient[targets == 0]).item(), 0)

    def test_criterion_includes_matched_non_target_class(self):
        targets = [{"labels": torch.tensor([0])}]
        indices = [(torch.tensor([0]), torch.tensor([0]))]
        for weight in (1., .75, .5, .25, 0.):
            criterion = self.ablation["DetectionCriterion"](
                num_classes=2, matcher=None, weight_dict={}, method="pn",
                weight_p=5., reduction="global", focal_alpha=.25, focal_gamma=2.,
                auxiliary_loss=False, decoder_layers=1, negative_weight=weight,
            )
            logits = torch.tensor([[[0.2, 0.7], [0.3, -0.4]]], requires_grad=True)
            loss = criterion.loss_labels({"pred_logits": logits}, targets, indices, torch.tensor(1.))["loss_ce"]
            onehot = torch.tensor([[[1., 0.], [0., 0.]]])
            focal = self.original["sigmoid_focal_loss_tensor"](logits, onehot, .25, 2.)
            expected = focal[0, 0, 0] + weight * (focal[0, 0, 1] + focal[0, 1].sum())
            torch.testing.assert_close(loss, expected)

    def test_cli_default_values_and_invalid_combinations(self):
        base = ["--experiment-name", "test", "--train-json", "train.json",
                "--val-json", "val.json", "--trainval-image-dir", "images", "--skip-test"]
        parse = self.ablation["parse_args"]
        self.assertEqual(parse(base).negative_weight, 1.)
        for weight in (1., .75, .5, .25, 0.):
            self.assertEqual(parse(base + ["--method", "pn", "--negative-weight", str(weight)]).negative_weight, weight)
        for extra in (["--negative-weight", ".5"], *(
            ["--method", "pn", "--negative-weight", value]
            for value in ("-0.1", "1.1", "nan", "inf")
        )):
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                parse(base + extra)


if __name__ == "__main__":
    unittest.main()
