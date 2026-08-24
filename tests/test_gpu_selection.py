import os
import unittest
from unittest.mock import patch

from scripts.gpu_selection import (
    DEFAULT_DETERMINISTIC,
    concrete_cuda_index,
    configure_cuda_visibility,
    device_argument_from_argv,
    lightning_deterministic_setting,
    normalize_device_argument,
)


class GpuSelectionTest(unittest.TestCase):
    def test_deformable_detr_does_not_use_strict_determinism(self):
        self.assertFalse(DEFAULT_DETERMINISTIC)
        self.assertFalse(lightning_deterministic_setting(False))
        self.assertEqual(lightning_deterministic_setting(True), "warn")

    def test_accepts_numeric_and_legacy_cuda_device_forms(self):
        self.assertEqual(normalize_device_argument("0"), "0")
        self.assertEqual(normalize_device_argument("1"), "1")
        self.assertEqual(normalize_device_argument("cuda:1"), "1")
        self.assertEqual(concrete_cuda_index("cuda:0"), 0)
        self.assertIsNone(concrete_cuda_index("auto"))

    def test_rejects_invalid_or_negative_device_ids(self):
        for value in ("gpu", "cuda:", "cuda:-1", "-1", "0,1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_device_argument(value)

    def test_reads_both_argument_styles_and_uses_last_value(self):
        self.assertEqual(device_argument_from_argv(["--device", "0"]), "0")
        self.assertEqual(device_argument_from_argv(["--device=1"]), "1")
        self.assertEqual(
            device_argument_from_argv(["--device", "0", "--device=cuda:1"]),
            "cuda:1",
        )

    def test_concrete_device_restricts_cuda_visibility(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}, clear=False):
            selected = configure_cuda_visibility(["--device", "1"])
            self.assertEqual(selected, 1)
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "1")

    def test_generic_device_does_not_change_existing_visibility(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"}, clear=False):
            selected = configure_cuda_visibility(["--device", "cuda"])
            self.assertIsNone(selected)
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "1")


if __name__ == "__main__":
    unittest.main()
