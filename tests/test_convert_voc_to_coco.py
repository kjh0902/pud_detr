import argparse
import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from scripts.convert_voc_to_coco import run


def write_annotation(path: Path, width: int, height: int, objects):
    root = ET.Element("annotation")
    ET.SubElement(root, "filename").text = f"{path.stem}.jpg"
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"
    for class_name, difficult, bbox_values in objects:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = class_name
        ET.SubElement(obj, "difficult").text = str(difficult)
        bbox = ET.SubElement(obj, "bndbox")
        for key, value in zip(("xmin", "ymin", "xmax", "ymax"), bbox_values):
            ET.SubElement(bbox, key).text = str(value)
    ET.ElementTree(root).write(path, encoding="utf-8")


class ConvertVocToCocoTest(unittest.TestCase):
    def make_dataset(self, root: Path):
        annotations = root / "Annotations"
        split_dir = root / "ImageSets" / "Main"
        annotations.mkdir(parents=True)
        split_dir.mkdir(parents=True)
        samples = {
            "000012": (500, 333, [("car", 0, (156, 97, 351, 270)), ("person", 1, (1, 2, 20, 30))]),
            "000017": (480, 320, [("person", 0, (185, 62, 279, 199))]),
            "000005": (500, 375, [("chair", 0, (10, 11, 50, 70))]),
            "000001": (353, 500, [("dog", 0, (48, 240, 195, 371))]),
        }
        for image_id, (width, height, objects) in samples.items():
            write_annotation(annotations / f"{image_id}.xml", width, height, objects)
        (split_dir / "train.txt").write_text("000012\n000017\n", encoding="utf-8")
        (split_dir / "val.txt").write_text("000005\n", encoding="utf-8")
        (split_dir / "test.txt").write_text("000001\n", encoding="utf-8")
        return annotations

    @staticmethod
    def args(voc_root: Path, output_dir: Path, **overrides):
        values = dict(
            voc_root=voc_root,
            annotations_dir=None,
            output_dir=output_dir,
            drop_annotations_dir=[],
            skip_drop_annotations=False,
            overwrite=False,
        )
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_matches_pud_detr_fields_coordinates_categories_and_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            voc_root = Path(temporary) / "VOC2007"
            self.make_dataset(voc_root)
            output_dir = Path(temporary) / "json"

            run(self.args(voc_root, output_dir, skip_drop_annotations=True))

            train = json.loads((output_dir / "pascal_train.json").read_text(encoding="utf-8"))
            self.assertEqual(
                list(train),
                ["type", "categories", "info", "licenses", "images", "annotations"],
            )
            self.assertEqual([image["id"] for image in train["images"]], [12, 17])
            self.assertEqual([ann["id"] for ann in train["annotations"]], [1, 2, 3])
            self.assertEqual(
                train["annotations"][0],
                {
                    "segmentation": [[155, 96, 155, 270, 351, 270, 351, 96]],
                    "area": 34104,
                    "iscrowd": 0,
                    "image_id": 12,
                    "bbox": [155, 96, 196, 174],
                    "category_id": 6,
                    "id": 1,
                    "ignore": 0,
                },
            )
            self.assertEqual(train["annotations"][1]["ignore"], 1)
            self.assertEqual(train["categories"][0]["name"], "aeroplane")
            self.assertEqual(train["categories"][19]["name"], "tvmonitor")

            val = json.loads((output_dir / "pascal_val.json").read_text(encoding="utf-8"))
            test = json.loads((output_dir / "pascal_test.json").read_text(encoding="utf-8"))
            self.assertEqual([ann["id"] for ann in val["annotations"]], [1])
            self.assertEqual([ann["id"] for ann in test["annotations"]], [1])
            self.assertEqual(
                list(test),
                ["images", "type", "annotations", "categories", "info", "licenses"],
            )

    def test_discovers_and_converts_bbox_drop_annotations(self):
        with tempfile.TemporaryDirectory() as temporary:
            voc_root = Path(temporary) / "VOC2007"
            source = self.make_dataset(voc_root)
            drop_dir = voc_root / "Annotations_drop_ratio_0.3_seed_42"
            drop_dir.mkdir()
            for image_id in ("000012", "000017"):
                (drop_dir / f"{image_id}.xml").write_bytes((source / f"{image_id}.xml").read_bytes())
            tree = ET.parse(drop_dir / "000012.xml")
            tree.getroot().remove(tree.getroot().findall("object")[1])
            tree.write(drop_dir / "000012.xml", encoding="utf-8")
            (drop_dir / "drop_statistics.json").write_text(
                json.dumps({"requested": {"drop_ratio": 0.3}}), encoding="utf-8"
            )
            output_dir = Path(temporary) / "json"

            run(self.args(voc_root, output_dir))

            dropped = json.loads(
                (output_dir / "pascal_train_drop_0.3.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                list(dropped), ["info", "licenses", "images", "annotations", "categories"]
            )
            self.assertEqual([image["id"] for image in dropped["images"]], [12, 17])
            self.assertEqual([ann["id"] for ann in dropped["annotations"]], [1, 2])
            self.assertEqual([ann["image_id"] for ann in dropped["annotations"]], [12, 17])

    def test_refuses_existing_outputs_without_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            voc_root = Path(temporary) / "VOC2007"
            self.make_dataset(voc_root)
            output_dir = Path(temporary) / "json"
            args = self.args(voc_root, output_dir, skip_drop_annotations=True)
            run(args)
            with self.assertRaisesRegex(FileExistsError, "--overwrite"):
                run(args)

    def test_rejects_overlapping_splits(self):
        with tempfile.TemporaryDirectory() as temporary:
            voc_root = Path(temporary) / "VOC2007"
            self.make_dataset(voc_root)
            (voc_root / "ImageSets" / "Main" / "val.txt").write_text(
                "000012\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "overlap"):
                run(self.args(voc_root, Path(temporary) / "json"))


if __name__ == "__main__":
    unittest.main()
