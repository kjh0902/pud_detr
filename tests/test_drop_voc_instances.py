import argparse
import hashlib
import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from scripts.drop_voc_instances import run


def write_annotation(path: Path, classes):
    root = ET.Element("annotation")
    ET.SubElement(root, "filename").text = f"{path.stem}.jpg"
    for class_name in classes:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = class_name
        bbox = ET.SubElement(obj, "bndbox")
        for key, value in (("xmin", "1"), ("ymin", "2"), ("xmax", "10"), ("ymax", "20")):
            ET.SubElement(bbox, key).text = value
    ET.ElementTree(root).write(path, encoding="utf-8")


def object_classes(path: Path):
    return [obj.findtext("name") for obj in ET.parse(path).getroot().findall("object")]


def digest(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DropVocInstancesTest(unittest.TestCase):
    def make_dataset(self, root: Path):
        annotations = root / "Annotations"
        main = root / "ImageSets" / "Main"
        annotations.mkdir(parents=True)
        main.mkdir(parents=True)

        samples = {
            "train1": ["cat", "dog"],
            "train2": ["cat", "cat"],
            "train3": ["dog", "dog", "dog"],
            "val1": ["cat", "dog"],
            "test1": ["dog"],
        }
        for image_id, classes in samples.items():
            write_annotation(annotations / f"{image_id}.xml", classes)
        (main / "train.txt").write_text("train1\ntrain2\ntrain3\n")
        (main / "val.txt").write_text("val1\n")
        (main / "test.txt").write_text("test1\n")
        return samples

    def test_balanced_drop_preserves_source_and_non_train_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            voc_root = Path(temporary) / "VOC2007"
            samples = self.make_dataset(voc_root)
            source_hashes = {
                path.name: digest(path) for path in (voc_root / "Annotations").glob("*.xml")
            }
            output = voc_root / "Annotations_drop_test"

            run(
                argparse.Namespace(
                    voc_root=voc_root,
                    drop_ratio=0.5,
                    seed=7,
                    output_dir=output,
                    overwrite=False,
                )
            )

            self.assertEqual(
                source_hashes,
                {path.name: digest(path) for path in (voc_root / "Annotations").glob("*.xml")},
            )
            self.assertEqual(digest(voc_root / "Annotations" / "val1.xml"), digest(output / "val1.xml"))
            self.assertEqual(digest(voc_root / "Annotations" / "test1.xml"), digest(output / "test1.xml"))

            remaining = []
            for image_id in ("train1", "train2", "train3"):
                classes = object_classes(output / f"{image_id}.xml")
                self.assertGreaterEqual(len(classes), 1)
                remaining.extend(classes)
            original = [name for image_id in ("train1", "train2", "train3") for name in samples[image_id]]
            # 7 * 0.5 rounds to a global target of 4. Class balance is the
            # tertiary objective, so the best feasible allocation is 2/2.
            self.assertEqual(original.count("cat") - remaining.count("cat"), 2)
            self.assertEqual(original.count("dog") - remaining.count("dog"), 2)

            statistics = json.loads(
                (output / "drop_statistics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(statistics["schema_version"], 1)
            self.assertEqual(statistics["requested"]["drop_ratio"], 0.5)
            self.assertTrue(statistics["requested"]["feasible"])
            self.assertEqual(statistics["images"]["train"], 3)
            self.assertEqual(statistics["images"]["unchanged_val"], 1)
            self.assertEqual(statistics["images"]["unchanged_test"], 1)
            self.assertEqual(statistics["boxes"]["before"], 7)
            self.assertEqual(statistics["boxes"]["dropped"], 4)
            self.assertEqual(statistics["boxes"]["remaining"], 3)
            self.assertAlmostEqual(statistics["boxes"]["actual_drop_ratio"], 4 / 7)
            self.assertEqual(statistics["classes"]["cat"]["dropped"], 2)
            self.assertEqual(statistics["classes"]["dog"]["remaining"], 2)

    def test_global_ratio_has_priority_over_class_balance(self):
        with tempfile.TemporaryDirectory() as temporary:
            voc_root = Path(temporary) / "VOC2007"
            annotations = voc_root / "Annotations"
            main = voc_root / "ImageSets" / "Main"
            annotations.mkdir(parents=True)
            main.mkdir(parents=True)
            write_annotation(annotations / "cat1.xml", ["cat"])
            write_annotation(annotations / "cat2.xml", ["cat"])
            write_annotation(annotations / "dogs.xml", ["dog"] * 5)
            (main / "train.txt").write_text("cat1\ncat2\ndogs\n")
            (main / "val.txt").write_text("")
            (main / "test.txt").write_text("")
            output = voc_root / "Annotations_drop_test"

            run(
                argparse.Namespace(
                    voc_root=voc_root,
                    drop_ratio=0.5,
                    seed=5,
                    output_dir=output,
                    overwrite=False,
                )
            )

            # Four of seven boxes must be dropped (nearest integer to 50%),
            # even though the singleton cat boxes cannot be removed.
            remaining_count = sum(
                len(object_classes(output / f"{image_id}.xml"))
                for image_id in ("cat1", "cat2", "dogs")
            )
            self.assertEqual(7 - remaining_count, 4)
            self.assertEqual(len(object_classes(output / "dogs.xml")), 1)

    def test_impossible_ratio_is_reduced_to_keep_one_box_per_image(self):
        with tempfile.TemporaryDirectory() as temporary:
            voc_root = Path(temporary) / "VOC2007"
            annotations = voc_root / "Annotations"
            main = voc_root / "ImageSets" / "Main"
            annotations.mkdir(parents=True)
            main.mkdir(parents=True)
            write_annotation(annotations / "one.xml", ["cat"])
            write_annotation(annotations / "two.xml", ["dog"])
            (main / "train.txt").write_text("one\ntwo\n")
            (main / "val.txt").write_text("")
            (main / "test.txt").write_text("")
            output = voc_root / "Annotations_drop_test"

            run(
                argparse.Namespace(
                    voc_root=voc_root,
                    drop_ratio=1.0,
                    seed=11,
                    output_dir=output,
                    overwrite=False,
                )
            )

            self.assertEqual(object_classes(output / "one.xml"), ["cat"])
            self.assertEqual(object_classes(output / "two.xml"), ["dog"])

    def test_same_seed_is_reproducible(self):
        with tempfile.TemporaryDirectory() as temporary:
            voc_root = Path(temporary) / "VOC2007"
            self.make_dataset(voc_root)
            outputs = [voc_root / "output_a", voc_root / "output_b"]
            for output in outputs:
                run(
                    argparse.Namespace(
                        voc_root=voc_root,
                        drop_ratio=0.5,
                        seed=99,
                        output_dir=output,
                        overwrite=False,
                    )
                )

            for image_id in ("train1", "train2", "train3"):
                self.assertEqual(
                    digest(outputs[0] / f"{image_id}.xml"),
                    digest(outputs[1] / f"{image_id}.xml"),
                )

    def test_output_cannot_contain_or_be_inside_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            voc_root = Path(temporary) / "VOC2007"
            self.make_dataset(voc_root)
            common = dict(
                voc_root=voc_root,
                drop_ratio=0.25,
                seed=0,
                overwrite=True,
            )
            with self.assertRaisesRegex(ValueError, "inside the original"):
                run(
                    argparse.Namespace(
                        output_dir=voc_root / "Annotations" / "nested", **common
                    )
                )
            with self.assertRaisesRegex(ValueError, "contain the original"):
                run(argparse.Namespace(output_dir=voc_root, **common))

    def test_train_overlap_with_validation_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            voc_root = Path(temporary) / "VOC2007"
            self.make_dataset(voc_root)
            (voc_root / "ImageSets" / "Main" / "val.txt").write_text("train1\n")
            with self.assertRaisesRegex(ValueError, "overlaps val.txt"):
                run(
                    argparse.Namespace(
                        voc_root=voc_root,
                        drop_ratio=0.25,
                        seed=0,
                        output_dir=voc_root / "output",
                        overwrite=False,
                    )
                )


if __name__ == "__main__":
    unittest.main()
