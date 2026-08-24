# PASCAL VOC instance dropping

`scripts/drop_voc_instances.py` creates a separate PASCAL VOC annotation directory
with priority-constrained random bounding-box instance dropping.

## Behavior

- Reads the training image IDs only from `ImageSets/Main/train.txt`.
- Copies every original XML into a new directory before changing anything.
- Keeps validation and test XML files byte-for-byte unchanged.
- Applies constraints in this strict priority order:
  1. Every training image retains at least one object.
  2. The requested global drop ratio is met to the nearest whole box whenever feasible.
  3. Class-specific drop rates are made as similar as possible by minimizing their
     squared deviation from the requested rate.
- If priority 1 makes the requested global count impossible, drops the maximum
  feasible number and reports the shortfall.
- Prints per-class and aggregate statistics after completion.

## Usage

Run from the repository root:

```bash
python3 scripts/drop_voc_instances.py \
  --voc-root datasets/VOC2007 \
  --drop-ratio 0.3 \
  --seed 42
```

The default output for this example is:

```text
datasets/VOC2007/Annotations_drop_ratio_0.3_seed_42
datasets/VOC2007/Annotations_drop_ratio_0.3_seed_42/drop_statistics.json
```

`drop_statistics.json` records the requested and actual drop ratios, dropped and
remaining box counts, split image counts, modified training image count, and
per-class drop statistics. The same summary is also printed to the terminal.

Use that directory as the training annotation directory. The original
`datasets/VOC2007/Annotations` directory is never modified.

An explicit destination can be supplied when needed:

```bash
python3 scripts/drop_voc_instances.py \
  --voc-root datasets/VOC2007 \
  --drop-ratio 0.5 \
  --seed 123 \
  --output-dir datasets/VOC2007/Annotations_drop_50pct_seed_123
```

The command refuses to replace an existing output directory unless
`--overwrite` is explicitly provided.

## Tests

```bash
python3 -m unittest discover -s tests -v
```
