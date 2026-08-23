# PASCAL VOC instance dropping

`scripts/drop_voc_instances.py` creates a separate PASCAL VOC annotation directory
with class-balanced random bounding-box instance dropping.

## Behavior

- Reads the training image IDs only from `ImageSets/Main/train.txt`.
- Copies every original XML into a new directory before changing anything.
- Keeps validation and test XML files byte-for-byte unchanged.
- Targets the same drop ratio independently for every class.
- Uses a max-flow constraint so every training image retains at least one object.
- If the requested balanced ratio is impossible, uses the largest common feasible
  ratio and reports the reduction.
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
```

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
