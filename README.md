# PASCAL VOC instance dropping

Install the pinned training dependencies with:

```bash
python3 -m pip install -r requirements.txt
```

The requirements select the official PyTorch CUDA 12.6 wheels for the server's
NVIDIA 560.35.03 driver and RTX 3090 GPUs.

Select one physical GPU directly with `--device 0` or `--device 1`. The chosen
ID is applied to `CUDA_VISIBLE_DEVICES` before PyTorch and Lightning are
imported, so the process sees only that GPU (as logical `cuda:0`). For example:

```bash
python3 train_pud_detr.py \
  --device 1 \
  --precision 16-mixed \
  --experiment-name pud_drop03_gpu1 \
  --train-json /hdd1/junhyung/pud_detr/datasets/VOC2007/coco_annotations/pascal_train_drop_0.3.json \
  --val-json /hdd1/junhyung/pud_detr/datasets/VOC2007/coco_annotations/pascal_val.json \
  --test-json /hdd1/junhyung/pud_detr/datasets/VOC2007/coco_annotations/pascal_test.json \
  --trainval-image-dir /hdd1/junhyung/pud_detr/datasets/VOC2007/JPEGImages \
  --test-image-dir /hdd1/junhyung/pud_detr/datasets/VOC2007/JPEGImages
```

RTX 3090 supports FP16 Tensor Cores, so `16-mixed` is the recommended precision
for this server. Use `32-true` if mixed-precision stability needs to be ruled
out during debugging.

Strict deterministic algorithms are disabled by default because Deformable
DETR backpropagates through CUDA `grid_sample`, whose backward kernel is not
deterministic. Random seeds are still applied. Passing `--deterministic` enables
Lightning's best-effort `warn` mode: deterministic kernels are selected where
available, while unsupported operations warn and continue instead of raising a
`grid_sampler_2d_backward_cuda` runtime error.

Use `--seed 42` for a single run. To evaluate the same configuration with
exactly three independent seeds, replace it with `--seeds`, for example:

```bash
python3 train_pud_detr.py \
  --seeds 7 11 19 \
  --experiment-name pud_three_seeds \
  --train-json /path/to/pascal_train_drop_0.3.json \
  --val-json /path/to/pascal_val.json \
  --test-json /path/to/pascal_test.json \
  --trainval-image-dir /path/to/JPEGImages \
  --test-image-dir /path/to/JPEGImages
```

Each seed gets its own `seed_<seed>` run directory. Three-seed runs also write
`multi_seed_results.csv` under the experiment directory with per-seed best
validation AP plus its mean and population standard deviation.

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

## Convert VOC XML to PUD-DETR COCO JSON

`scripts/convert_voc_to_coco.py` converts the original VOC XML files and every
discovered `Annotations_drop_ratio_*` directory to the COCO representation used
by the reference PUD-DETR annotations.

The converter preserves the official VOC2007 detection splits from
`ImageSets/Main`: 2,501 train images, 2,510 validation images, and 4,952 test
images. Within each split, image order follows the split text file and image IDs
are the numeric JPEG stem (`000012.jpg` becomes `12`). Annotation IDs are
assigned sequentially from 1 in split and XML object order, which is the rule
used by the reference converter before its later custom train/validation
reshuffle. This keeps IDs unique for `pycocotools` and `train_pud_detr.py`.

VOC's 1-based inclusive `(xmin, ymin, xmax, ymax)` coordinates become COCO
zero-based `[x, y, width, height]`. The output also includes the reference
rectangle `segmentation`, `area`, `iscrowd`, and `ignore` fields; `ignore` is
copied from VOC's `difficult` flag. Category IDs are fixed to the reference
zero-based 20-class order (`aeroplane=0` through `tvmonitor=19`).

Run the complete conversion on the dataset server with:

```bash
python3 scripts/convert_voc_to_coco.py \
  --voc-root /hdd1/junhyung/pud_detr/datasets/VOC2007
```

The default destination is
`/hdd1/junhyung/pud_detr/datasets/VOC2007/coco_annotations`. It contains
`pascal_train.json`, `pascal_val.json`, `pascal_test.json`, and one
`pascal_train_drop_<ratio>.json` for each discovered bbox-drop XML directory.
Use `--output-dir` to choose another destination, repeat
`--drop-annotations-dir` to select directories explicitly, or add
`--skip-drop-annotations` to generate only the three complete split files.
Existing JSON files are protected unless `--overwrite` is supplied.

The generated files can be passed directly to the training entry point, for
example:

```bash
python3 train_pud_detr.py \
  --method pud \
  --experiment-name pud_drop03 \
  --train-json /hdd1/junhyung/pud_detr/datasets/VOC2007/coco_annotations/pascal_train_drop_0.3.json \
  --val-json /hdd1/junhyung/pud_detr/datasets/VOC2007/coco_annotations/pascal_val.json \
  --test-json /hdd1/junhyung/pud_detr/datasets/VOC2007/coco_annotations/pascal_test.json \
  --trainval-image-dir /hdd1/junhyung/pud_detr/datasets/VOC2007/JPEGImages \
  --test-image-dir /hdd1/junhyung/pud_detr/datasets/VOC2007/JPEGImages
```

## Validation ablation

`run_val_ablation.py` sweeps the Cartesian product of `weight_p` values and
non-negative correction reductions. Every combination trains PUD-DETR for 20
epochs, validates after every epoch, and records the best validation AP in one
CSV. It forces `--skip-test`, so test annotations and images are neither
required nor loaded.

Pass the sweep settings before `--` and the ordinary training arguments after
it:

```bash
python3 run_val_ablation.py \
  --weight-p-values 1 2 5 10 \
  --reductions global query_wise element_wise \
  --seeds 7 11 19 \
  --output-dir outputs/val_ablation \
  --results-csv outputs/val_ablation/results.csv \
  -- \
  --device 1 \
  --precision 16-mixed \
  --train-json /hdd1/junhyung/pud_detr/datasets/VOC2007/coco_annotations/pascal_train_drop_0.3.json \
  --val-json /hdd1/junhyung/pud_detr/datasets/VOC2007/coco_annotations/pascal_val.json \
  --trainval-image-dir /hdd1/junhyung/pud_detr/datasets/VOC2007/JPEGImages
```

The runner owns `--method`, `--weight-p`, `--reduction`, `--epochs`,
`--experiment-name`, `--output-dir`, `--skip-test`, `--seed`, and `--seeds`;
these options cannot be overridden after `--`. Omit `--seeds` (or pass one
`--seed`) to run the ablation in single-seed mode. In three-seed mode, the CSV
contains one row per seed and repeats the condition's validation AP mean and
population standard deviation on those rows.
