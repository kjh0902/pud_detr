# PUD-DETR

**Positive-Unlabeled Deformable DETR for Object Detection under Incomplete Annotations**

PUD-DETR integrates non-negative positive-unlabeled (nnPU) classification risk into Deformable DETR. With incomplete annotations, a real object may have no bounding-box label and receive incorrect background supervision. PUD-DETR treats query–class entries without observed positive assignments as unlabeled and reconstructs their negative risk, while retaining Hungarian matching and bounding-box regression.

`main` contains the paper implementation preserved from `codex/nnpu-val-ablation`, commit [`a6278fa`](https://github.com/kjh0902/pud_detr/commit/a6278fa9cd5fbddef3d807edcd80a2a56313b14d). The release cleanup changes documentation and tracked assets, not the experimental Python code or dependency pins.

![Examples of incomplete diatom annotations](figures/intro_diatom.png)

Available annotations (left) and additional visible diatom instances (right), illustrating missing labels.

## Paper results

The paper reports three-seed experiments on PASCAL VOC 2007 and an inherently incompletely annotated diatom dataset. PN has higher VOC AP at configured drop ratios 0.0–0.3; PUD-DETR has higher AP at 0.4–0.7. At configured drop=0.7, AP is **41.07 ± 0.72** for PUD-DETR and **37.51 ± 1.21** for PN. On diatoms, AR1 is **84.33 ± 1.95** versus **82.53 ± 0.87**, using the paper's selected positive-risk scale of 7. These values are on a 0–100 scale.

![VOC AP and AR1 across configured annotation drop ratios](figures/experiment_result.png)

The supplied paper figure uses a 0–1 metric scale. The benefit depends on annotation omission and the evaluation metric.

## Repository structure

```text
.
├── train_pud_detr.py                    # PN baseline and PUD-DETR
├── train_pud_detr_negative_ablation.py  # Pairwise negative-loss weighting control
├── run_val_ablation.py                 # Validation-only weight_p / clamp sweeps
├── requirements.txt                    # Preserved experimental dependency pins
├── scripts/
│   ├── drop_voc_instances.py            # Constrained training-box removal
│   ├── convert_voc_to_coco.py           # VOC XML → COCO JSON
│   └── gpu_selection.py                # Device selection / determinism helpers
├── tests/                              # Data, loss, device, and runner checks
├── datasets/
│   ├── VOC2007/coco_annotations/        # Complete splits + drop=0.1 through 0.7
│   └── diatom/                         # Experiment train/val/test COCO annotations
├── figures/                            # Five supplied paper figures
└── docs/
    ├── REPRODUCIBILITY.md
    └── annotation_manifest.json         # Annotation counts and SHA-256 hashes
```

Raw images, generated XML variants, checkpoints, logs, caches, and temporary files are excluded from Git. Released annotations are ordinary Git files; Git LFS is not required.

## Environment / dependencies

The preserved requirements use CUDA 12.6 PyTorch wheels. Use a Linux environment with a compatible NVIDIA GPU/driver for training; GPU selection was developed for an RTX 3090 setup. The original Python interpreter version is not recorded. Python 3.12 is a setup starting point, not a claim about the original interpreter.

```bash
git clone https://github.com/kjh0902/pud_detr.git
cd pud_detr
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Key pins are PyTorch `2.13.0+cu126`, torchvision `0.28.0+cu126`, PyTorch Lightning `2.6.5`, Transformers `4.44.2`, timm `1.0.28`, and pycocotools `2.0.11`. See [requirements.txt](requirements.txt) for the complete snapshot. These pins are preserved from the experiment branch; the release cleanup did not reinstall or revalidate the complete GPU environment.

The default pretrained model is `SenseTime/deformable-detr` on Hugging Face. Initial loading requires access to its files. Use `--hf-revision <commit>` to pin a model revision and `--local-files-only` once the required files are cached. The original pretrained revision is not supplied.

Examples use Bash syntax. `--device 0` selects physical GPU 0 before importing PyTorch. Precision defaults to `32-true`; `--precision 16-mixed` is supported but should match the run being reproduced.

## Dataset preparation

### PASCAL VOC 2007

Obtain the VOC2007 train/validation and test data from the [PASCAL VOC 2007 project](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/), following its access and use conditions. Place the real JPEG files from both archives in this layout:

```text
datasets/VOC2007/
├── JPEGImages/          # Download train, validation, and test images separately
├── Annotations/         # Original XMLs; only needed for optional regeneration
├── ImageSets/Main/      # Original split files; only needed for optional regeneration
└── coco_annotations/    # Included in this repository
```

Training consumes the committed JSON files and JPEGs directly. Separate test images can instead be supplied with `--test-image-dir`. The released splits contain **2,501 train / 2,510 validation / 4,952 test** images, with no shared filenames between splits. Category IDs are contiguous **0–19**. Boxes use COCO zero-based `[x, y, width, height]`; `file_name` identifies an image relative to its image directory.

### Drop-ratio annotations

Use the committed files directly to preserve the supplied experimental masks. Validation and test use the same complete annotation files for every training drop setting.

| Configured drop | Training JSON | Retained boxes | Actual removed fraction |
| --- | --- | ---: | ---: |
| 0.0 | `pascal_train.json` | 7,844 | 0.000000 |
| 0.1 | `pascal_train_drop_0.1.json` | 7,060 | 0.099949 |
| 0.2 | `pascal_train_drop_0.2.json` | 6,275 | 0.200025 |
| 0.3 | `pascal_train_drop_0.3.json` | 5,491 | 0.299975 |
| 0.4 | `pascal_train_drop_0.4.json` | 4,706 | 0.400051 |
| 0.5 | `pascal_train_drop_0.5.json` | 3,922 | 0.500000 |
| 0.6 | `pascal_train_drop_0.6.json` | 3,138 | 0.599949 |
| 0.7 | `pascal_train_drop_0.7.json` | 2,501 | **0.681158** |

Each incomplete training file retains the same 2,501 images and at least one box per image. Configured drop=0.7 reaches the maximum feasible removal of 5,343 / 7,844 boxes (about 68.12%). The filename and paper's configured label remain 0.7. Validation has 7,818 boxes and test has 14,976 boxes. Counts and hashes are recorded in [the annotation manifest](docs/annotation_manifest.json).

![Illustration of training annotation removal](figures/experiment_pascal_drop.png)

Figure labels denote configured settings; the table records actual aggregate removal in the released files.

For **new annotation variants**, the existing utilities can drop training boxes and convert VOC XMLs:

```bash
python scripts/drop_voc_instances.py --voc-root datasets/VOC2007 --drop-ratio 0.3 --seed 42
python scripts/convert_voc_to_coco.py --voc-root datasets/VOC2007 --output-dir outputs/voc_regenerated
```

Dropping prioritizes keeping one object per training image, meeting the feasible global drop count, and balancing class-specific drop rates, in that order. Validation/test XMLs remain unchanged. This example seed is not verified as the generation seed of the released JSONs. Regeneration is optional and does not establish identical masks; do not overwrite the released files for paper reproduction.

### Diatom dataset

The manuscript uses the dataset described by Gündüz, Solak, and Günal, *Segmentation of diatoms using edge detection and deep learning* (2022), [DOI: 10.55730/1300-0632.3938](https://doi.org/10.55730/1300-0632.3938). After excluding 13 unannotated images, the paper reports 2,184 microscopy images across 68 species, split into **1,520 train / 327 validation / 337 test** images. No synthetic drop is applied.

The experiment-specific COCO annotations and split membership are included as `datasets/diatom/diatom_train.json`, `diatom_val.json`, and `diatom_test.json`. They contain 2,118 / 456 / 453 boxes respectively (3,027 total), with category IDs 0–67. Obtain the raw microscopy images through the dataset authors' distribution and place them under a shared image root matching the JSON `file_name` entries:

```text
datasets/diatom/
├── images/             # Download separately; preserve JSON-relative filenames
├── diatom_train.json
├── diatom_val.json
└── diatom_test.json
```

Use the committed split membership and category mapping rather than creating a new random split. See the manifest for the actual annotation counts and IDs. The paper selects **`--weight-p 7` using validation AR1** and reports AR1/AR10/AR100 against the incomplete references; these metrics do not measure recall over every visible diatom. Checkpoint selection in the preserved code still uses validation AP. The distinction and tuning limitations are explained in [reproducibility notes](docs/REPRODUCIBILITY.md).

## Training

Run from the repository root. Examples explicitly set the paper's **batch size 2** (the code default is 4). Seeds 0, 1, 2 illustrate how to request three independent runs.

### PN baseline

```bash
python train_pud_detr.py \
  --method pn --experiment-name voc_drop07_pn \
  --seeds 0 1 2 --device 0 --batch-size 2 --epochs 20 \
  --drop-ratio 0.7 \
  --train-json datasets/VOC2007/coco_annotations/pascal_train_drop_0.7.json \
  --val-json datasets/VOC2007/coco_annotations/pascal_val.json \
  --test-json datasets/VOC2007/coco_annotations/pascal_test.json \
  --trainval-image-dir datasets/VOC2007/JPEGImages \
  --test-image-dir datasets/VOC2007/JPEGImages
```

### PUD-DETR

```bash
python train_pud_detr.py \
  --method pud --weight-p 8 --reduction global \
  --experiment-name voc_drop07_pud \
  --seeds 0 1 2 --device 0 --batch-size 2 --epochs 20 \
  --drop-ratio 0.7 \
  --train-json datasets/VOC2007/coco_annotations/pascal_train_drop_0.7.json \
  --val-json datasets/VOC2007/coco_annotations/pascal_val.json \
  --test-json datasets/VOC2007/coco_annotations/pascal_test.json \
  --trainval-image-dir datasets/VOC2007/JPEGImages \
  --test-image-dir datasets/VOC2007/JPEGImages
```

`--weight-p` is the paper's positive-risk scaling factor, distinct from `--focal-alpha`. For other VOC drops, change the training JSON and metadata label and use the validation-selected scale. The paper tunes this scale from 1–10; the value 8 below is specific to drop=0.7. `--drop-ratio` records metadata; it does not remove annotations at runtime. Use `pascal_train.json` for drop=0.0.

Diatom training with the paper-selected scale:

```bash
python train_pud_detr.py \
  --method pud --weight-p 7 --reduction global \
  --experiment-name diatom_pud \
  --seeds 0 1 2 --device 0 --batch-size 2 --epochs 20 \
  --train-json datasets/diatom/diatom_train.json \
  --val-json datasets/diatom/diatom_val.json \
  --test-json datasets/diatom/diatom_test.json \
  --trainval-image-dir datasets/diatom/images \
  --test-image-dir datasets/diatom/images
```

For the diatom PN baseline, use `--method pn`, omit `--weight-p`/`--reduction`, and choose a different experiment name. For one run, replace `--seeds 0 1 2` with `--seed <integer>`.

Training uses AdamW, transformer/backbone learning rates `1e-4`/`1e-5`, weight decay `1e-4`, gradient clipping `0.1`, 2,000 warm-up steps, cosine annealing, and no auxiliary decoder loss. Each epoch is evaluated on validation AP@[0.50:0.95]. The best-validation-AP checkpoint is evaluated on the test set. `--skip-test` enables validation-only runs without test paths.

Outputs appear under `outputs/<experiment-name>/seed_<seed>/` (a suffix prevents overwriting): `config.json`, `metrics.csv`, `checkpoints/`, and `lightning_logs/`. Three-seed runs add `multi_seed_results.csv` with **validation AP** mean and population standard deviation; test metrics remain in individual run files.

## Ablations

Validation-only VOC positive-risk-scale selection:

```bash
python run_val_ablation.py \
  --weight-p-values 1 2 3 4 5 6 7 8 9 10 \
  --reductions global --seeds 0 1 2 \
  --output-dir outputs/voc_drop07_selection \
  --results-csv outputs/voc_drop07_selection/results.csv \
  -- \
  --device 0 --batch-size 2 \
  --train-json datasets/VOC2007/coco_annotations/pascal_train_drop_0.7.json \
  --val-json datasets/VOC2007/coco_annotations/pascal_val.json \
  --trainval-image-dir datasets/VOC2007/JPEGImages
```

For the clamp ablation, use `--weight-p-values 8 --reductions global query_wise element_wise`. The runner forces 20 epochs and `--skip-test`; its CSV is for validation AP selection, not the paper's test tables or diatom AR1 selection. Evaluate test metrics with the main training entry point and test paths after choosing the setting.

For weighted PN, use `train_pud_detr_negative_ablation.py` with the PN command above and add `--negative-weight 0.25` (also 0, 0.5, 0.75, or 1). Use a distinct experiment name per condition. In the preserved code, the multiplier applies to **every zero one-hot target position**, including non-target classes at matched queries. `1` recovers PN. The supplied manuscript reports AP **3.47 ± 0.13** for weight 0.

## Reproducibility

- Keep the committed annotation masks fixed across methods and training seeds; verify their SHA-256 values in the manifest.
- Set `--batch-size 2` explicitly. The historical defaults of batch size 4 and `--weight-p 5` differ from the selected paper settings.
- Save the Git commit, command, model revision, package versions, precision, hardware, per-seed configs, and metrics. `config.json` records arguments, dataset counts, selected checkpoint, and core package versions.
- CUDA `grid_sample` backward is not strictly deterministic. `--deterministic` requests best-effort execution with warnings; seeds do not guarantee bitwise-identical GPU runs.
- Code metrics use a 0–1 scale; multiply by 100 for the paper tables. Built-in aggregation uses population standard deviation (`ddof=0`) for validation AP only. The supplied manuscript does not specify the test-table standard-deviation convention.
- The release preserves the experimental all-zero-one-hot PU mask, image-wise global clamp, and weighted-PN behavior. See [implementation details](docs/REPRODUCIBILITY.md) before claiming exact table reproduction.

Run existing checks in the training environment:

```bash
python -m unittest discover -s tests -v
```

Loss tests require the training dependencies and may be reported as skipped when these are absent.

## Citation

Paper title: *PUD-DETR: Positive-Unlabeled Deformable DETR for Object Detection under Incomplete Annotations*.

Authors: Junhyung Kim, Jiseok Son, Jungjae Park, Jiwoo Han, and Tae Hyung Kim.

Publication details, a paper link, and the final BibTeX entry will be added here when available.

<!-- Add the verified venue, year, DOI/arXiv identifier, and final BibTeX here. -->
