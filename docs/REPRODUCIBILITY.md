# Reproducibility and implementation notes

## Preserved implementation

The paper code is preserved from commit `a6278fa9cd5fbddef3d807edcd80a2a56313b14d` on the former `codex/nnpu-val-ablation` branch. All Python entry points, data utilities, tests, and `requirements.txt` are unchanged. Earlier main and development commits remain in Git history.

Documentation follows the author-supplied final Overleaf manuscript and the actual code. All five repository figures and all thirteen annotation JSONs are preserved byte-for-byte. Manuscript templates, editorial work files, raw images, caches, and generated training artifacts are not required in the code release.

## Paper settings and code behavior

| Setting | Paper / released behavior | CLI |
| --- | --- | --- |
| Training duration | 20 epochs | `--epochs 20` |
| Batch size | 2; historical default is 4 | `--batch-size 2` |
| Transformer / backbone LR | 1e-4 / 1e-5 | `--lr 1e-4 --lr-backbone 1e-5` |
| Weight decay / clipping | 1e-4 / 0.1 | `--weight-decay 1e-4 --gradient-clip 0.1` |
| Warm-up | 2,000 steps, then cosine | `--warmup-steps 2000` |
| Focal parameters | Alpha 0.25, gamma 2 | `--focal-alpha 0.25 --focal-gamma 2` |
| Auxiliary decoder loss | Disabled | Fixed in entry point |
| VOC drop=0.7 positive-risk scale | 8 | `--weight-p 8` |
| Diatom positive-risk scale | 7 | `--weight-p 7` |
| Main non-negative correction | Image-wise global | `--reduction global` |
| Checkpoint selection | Maximum validation AP@[0.50:0.95] | Fixed in entry point |
| Test evaluation | Best-validation-AP checkpoint | Supply test paths; omit `--skip-test` |

The paper selects the positive-risk scale using validation AP for VOC and validation AR1 for diatoms. The existing ablation runner summarizes validation AP only; its selection CSV does not implement diatom AR1-based selection. The training entry point logs validation AR1 every epoch. The supplied diatom command uses the paper-selected scale directly, without changing the code's checkpoint-selection behavior.

## Annotation identity

`annotation_manifest.json` records byte sizes, SHA-256 hashes, image/annotation counts, category IDs, and configured/actual VOC training drop ratios. All JSONs are committed as supplied. The VOC drop masks retain the complete training image records and category mapping, and their boxes are subsets of the complete training annotation multiset. Within each dataset, train/validation/test filenames are disjoint. Every supplied training image retains at least one box.

The VOC `0.7` condition retains exactly one annotation per image, so its actual removal fraction is `5343 / 7844 = 0.6811575726670066`. This is the maximum compatible with keeping one annotation per training image. Filenames and configured experimental labels are preserved. Lower settings differ slightly from nominal ratios through integer rounding. The committed masks define this release; optional regeneration is for new variants.

| Diatom split | Images | Boxes |
| --- | ---: | ---: |
| Train | 1,520 | 2,118 |
| Validation | 327 | 456 |
| Test | 337 | 453 |
| Total | 2,184 | 3,027 |

Diatom category IDs are contiguous 0–67. JSON `file_name` entries such as `2069.png` resolve relative to the supplied image directory. Raw images must be obtained separately.

Verify hashes from the repository root with Python (standard library only):

```bash
python - <<'PY'
import hashlib
import json
from pathlib import Path

manifest = json.loads(Path("docs/annotation_manifest.json").read_text())
for entry in manifest["files"]:
    path = Path(entry["path"])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], path
print("All released annotation hashes match.")
PY
```

## Classification-loss semantics

- The PU unlabeled mask covers every zero entry in the one-hot class target, including non-target classes at matched queries and all classes at unmatched queries.
- `weight_p` scales both the positive risk and positive-as-negative subtraction. It is distinct from focal-loss alpha.
- The positive-as-negative term uses `probability**gamma` focal modulation. The unlabeled coefficient is total query count divided by unmatched query count, with a denominator floor of one.
- `global` averages over queries, sums over classes, and clamps once per image. `query_wise` averages over queries and clamps separately for each image/class. `element_wise` clamps individual query/class contributions before aggregation.
- The loss is normalized by the annotated box count and multiplied by query count in the criterion. Hungarian matching and box regression are preserved.
- The weighted-PN ablation scales every zero one-hot target position. Although the manuscript describes this control in terms of unmatched-query losses, the code also scales non-target classes at matched queries. The release preserves the experiment implementation.

## Outputs and numerical reproducibility

Each run saves arguments, resolved paths, dataset summary, core package versions, checkpoint selection, and metrics in `config.json`. `metrics.csv` contains per-run validation AP and available test metrics. Three-seed execution adds `multi_seed_results.csv` for validation AP only. Test results must be aggregated from the individual run files.

The built-in summary uses population standard deviation (`numpy.std`, `ddof=0`). Code metrics are on a 0–1 scale, while the manuscript tables use 0–100. Record the standard-deviation convention when producing new tables.

Seed setting covers Python, NumPy, PyTorch, and data-loader workers. CUDA `grid_sample` backward can remain nondeterministic. Match precision, hardware, package versions, model revision, annotations, and command-line options when comparing runs. `--deterministic` requests best-effort warning mode rather than bitwise guarantees. The CLI supports a pinned pretrained revision through `--hf-revision`.

## Release validation scope

Release checks cover code/dependency identity against the reference commit, annotation integrity and split separation, image-file integrity, README paths/options, and Git ignore rules. Existing unit tests are run without retraining. Loss tests require PyTorch and the relevant training packages; skipped modules are not passing GPU-loss validation.

The local raw VOC image/XML/split directories contain placeholder files and are excluded from Git. Download actual data before training or regenerating annotations. Counts and disjointness were validated from the released JSONs; equality with official split text files could not be independently checked against the empty local placeholders. Full GPU training and paper-table reproduction are outside this repository cleanup.
