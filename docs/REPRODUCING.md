# Reproducing the experiments

The study was run in two independent Kaggle sessions (one NVIDIA Tesla T4 each). The two scripts do **not** depend on
each other; each re-creates its own data cache and writes its own results folder.

| Script | Runs | Observed wall time (T4) |
|---|---|---|
| `scripts/plantvillage_part1.py` | SSPathNet main run (+Grad-CAM, robustness), 4 from-scratch baselines, 5-fold CV of SSPathNet | **8.92 h** |
| `scripts/plantvillage_part2.py` | Full-model reference A0 + nine one-factor ablations A1-A9 (+robustness) | **10.50 h** |

Both include about 10 minutes of one-off image decoding. The scripts keep an internal time budget of 11 h
(`TIME_BUDGET_H`) and skip runs that cannot finish within it; in our sessions **no run was skipped or truncated**
(all 20 runs have `status: ok`, 30 epochs).

## 1. On Kaggle (recommended; this is what was used)

1. Create a notebook, **Accelerator = GPU T4**, internet not required.
2. **Add data**: the Kaggle dataset `abdallahalidev/plantvillage-dataset` (we used its `color` folder: 54,305 images,
   38 classes). Do not use the *augmented* variants - they contain pre-made copies of images that leak across splits.
3. Paste the whole content of `scripts/plantvillage_part1.py` into a cell and run it (or upload it and
   `!python plantvillage_part1.py`). Start Part 2 in a **fresh session**, because the 11 h budget is measured from
   script start and the Kaggle session limit is 12 h.
4. Results appear in `/kaggle/working/part1_results/` and `/kaggle/working/part2_results/`
   (the copies used for the paper are in `results/part1` and `results/part2` of this repository).

The dataset folder is auto-detected under `/kaggle/input` (the `color` folder is preferred). To force a path, edit
`DATASET_PATH` at the top of the script, e.g.
`DATASET_PATH = "/kaggle/input/datasets/abdallahalidev/plantvillage-dataset/color"`.

### Smoke test (about 5 minutes)

Set `SMOKE_TEST = True` at the top of a script: it uses 30 images per class, 1 epoch, 2 CV folds / 2 ablations and a
30-minute budget. Use it to verify the environment, then set it back to `False`.

### Adjusting to a slower GPU session

Runtime scales with `number of runs x EPOCHS x seconds per epoch`. Each run prints its estimated seconds/epoch before
starting. If Part 2 projects beyond ~11 h, lower `EPOCHS` (note: this changes the results and breaks comparability with
the paper), or accept that the last-priority ablations (A3, A7) are skipped by the time guard.

## 2. Locally

The scripts are written for Kaggle but are plain Python. Edit the constants at the top of a script:

```python
DATASET_PATH = "/path/to/plantvillage-dataset/color"   # folder whose sub-folders are the 38 classes
OUT_ROOT     = "./outputs"                              # default: /kaggle/working
```

A CUDA GPU with ~16 GB is expected (batch size 128 at 160 x 160; the script halves the batch size automatically on
out-of-memory, which would change the optimisation and therefore the numbers). CPU-only execution is only suitable for
`SMOKE_TEST`.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/plantvillage_part1.py
python scripts/plantvillage_part2.py
```

## 3. Regenerating the paper's tables and figures from stored results

```bash
python analysis/compute_tables.py      # -> results/tables/*.csv and tables.md   (Tables 4-8, robustness)
python analysis/make_figures.py        # -> figures/fig1..fig6  (fig6 = copy of the Grad-CAM figure)
```

| Paper item | File |
|---|---|
| Table 4 main comparison | `results/tables/table4_main_comparison.csv` |
| Table 5 cross-validation | `results/tables/table5_cross_validation.csv` |
| Table 6 efficiency | `results/tables/table6_efficiency.csv` |
| Table 7 ablation (+exploratory contrasts) | `results/tables/table7_ablation.csv`, `table7b_stem_contrasts_exploratory.csv` |
| Table 8 confusions | `results/tables/table8_top_confusions.csv` |
| Figure 5 robustness (numbers) | `results/tables/robustness.csv` |
| Figure 1 architecture | `figures/fig1_architecture.png` |
| Figure 2 training curves | `figures/fig2_curves.png` |
| Figure 3 efficiency | `figures/fig3_efficiency.png` |
| Figure 4 ablation | `figures/fig4_ablation.png` |
| Figure 5 robustness heat-map | `figures/fig5_robustness.png` |
| Figure 6 Grad-CAM | `figures/fig6_gradcam.png` |

## 4. Layout of a run folder

```
results/part1/<run>/
    history.csv                 per-epoch train loss/acc, val acc, lr, seconds
    classification_report.json  per-class precision/recall/F1/support on the test split
    confusion_matrix.npy|.png   38 x 38, rows = true class
    curves.png
    gradcam.png                 SSPathNet_main only
results/part1/results.json      all metrics, efficiency numbers and robustness accuracies
results/part1/summary.csv       one line per run
results/part1/log.txt           full console log of the Kaggle session
```

`best.pt` weights are not versioned (about 40 MB per run, `.gitignore`d). If you want to publish them, attach them to a
GitHub Release.

## 5. Notes on determinism

* Splits are deterministic (stratified, `seed = 42`; the split code is identical in both scripts and tested).
* Training uses `cudnn.benchmark = True` and fp16 autocast, so re-running on a GPU will not be bit-identical.
  Expect small run-to-run differences: in our data, the same configuration trained twice (main run vs. A0) differed by
  2 test images (99.40 % vs. 99.42 %). Treat differences below roughly 0.2 percentage points as noise.
* Per-image predictions were not stored, so paired tests (McNemar) cannot be computed from this repository.
  If you re-run, consider saving predictions.

## 6. Unit tests

```bash
pip install -r requirements-dev.txt
pytest -q tests          # 30 tests, ~1 minute on CPU, no dataset or Kaggle needed
```

The tests check the parameter budget (9.797 M), shapes, gradient flow through all ten model variants, the DCT/HSV
transforms, the deformable convolution (zero-initialised = ordinary depthwise conv), center loss, split determinism,
and that the stored results are internally consistent with the numbers reported in the paper.
