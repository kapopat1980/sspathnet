#!/usr/bin/env python3
"""
Regenerate every table of the paper from the stored experiment outputs.

    python analysis/compute_tables.py                      # defaults: results/ -> results/tables/

Statistics
  * 95 % Wilson score intervals for accuracies (Wilson 1927; Brown et al. 2001).
  * Differences between two models on the SAME test set: difference of accuracies with an
    UNPAIRED 95 % interval and z-statistic. (Per-image predictions were not stored, so a paired
    McNemar test is not possible; the unpaired interval is conservative for same-image comparisons.)
  * Multiplicity: Bonferroni two-sided threshold |z| > 2.77 for the 9 ablations (alpha = 0.05/9),
    |z| > 2.50 for the 4 baseline comparisons (alpha = 0.05/4).
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    a = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return 100 * (c - a) / d, 100 * (c + a) / d


def diff_stats(p_ref, p_other, n):
    """Return (difference in pp, z, two-sided p, half-width of unpaired 95% CI in pp)."""
    se = math.sqrt(p_ref * (1 - p_ref) / n + p_other * (1 - p_other) / n)
    d = p_other - p_ref
    z = d / se if se > 0 else 0.0
    return 100 * d, z, math.erfc(abs(z) / math.sqrt(2)), 196 * se


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", default="results/tables")
    a = ap.parse_args()
    rd, out = Path(a.results_dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    r1 = json.load(open(rd / "part1" / "results.json"))
    r2 = json.load(open(rd / "part2" / "results.json"))
    classes = json.load(open(rd / "part1" / "classes.json"))
    short = [c.replace("___", ": ").replace("_", " ") for c in classes]
    md = []

    # ---------------- Table 4: main comparison ----------------
    main_runs = [("SSPathNet", r1["SSPathNet_main"]), ("MobileNetV2", r1["baseline_mobilenet_v2"]),
                 ("ResNet-18", r1["baseline_resnet18"]), ("EfficientNet-B0", r1["baseline_efficientnet_b0"]),
                 ("DenseNet-121", r1["baseline_densenet121"])]
    n = r1["SSPathNet_main"]["n_test"]
    ref = r1["SSPathNet_main"]["test"]["accuracy"]
    rows = []
    for name, v in main_runs:
        err = round((1 - v["test"]["accuracy"]) * n)
        lo, hi = wilson(n - err, n)
        d, z, p, _ = diff_stats(ref, v["test"]["accuracy"], n) if name != "SSPathNet" else (0, 0, 1, 0)
        rows.append([name, f"{v['params_M']:.2f}", f"{100 * (1 - err / n):.2f}", f"{lo:.2f}", f"{hi:.2f}", err,
                     f"{100 * v['test']['macro_f1']:.2f}", f"{v['test']['cohen_kappa']:.4f}",
                     "" if name == "SSPathNet" else f"{d:+.2f}", "" if name == "SSPathNet" else f"{z:.2f}"])
    hdr = ["model", "params_M", "accuracy_pct", "ci95_lo", "ci95_hi", "errors", "macro_f1_pct", "kappa",
           "delta_acc_vs_SSPathNet_pp", "z_unpaired"]
    write_csv(out / "table4_main_comparison.csv", hdr, rows)
    md += [f"### Table 4 - main comparison (n_test = {n}; Bonferroni |z| > 2.50 for the 4 comparisons)", "",
           "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)] + ["| " + " | ".join(map(str, r)) + " |" for r in rows] + [""]

    # ---------------- Table 5: cross-validation ----------------
    cv = [(k, v) for k, v in r1.items() if k.startswith("SSPathNet_cv_fold")]
    rows = [[k[-5:], v["n_test"], f"{100 * v['test']['accuracy']:.2f}", f"{100 * v['test']['macro_f1']:.2f}",
             f"{v['test']['cohen_kappa']:.4f}"] for k, v in cv]
    A = np.array([100 * v["test"]["accuracy"] for _, v in cv]); F = np.array([100 * v["test"]["macro_f1"] for _, v in cv])
    K = np.array([v["test"]["cohen_kappa"] for _, v in cv])
    rows.append(["mean+-SD", "", f"{A.mean():.2f} +- {A.std(ddof=1):.2f}", f"{F.mean():.2f} +- {F.std(ddof=1):.2f}",
                 f"{K.mean():.4f} +- {K.std(ddof=1):.4f}"])
    hdr = ["fold", "n_test", "accuracy_pct", "macro_f1_pct", "kappa"]
    write_csv(out / "table5_cross_validation.csv", hdr, rows)
    md += ["### Table 5 - five-fold cross-validation of SSPathNet", "", "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)] + \
          ["| " + " | ".join(map(str, r)) + " |" for r in rows] + [""]

    # ---------------- Table 6: efficiency ----------------
    eff = [("SSPathNet (Part 1)", r1["SSPathNet_main"]), ("MobileNetV2", r1["baseline_mobilenet_v2"]),
           ("ResNet-18", r1["baseline_resnet18"]), ("EfficientNet-B0", r1["baseline_efficientnet_b0"]),
           ("DenseNet-121", r1["baseline_densenet121"]), ("A0 full (Part 2)", r2["A0_full_SSPathNet"]),
           ("A4 no deformable", r2["A4_no_deformable"]), ("A5 no attention", r2["A5_no_attention"])]
    rows = [[nm, f"{v['params_M']:.2f}", f"{v['gflops_fwd']:.2f}", f"{v['throughput_img_s']:.0f}",
             f"{v['latency_ms_bs1']:.1f}", f"{v['train_min']:.1f}"] for nm, v in eff]
    hdr = ["model", "params_M", "fwd_gflops", "throughput_img_s_bs128", "latency_ms_bs1", "train_min_30ep"]
    write_csv(out / "table6_efficiency.csv", hdr, rows)
    md += ["### Table 6 - efficiency (T4, fp16, 160x160; GFLOPs exclude grid sampling)", "", "| " + " | ".join(hdr) + " |",
           "|" + "---|" * len(hdr)] + ["| " + " | ".join(map(str, r)) + " |" for r in rows] + [""]

    # ---------------- Table 7: ablations ----------------
    abl = ["A0_full_SSPathNet", "A1_stem_rgb_only", "A2_stem_rgb_hsv", "A3_stem_rgb_dct", "A4_no_deformable",
           "A5_no_attention", "A6_agg_none", "A7_agg_concat", "A8_no_center_loss", "A9_gelu_head"]
    n2 = r2["A0_full_SSPathNet"]["n_test"]
    p0 = r2["A0_full_SSPathNet"]["test"]["accuracy"]
    rows = []
    for k in abl:
        v = r2[k]
        err = round((1 - v["test"]["accuracy"]) * n2)
        if k == abl[0]:
            d = z = p = None
        else:
            d, z, p, _ = diff_stats(p0, v["test"]["accuracy"], n2)
        rows.append([k.split("_")[0], k, f"{v['params_M']:.2f}", f"{100 * (1 - err / n2):.2f}", err,
                     f"{100 * v['test']['macro_f1']:.2f}", "" if d is None else f"{d:+.2f}", "" if z is None else f"{z:.2f}",
                     "" if p is None else f"{p:.4f}", "" if z is None else ("yes" if abs(z) > 2.77 else "no"),
                     f"{100 * v['robustness']['gauss_noise']:.1f}"])
    hdr = ["id", "run", "params_M", "accuracy_pct", "errors", "macro_f1_pct", "delta_acc_pp", "z", "p_unadjusted",
           "significant_after_bonferroni_z_gt_2.77", "noise_acc_pct_sigma0.08"]
    write_csv(out / "table7_ablation.csv", hdr, rows)
    md += ["### Table 7 - ablation study (delta vs A0)", "", "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)] + \
          ["| " + " | ".join(map(str, r)) + " |" for r in rows] + [""]

    # exploratory (post hoc) stem contrasts
    pa = lambda k: r2[k]["test"]["accuracy"]
    contrasts = [("A0->A2 (remove DCT from full)", "A0_full_SSPathNet", "A2_stem_rgb_hsv"),
                 ("A0->A1 (remove DCT and HSV)", "A0_full_SSPathNet", "A1_stem_rgb_only"),
                 ("A1->A3 (add DCT to RGB)", "A1_stem_rgb_only", "A3_stem_rgb_dct"),
                 ("A1->A2 (add HSV to RGB)", "A1_stem_rgb_only", "A2_stem_rgb_hsv"),
                 ("A3->A0 (add HSV to RGB+DCT)", "A3_stem_rgb_dct", "A0_full_SSPathNet"),
                 ("A2->A0 (add DCT to RGB+HSV)", "A2_stem_rgb_hsv", "A0_full_SSPathNet")]
    rows = []
    for lab, x, y in contrasts:
        d, z, p, hw = diff_stats(pa(x), pa(y), n2)
        rows.append([lab, f"{d:+.2f}", f"{z:.2f}", f"{p:.4f}", f"+-{hw:.2f}"])
    hdr = ["contrast (exploratory, post hoc)", "delta_acc_pp", "z", "p_unadjusted", "ci95_halfwidth_pp"]
    write_csv(out / "table7b_stem_contrasts_exploratory.csv", hdr, rows)
    md += ["### Table 7b - exploratory stem contrasts (identified after inspecting results)", "", "| " + " | ".join(hdr) + " |",
           "|" + "---|" * len(hdr)] + ["| " + " | ".join(map(str, r)) + " |" for r in rows] + [""]

    # ---------------- robustness ----------------
    pert = ["clean", "dark", "bright", "low_contrast", "gauss_noise", "blur", "hue_shift", "occlusion"]
    allruns = [("SSPathNet_main", r1["SSPathNet_main"]), ("MobileNetV2", r1["baseline_mobilenet_v2"]),
               ("ResNet-18", r1["baseline_resnet18"]), ("EfficientNet-B0", r1["baseline_efficientnet_b0"]),
               ("DenseNet-121", r1["baseline_densenet121"])] + [(k, r2[k]) for k in abl]
    rows = [[nm] + [f"{100 * v['robustness'][c]:.2f}" for c in pert] for nm, v in allruns]
    write_csv(out / "robustness.csv", ["model"] + pert, rows)
    md += ["### Robustness - test accuracy (%) under synthetic shift", "", "| model | " + " | ".join(pert) + " |",
           "|" + "---|" * (len(pert) + 1)] + ["| " + " | ".join(r) + " |" for r in rows] + [""]
    dct = [r2[k]["robustness"]["gauss_noise"] * 100 for k in abl if k not in ("A1_stem_rgb_only", "A2_stem_rgb_hsv")]
    dct.append(r1["SSPathNet_main"]["robustness"]["gauss_noise"] * 100)
    md += [f"Noise accuracy of the {len(dct)} DCT-containing models: min {min(dct):.2f}, max {max(dct):.2f}, mean {np.mean(dct):.2f}; "
           f"without DCT (A1, A2): {100 * r2['A1_stem_rgb_only']['robustness']['gauss_noise']:.2f}, "
           f"{100 * r2['A2_stem_rgb_hsv']['robustness']['gauss_noise']:.2f}.", ""]

    # ---------------- confusions of the main run ----------------
    cm = np.load(rd / "part1" / "SSPathNet_main" / "confusion_matrix.npy")
    off = cm.copy()
    np.fill_diagonal(off, 0)
    rows = []
    for i, j in zip(*np.unravel_index(np.argsort(-off, axis=None)[:10], off.shape)):
        rows.append([int(off[i, j]), short[i], short[j]])
    write_csv(out / "table8_top_confusions.csv", ["count", "true_class", "predicted_class"], rows)
    md += ["### Table 8 - top confusions (SSPathNet main run)", "", "| count | true | predicted |", "|---|---|---|"] + \
          [f"| {r[0]} | {r[1]} | {r[2]} |" for r in rows] + [""]
    rep = json.load(open(rd / "part1" / "SSPathNet_main" / "classification_report.json"))
    weak = sorted([(rep[c]["f1-score"], int(rep[c]["support"]), s) for c, s in zip(classes, short)])[:5]
    md += ["Weakest classes (F1, test support): " + "; ".join(f"{s} ({f:.3f}, n={sup})" for f, sup, s in weak), ""]

    (out / "tables.md").write_text("\n".join(md))
    print(f"wrote {out / 'tables.md'}")


if __name__ == "__main__":
    main()
