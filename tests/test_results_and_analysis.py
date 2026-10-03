"""Consistency checks on the stored experiment outputs and the analysis helpers."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("compute_tables", ROOT / "analysis" / "compute_tables.py")
ct = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ct)
R1 = json.load(open(ROOT / "results" / "part1" / "results.json"))
R2 = json.load(open(ROOT / "results" / "part2" / "results.json"))
RUNS = [(k, v) for r in (R1, R2) for k, v in r.items() if isinstance(v, dict) and "status" in v]


def test_wilson_interval_matches_paper():
    lo, hi = ct.wilson(8146 - 49, 8146)
    assert round(lo, 2) == 99.21 and round(hi, 2) == 99.54


def test_difference_statistics():
    d, z, p, hw = ct.diff_stats(0.994, 0.9975, 8146)
    assert d > 0 and 3.0 < z < 4.0 and p < 0.001 and hw > 0


def test_all_runs_completed():
    assert len(RUNS) == 20
    assert all(v["status"] == "ok" and v["epochs_run"] == 30 for _, v in RUNS)


def test_same_fixed_split_in_both_parts():
    fixed = [v for k, v in RUNS if "cv_fold" not in k]
    assert {(v["n_train"], v["n_val"], v["n_test"]) for v in fixed} == {(38013, 8146, 8146)}


def test_cv_folds_use_full_dataset():
    for k, v in R1.items():
        if k.startswith("SSPathNet_cv_fold"):
            assert v["n_train"] + v["n_val"] + v["n_test"] == 54305


def test_headline_numbers_of_the_paper():
    assert abs(R1["SSPathNet_main"]["params_M"] - 9.797) < 0.001
    assert round(100 * R1["SSPathNet_main"]["test"]["accuracy"], 2) == 99.40
    assert round(100 * R2["A0_full_SSPathNet"]["test"]["accuracy"], 2) == 99.42
    assert round(100 * R1["baseline_efficientnet_b0"]["test"]["accuracy"], 2) == 99.75
    noise_dct = [R2[k]["robustness"]["gauss_noise"] for k in R2 if k.startswith("A") and k[:2] not in ("A1", "A2")]
    noise_no_dct = [R2[k]["robustness"]["gauss_noise"] for k in R2 if k[:2] in ("A1", "A2")]
    assert max(noise_dct) < 0.20 and min(noise_no_dct) > 0.85
