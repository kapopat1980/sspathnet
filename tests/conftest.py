"""Loads the (standalone, Kaggle-oriented) scripts as modules without touching /kaggle."""
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(script, out_dir):
    path = ROOT / "scripts" / script
    src = path.read_text()
    needle = 'OUT_ROOT       = "/kaggle/working"'
    assert needle in src, "OUT_ROOT line changed; update tests/conftest.py"
    src = src.replace(needle, f"OUT_ROOT       = {str(out_dir)!r}")
    mod = types.ModuleType(script[:-3])
    mod.__file__ = str(path)
    exec(compile(src, str(path), "exec"), mod.__dict__)   # defines everything; does not run main()
    return mod


@pytest.fixture(scope="session")
def p1(tmp_path_factory):
    return _load("plantvillage_part1.py", tmp_path_factory.mktemp("out_part1"))


@pytest.fixture(scope="session")
def p2(tmp_path_factory):
    return _load("plantvillage_part2.py", tmp_path_factory.mktemp("out_part2"))
