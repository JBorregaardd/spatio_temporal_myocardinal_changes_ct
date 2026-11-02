"""Tests for the label storage of the valve labelling tool in side_project/valve_planes."""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "side_project" / "valve_planes"))

from label_valves import find_patients, load_labels, save_labels, set_label  # noqa: E402


def test_labels_round_trip_through_csv(tmp_path: Path) -> None:
    labels = load_labels(str(tmp_path / "labels.csv"))
    assert labels == {}
    set_label(labels, "7", "aortic", "closed")
    set_label(labels, "7", "mitral", "open")
    set_label(labels, "12", "mitral", "unclear")
    save_labels(str(tmp_path / "labels.csv"), labels)

    loaded = load_labels(str(tmp_path / "labels.csv"))
    assert loaded["7"]["aortic"] == "closed"
    assert loaded["7"]["mitral"] == "open"
    assert loaded["12"] == {"aortic": "", "mitral": "unclear", "updated": labels["12"]["updated"]}


def test_set_label_rejects_unknown_values() -> None:
    with pytest.raises(ValueError):
        set_label({}, "7", "tricuspid", "open")
    with pytest.raises(ValueError):
        set_label({}, "7", "aortic", "half-open")


def test_find_patients_sorts_numeric_ids_numerically(tmp_path: Path) -> None:
    for patient_id in ("10", "2", "1"):
        (tmp_path / f"{patient_id}_valves.json").write_text("{}")
    assert find_patients(str(tmp_path)) == ["1", "2", "10"]
