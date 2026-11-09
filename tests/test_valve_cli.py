"""Tests for the GloViTa training CLI in side_project/valve_planes/valve.py (no GloViTa or GPU needed)."""
import csv
import json
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "side_project" / "valve_planes"))

import valve  # noqa: E402
from train_glovita import split_mirror_axes  # noqa: E402

RUN_NAME = "20261104_124730_nnfoundationencoder_classificationhead_full_finetuning_04f53122"


def make_run(experiments: Path, name: str, data_root_dir: str, val_aurocs: list[float], done: bool) -> Path:
    """Write a minimal GloViTa run folder and return its fold folder."""
    fold = experiments / "generic_3d_dataset" / name / "0"
    (fold / "checkpoints").mkdir(parents=True)
    (fold / "checkpoints" / "best.pt").write_bytes(b"")
    resolved = {
        "data": {"data_root_dir": data_root_dir},
        "model": {"encoder": {"checkpoint_path": "checkpoints/nnFoundationViT/checkpoint_final.pth"}},
        "peft": {"method": "full_finetuning"},
        "optimizer": {"lr": 1e-4},
        "training": {"epochs": 100},
        "add_log": {"note": "baseline"},
    }
    (fold / "resolved_config.json").write_text(json.dumps(resolved))
    val = [{"epoch": epoch, "val_AUROC": auroc} for epoch, auroc in enumerate(val_aurocs)]
    (fold / "metrics.json").write_text(json.dumps({"train": [], "val": val}))
    if done:
        (fold / "final_val_metrics.json").write_text(json.dumps({"final_val_F1": 0.9}))
    return fold


def write_predictions(path: Path, rows: list[tuple[str, int, int, float]]) -> None:
    """Write a predictions CSV in the format of predictions_to_csv.py."""
    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["case_id", "label", "pred", "p_open"])
        writer.writerows(rows)


@pytest.fixture
def config() -> dict:
    return valve.load_config()


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    data_dir, glovita_dir = tmp_path / "data", tmp_path / "glovita"
    data_dir.mkdir()
    glovita_dir.mkdir()
    monkeypatch.setenv("VALVE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("GLOVITA_DIR", str(glovita_dir))
    return data_dir, glovita_dir


def test_split_mirror_axes_removes_the_option() -> None:
    axes, rest = split_mirror_axes(["--mirror_axes", "0", "1", "--task.metrics", "f1", "data:x"])
    assert axes == [0, 1]
    assert rest == ["--task.metrics", "f1", "data:x"]


def test_split_mirror_axes_defaults_when_absent() -> None:
    assert split_mirror_axes(["data:x"]) == ([0, 1], ["data:x"])


def test_train_settings_applies_smoke_then_command_line(config: dict) -> None:
    settings = valve.train_settings(config, smoke=True, overrides={"lr": 1e-5, "epochs": None})
    assert settings["epochs"] == config["smoke"]["epochs"]
    assert settings["lr"] == 1e-5
    assert settings["batch_size"] == config["train"]["batch_size"]


def test_build_train_args_full_run(config: dict) -> None:
    settings = valve.train_settings(config, smoke=False, overrides={"note": "baseline"})
    args = valve.build_train_args(config, settings, Path("/data/crops/mitral"))
    assert args[:3] == ["--mirror_axes", "0", "1"]
    assert args[args.index("--optimizer.lr") + 1] == "0.0001"
    assert args[args.index("--add_log.note") + 1] == "baseline"
    assert args[args.index("--model.encoder.checkpoint_path") + 1] == config["encoders"]["vit"]
    assert "--training.train_steps_per_epoch" not in args
    assert args.index("data:generic3d_dataset_config") < args.index("model.encoder:nnfoundation_encoder_config")
    assert args[-1] == "logger:no_logger_config"


def test_build_train_args_smoke_linear_probe_cnn(config: dict) -> None:
    overrides = {"linear_probe": True, "encoder": "cnn"}
    args = valve.build_train_args(config, valve.train_settings(config, True, overrides), Path("crops/aortic"))
    assert args[args.index("--training.train_steps_per_epoch") + 1] == "10"
    assert args[args.index("--model.encoder.checkpoint_path") + 1] == config["encoders"]["cnn"]
    assert args[-1] == "peft:linear_probing_config"


@pytest.mark.parametrize("path", ["/data/crops/mitral", "C:\\data\\crops\\mitral", "/data/crops/mitral/"])
def test_valve_of_handles_linux_and_windows_paths(path: str) -> None:
    assert valve.valve_of(path) == "mitral"


def test_find_select_and_summarize_runs(tmp_path: Path, env: tuple[Path, Path]) -> None:
    _, glovita_dir = env
    experiments = glovita_dir / "experiments"
    make_run(experiments, RUN_NAME, "/data/crops/mitral", [0.7, 0.9, 0.8], done=True)
    later = RUN_NAME.replace("124730", "150000").replace("04f53122", "aaaaaaaa")
    make_run(experiments, later, "C:\\data\\crops\\mitral", [0.6], done=False)
    make_run(experiments, RUN_NAME.replace("04f53122", "bbbbbbbb"), "/data/crops/aortic", [0.5], done=True)

    paths = valve.load_paths()
    assert [run.valve for run in valve.find_runs(experiments)] == ["mitral", "aortic", "mitral"]
    assert valve.select_run(paths, "mitral", None).name == later
    assert valve.select_run(paths, "mitral", "04f53122").name == RUN_NAME

    summary = valve.summarize_run(valve.select_run(paths, "mitral", "04f53122"))
    assert summary["run"] == "20261104_124730_04f53122"
    assert (summary["encoder"], summary["best_metric"], summary["best_val"], summary["best_epoch"]) == (
        "vit",
        "AUROC",
        0.9,
        1,
    )
    assert (summary["epochs"], summary["status"], summary["note"]) == ("3/100", "done", "baseline")


def test_select_run_without_runs_exits(env: tuple[Path, Path]) -> None:
    with pytest.raises(valve.typer.Exit):
        valve.select_run(valve.load_paths(), "mitral", None)


def test_compute_metrics_worked_example() -> None:
    rows = [{"label": 0, "pred": 0, "p_open": 0.1}] * 54 + [{"label": 0, "pred": 1, "p_open": 0.6}] * 6
    rows += [{"label": 1, "pred": 0, "p_open": 0.4}] * 10 + [{"label": 1, "pred": 1, "p_open": 0.9}] * 30
    metrics = valve.compute_metrics(rows)
    assert metrics["accuracy"] == pytest.approx(0.84)
    assert metrics["balanced_accuracy"] == pytest.approx(0.825)
    assert metrics["f1_macro"] == pytest.approx((60 / 76 + 108 / 124) / 2)
    assert metrics["confusion"]["true open"] == {"pred closed": 10, "pred open": 30}


def test_compute_metrics_without_both_classes_has_no_auroc() -> None:
    assert valve.compute_metrics([{"label": 0, "pred": 0, "p_open": 0.2}])["auroc"] is None


def test_split_counts(tmp_path: Path) -> None:
    (tmp_path / "labels.json").write_text(json.dumps({"S1": 0, "S2": 1, "S3": 1}))
    (tmp_path / "splits.json").write_text(json.dumps({"train": ["S1", "S2"], "val": ["S3"], "test": []}))
    counts = valve.split_counts(tmp_path)
    assert (counts["train"][0], counts["train"][1], counts["val"][1], sum(counts["test"].values())) == (1, 1, 1, 0)


def test_cli_train_dry_run_prints_the_command(env: tuple[Path, Path]) -> None:
    data_dir, _ = env
    result = CliRunner().invoke(valve.app, ["train", "aortic", "--lr", "1e-5", "--linear-probe", "--dry-run"])
    assert result.exit_code == 0, result.output
    output = " ".join(result.output.split())
    assert "train_glovita.py" in output
    assert "--optimizer.lr 1e-05" in output
    assert "peft:linear_probing_config" in output
    assert "aortic" in output


def test_cli_errors_with_phase(env: tuple[Path, Path]) -> None:
    data_dir, glovita_dir = env
    fold = make_run(glovita_dir / "experiments", RUN_NAME, str(data_dir / "crops" / "mitral"), [0.9], done=True)
    write_predictions(
        fold / "test_predictions.csv", [("S1", 0, 0, 0.1), ("S2", 0, 1, 0.7), ("S3", 1, 1, 0.9), ("S4", 1, 1, 0.8)]
    )
    with open(data_dir / "blinded_key.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["scan_id", "patient_id", "phase"])
        writer.writerows([("S1", "P1", "ED"), ("S2", "P1", "ES"), ("S3", "P2", "ED"), ("S4", "P2", "ES")])

    result = CliRunner().invoke(valve.app, ["errors", "mitral", "--show-phase"])
    assert result.exit_code == 0, result.output
    assert "1 of 4 test scans misclassified" in result.output
    assert "S2" in result.output and "S1" not in result.output
    assert "Phase ED" in result.output and "Phase ES" in result.output


def test_cli_errors_without_predictions_explains(env: tuple[Path, Path]) -> None:
    data_dir, glovita_dir = env
    make_run(glovita_dir / "experiments", RUN_NAME, str(data_dir / "crops" / "mitral"), [0.9], done=True)
    result = CliRunner().invoke(valve.app, ["errors", "mitral"])
    assert result.exit_code == 1
    assert "valve.py test mitral" in " ".join(result.output.split())


def test_gallery_rows_misclassified_with_references() -> None:
    rows = [
        {"case_id": "A", "label": 0, "pred": 0, "p_open": 0.05},
        {"case_id": "B", "label": 0, "pred": 0, "p_open": 0.3},
        {"case_id": "C", "label": 0, "pred": 1, "p_open": 0.6},
        {"case_id": "D", "label": 1, "pred": 1, "p_open": 0.97},
    ]
    chosen = valve.gallery_rows(rows, show_all=False, reference=True)
    assert [(kind, row["case_id"]) for kind, row in chosen] == [
        ("reference", "A"),
        ("misclassified", "C"),
        ("reference", "D"),
    ]
    assert len(valve.gallery_rows(rows, show_all=True, reference=True)) == 4


def test_crop_mid_slices_orientation() -> None:
    import numpy as np
    import valve_plots

    crop = np.zeros((1, 4, 6, 8))
    en_face, long_a, long_b = valve_plots.crop_mid_slices(crop)
    assert en_face.shape == (6, 4)
    assert long_a.shape == (8, 4)
    assert long_b.shape == (8, 6)


def test_cli_plots_writes_all_figures(env: tuple[Path, Path]) -> None:
    import blosc2
    import numpy as np
    from PIL import Image

    data_dir, glovita_dir = env
    fold = make_run(glovita_dir / "experiments", RUN_NAME, str(data_dir / "crops" / "mitral"), [0.6, 0.9], done=True)
    history = {
        "train": [{"epoch": 0, "train_loss": 0.7}, {"epoch": 1, "train_loss": 0.4}],
        "val": [
            {"epoch": 0, "val_loss": 0.8, "val_AUROC": 0.6, "val_Balanced_Accuracy": 0.5, "val_F1": 0.4},
            {"epoch": 1, "val_loss": 0.5, "val_AUROC": 0.9, "val_Balanced_Accuracy": 0.8, "val_F1": 0.8},
        ],
    }
    (fold / "metrics.json").write_text(json.dumps(history))
    write_predictions(fold / "test_predictions.csv", [("S1", 0, 0, 0.1), ("S2", 0, 1, 0.7), ("S3", 1, 1, 0.9)])
    planes = data_dir / "valve_planes"
    planes.mkdir()
    for view in ("en_face", "long_axis_a", "long_axis_b"):
        Image.fromarray(np.full((20, 20), 128, dtype=np.uint8)).save(planes / f"S2_mitral_{view}.png")
    (data_dir / "crops" / "mitral" / "images").mkdir(parents=True)
    blosc2.asarray(np.zeros((1, 8, 8, 8), dtype=np.float32), urlpath=str(data_dir / "crops/mitral/images/S2.b2nd"))

    result = CliRunner().invoke(valve.app, ["plots", "mitral"])
    assert result.exit_code == 0, result.output
    for name in ("training_curves.png", "confusion_matrix.png", "misclassified_scans.png"):
        assert (fold / "plots" / name).stat().st_size > 0
