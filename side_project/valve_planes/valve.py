"""Train and test the valve open/closed classifier with GloViTa, from this repo.

Settings live in ``valve_config.yaml`` next to this file; the paths in the repo-root ``.env``:

    VALVE_DATA_DIR    data folder holding crops/<valve>/ and blinded_key.csv
    GLOVITA_DIR       clone of MIC-DKFZ/image_classification with its own .venv and pretrained checkpoints
    VALVE_PLANES_DIR  optional: folder with the labelling PNGs of valve_planes.py (default <VALVE_DATA_DIR>/valve_planes)

GloViTa is run with the Python of its own ``.venv`` directly, not through ``uv run``, so its CUDA build of torch is
never re-synced away and an active conda environment does not matter. Examples:

    uv run side_project/valve_planes/valve.py check
    uv run side_project/valve_planes/valve.py smoke mitral
    uv run side_project/valve_planes/valve.py train mitral --lr 1e-5 --note "lower lr"
    uv run side_project/valve_planes/valve.py logs mitral
    uv run side_project/valve_planes/valve.py runs
    uv run side_project/valve_planes/valve.py test mitral
    uv run side_project/valve_planes/valve.py errors mitral
    uv run side_project/valve_planes/valve.py plots mitral
"""
from __future__ import annotations

import csv
import json
import os
import re
import shlex
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import typer
import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
CONFIG_PATH = HERE / "valve_config.yaml"
LABEL_NAMES = {0: "closed", 1: "open"}
SPLITS = ("train", "val", "test")
INFER_ENTRY = "import sys; from glovita.cli.infer import main; sys.exit(main())"
TORCH_CHECK = (
    "import json, torch; print(json.dumps({'version': torch.__version__, 'cuda': torch.cuda.is_available(), "
    "'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))"
)

console = Console()
app = typer.Typer(
    help="Train and test the valve open/closed classifier with GloViTa.", no_args_is_help=True, add_completion=False
)


class Valve(str, Enum):
    """Valves with an exported crop dataset."""

    mitral = "mitral"
    aortic = "aortic"


class Encoder(str, Enum):
    """Pretrained nnFoundation encoders (keys of ``encoders`` in ``valve_config.yaml``)."""

    vit = "vit"
    cnn = "cnn"


@dataclass(frozen=True)
class Paths:
    """Locations of the data and the GloViTa clone."""

    data_dir: Path
    glovita_dir: Path
    planes_dir: Path

    def crops(self, valve: str) -> Path:
        """Return the GloViTa dataset folder of one valve.

        Args:
            valve: Valve name.

        Returns:
            Path of ``<data_dir>/crops/<valve>``.
        """
        return self.data_dir / "crops" / valve

    @property
    def python(self) -> Path:
        """Python executable of GloViTa's ``.venv``."""
        if os.name == "nt":
            return self.glovita_dir / ".venv" / "Scripts" / "python.exe"
        return self.glovita_dir / ".venv" / "bin" / "python"

    @property
    def experiments_dir(self) -> Path:
        """Folder GloViTa writes its runs to."""
        return self.glovita_dir / "experiments"

    @property
    def logs_dir(self) -> Path:
        """Folder the logs of background training runs are written to."""
        return self.glovita_dir / "valve_logs"


@dataclass(frozen=True)
class Run:
    """One GloViTa training run (one fold folder holding ``resolved_config.json``)."""

    path: Path
    valve: str
    resolved: dict[str, Any]

    @property
    def name(self) -> str:
        """Run folder name, starting with the start time (``YYYYmmdd_HHMMSS``)."""
        return self.path.parent.name

    @property
    def short_name(self) -> str:
        """Start time and unique id of the run, e.g. ``20261104_124730_04f53122``."""
        return f"{self.name[:15]}_{self.name[-8:]}" if len(self.name) > 24 else self.name

    @property
    def group_dir(self) -> Path:
        """Run folder holding the fold folders, as expected by ``glovita_infer --exp_dir``."""
        return self.path.parent


def load_paths() -> Paths:
    """Read ``VALVE_DATA_DIR`` and ``GLOVITA_DIR`` from the repo-root ``.env`` or the environment.

    Returns:
        The data and GloViTa locations.

    Raises:
        typer.Exit: If a variable is missing.
    """
    load_dotenv(REPO_ROOT / ".env")
    values = {name: os.environ.get(name) for name in ("VALVE_DATA_DIR", "GLOVITA_DIR")}
    missing = [name for name, value in values.items() if not value]
    if missing:
        console.print(f"[red]Set {', '.join(missing)} in {REPO_ROOT / '.env'} (see .env.example).[/red]")
        raise typer.Exit(1)
    data_dir = Path(values["VALVE_DATA_DIR"]).expanduser()
    planes_dir = os.environ.get("VALVE_PLANES_DIR")
    return Paths(
        data_dir=data_dir,
        glovita_dir=Path(values["GLOVITA_DIR"]).expanduser(),
        planes_dir=Path(planes_dir).expanduser() if planes_dir else data_dir / "valve_planes",
    )


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Read ``valve_config.yaml``.

    Args:
        path: Config file.

    Returns:
        The parsed config.
    """
    with open(path) as file:
        return yaml.safe_load(file)


def train_settings(config: dict[str, Any], smoke: bool, overrides: dict[str, Any]) -> dict[str, Any]:
    """Merge the train settings, the smoke-test overrides and the command line overrides.

    Args:
        config: Parsed ``valve_config.yaml``.
        smoke: Apply the ``smoke`` section on top of ``train``.
        overrides: Command line values; ``None`` values are ignored.

    Returns:
        The settings of this run.
    """
    settings = dict(config["train"])
    if smoke:
        settings.update(config.get("smoke", {}))
    settings.update({key: value for key, value in overrides.items() if value is not None})
    return settings


def build_train_args(config: dict[str, Any], settings: dict[str, Any], crops_dir: Path) -> list[str]:
    """Build the ``train_glovita.py`` arguments of one run.

    Args:
        config: Parsed ``valve_config.yaml``.
        settings: Merged settings from ``train_settings``.
        crops_dir: GloViTa dataset folder of the valve.

    Returns:
        The arguments, starting with ``--mirror_axes``.
    """
    args = ["--mirror_axes", *map(str, config["mirror_axes"]), "--task.metrics", *config["metrics"]]
    args += [
        "--optimizer.name", "AdamW",
        "--optimizer.lr", str(float(settings["lr"])),
        "--optimizer.weight_decay", str(float(settings["weight_decay"])),
        "--optimizer.warmup_epochs", str(settings["warmup_epochs"]),
        "--training.epochs", str(settings["epochs"]),
        "--training.precision", str(settings["precision"]),
        "--training.seed", str(settings["seed"]),
        "--training.best_checkpoint_metric", str(settings["best_checkpoint_metric"]),
    ]  # fmt: skip
    for key in ("train_steps_per_epoch", "val_steps_per_epoch"):
        if settings.get(key):
            args += [f"--training.{key}", str(settings[key])]
    args += ["--dataloading.batch_size", str(settings["batch_size"])]
    args += ["--dataloading.num_workers", str(settings["num_workers"])]
    if settings.get("note"):
        args += ["--add_log.note", str(settings["note"])]
    args += ["data:generic3d_dataset_config", "--data.data_root_dir", str(crops_dir)]
    args += ["model.encoder:nnfoundation_encoder_config"]
    args += ["--model.encoder.checkpoint_path", config["encoders"][settings["encoder"]]]
    args += ["--model.encoder.input_channels", "1", "--model.encoder.input_shape", *map(str, config["input_shape"])]
    args += ["model.head:classification_head_config", "logger:no_logger_config"]
    if settings.get("linear_probe"):
        args += ["peft:linear_probing_config"]
    return args


def valve_of(data_root_dir: str) -> str:
    """Return the valve of a run from its dataset folder, written on Linux or Windows.

    Args:
        data_root_dir: ``data.data_root_dir`` of the run.

    Returns:
        The last path component, e.g. ``mitral``.
    """
    return re.split(r"[\\/]", data_root_dir.rstrip("\\/"))[-1]


def find_runs(experiments_dir: Path, valve: Optional[str] = None) -> list[Run]:
    """Find all GloViTa runs, oldest first.

    Args:
        experiments_dir: GloViTa experiments folder.
        valve: Only return runs of this valve.

    Returns:
        The runs, sorted by start time.
    """
    runs = []
    for config_path in experiments_dir.rglob("resolved_config.json"):
        resolved = json.loads(config_path.read_text())
        run_valve = valve_of(str(resolved.get("data", {}).get("data_root_dir", "")))
        if valve is None or run_valve == valve:
            runs.append(Run(path=config_path.parent, valve=run_valve, resolved=resolved))
    return sorted(runs, key=lambda run: (run.name, run.path.name))


def select_run(paths: Paths, valve: str, name: Optional[str]) -> Run:
    """Return the run called ``name`` (or the latest run of the valve) that has a best checkpoint.

    Args:
        paths: Data and GloViTa locations.
        valve: Valve name.
        name: Run folder name or a unique part of it; ``None`` for the latest run.

    Returns:
        The selected run.

    Raises:
        typer.Exit: If no run matches.
    """
    runs = [run for run in find_runs(paths.experiments_dir, valve) if (run.path / "checkpoints" / "best.pt").exists()]
    if name is not None:
        runs = [run for run in runs if name in run.name]
    if not runs:
        console.print(f"[red]No {valve} run with a best checkpoint{f' matching {name!r}' if name else ''}.[/red]")
        raise typer.Exit(1)
    return runs[-1]


def read_json(path: Path) -> Optional[dict[str, Any]]:
    """Read a JSON file if it exists.

    Args:
        path: JSON file.

    Returns:
        The content, or ``None`` if the file does not exist.
    """
    return json.loads(path.read_text()) if path.exists() else None


def summarize_run(run: Run) -> dict[str, Any]:
    """Collect the settings and results of a run for the ``runs`` table.

    Args:
        run: The run.

    Returns:
        Dict with the run name, settings, progress and validation and test metrics.
    """
    resolved = run.resolved
    checkpoint = str(resolved.get("model", {}).get("encoder", {}).get("checkpoint_path", ""))
    encoder = "vit" if "ViT" in checkpoint else "cnn" if "CNN" in checkpoint else checkpoint
    history = read_json(run.path / "metrics.json") or {"val": []}
    val_rows = [row for row in history.get("val", []) if "val_AUROC" in row]
    best = max(val_rows, key=lambda row: row["val_AUROC"], default=None)
    final = read_json(run.path / "final_val_metrics.json")
    return {
        "run": run.short_name,
        "valve": run.valve,
        "encoder": encoder,
        "peft": resolved.get("peft", {}).get("method", ""),
        "lr": resolved.get("optimizer", {}).get("lr"),
        "epochs": f"{len(history.get('val', []))}/{resolved.get('training', {}).get('epochs', '?')}",
        "status": "done" if final else "running or stopped",
        "best_val_auroc": best["val_AUROC"] if best else None,
        "best_epoch": best["epoch"] if best else None,
        "final_val": final,
        "test": read_json(run.path / "test_metrics.json"),
        "note": (resolved.get("add_log") or {}).get("note", ""),
    }


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute the test metrics from per-scan predictions.

    Args:
        rows: Rows with integer ``label`` and ``pred`` and float ``p_open``.

    Returns:
        Number of scans, accuracy, balanced accuracy, macro F1, AUROC (``None`` with only one class) and the
        confusion matrix as ``{"true closed": {"pred closed": n, ...}, ...}``.
    """
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

    labels = [row["label"] for row in rows]
    preds = [row["pred"] for row in rows]
    counts = Counter(zip(labels, preds))
    return {
        "n": len(rows),
        "accuracy": accuracy_score(labels, preds),
        "balanced_accuracy": balanced_accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "auroc": roc_auc_score(labels, [row["p_open"] for row in rows]) if len(set(labels)) == 2 else None,
        "confusion": {
            f"true {LABEL_NAMES[true]}": {f"pred {LABEL_NAMES[pred]}": counts[(true, pred)] for pred in LABEL_NAMES}
            for true in LABEL_NAMES
        },
    }


def read_predictions(path: Path) -> list[dict[str, Any]]:
    """Read the CSV written by ``predictions_to_csv.py``.

    Args:
        path: Predictions CSV.

    Returns:
        Rows with ``case_id``, integer ``label`` and ``pred`` and float ``p_open``.
    """
    with open(path, newline="") as file:
        return [
            {
                "case_id": row["case_id"],
                "label": int(row["label"]),
                "pred": int(row["pred"]),
                "p_open": float(row["p_open"]),
            }
            for row in csv.DictReader(file)
        ]


def split_counts(crops_dir: Path) -> dict[str, Counter]:
    """Count the closed and open scans per split of a valve dataset.

    Args:
        crops_dir: GloViTa dataset folder with ``labels.json`` and ``splits.json``.

    Returns:
        ``{split: Counter({label: count})}``.
    """
    labels = json.loads((crops_dir / "labels.json").read_text())
    splits = json.loads((crops_dir / "splits.json").read_text())
    if isinstance(splits, list):
        splits = splits[0]
    return {split: Counter(labels[case_id] for case_id in splits.get(split, [])) for split in SPLITS}


def format_metric(value: Optional[float]) -> str:
    """Format a metric with three decimals, or ``-`` when missing.

    Args:
        value: Metric value.

    Returns:
        The formatted value.
    """
    return "-" if value is None else f"{value:.3f}"


def print_metrics(title: str, metrics: dict[str, Any]) -> None:
    """Print metrics and the confusion matrix from ``compute_metrics``.

    Args:
        title: Heading.
        metrics: Output of ``compute_metrics``.
    """
    console.print(f"\n[bold]{title}[/bold] ({metrics['n']} scans)")
    for key in ("accuracy", "balanced_accuracy", "f1_macro", "auroc"):
        console.print(f"  {key:<18} {format_metric(metrics[key])}")
    table = Table("", *[f"pred {name}" for name in LABEL_NAMES.values()])
    for true, row in metrics["confusion"].items():
        table.add_row(true, *[str(count) for count in row.values()])
    console.print(table)


def require(path: Path, hint: str) -> None:
    """Stop with a message if a required file or folder is missing.

    Args:
        path: Required path.
        hint: What to do about it.

    Raises:
        typer.Exit: If the path does not exist.
    """
    if not path.exists():
        console.print(f"[red]Missing {path}[/red]\n  {hint}")
        raise typer.Exit(1)


def launch(command: list[str], cwd: Path, log_path: Optional[Path]) -> int:
    """Run a command in the foreground, or start it in the background with its output in ``log_path``.

    Args:
        command: Command and arguments.
        cwd: Working directory.
        log_path: Log file for a background run; ``None`` runs in the foreground.

    Returns:
        The exit code (foreground) or the process id (background).
    """
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    if log_path is None:
        return subprocess.run(command, cwd=cwd, env=env).returncode
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "w")
    log_file.write(f"$ cd {cwd}\n$ {shlex.join(command)}\n\n")
    log_file.flush()
    detach: dict[str, Any] = (
        {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS}
        if os.name == "nt"
        else {"start_new_session": True}
    )
    process = subprocess.Popen(
        command, cwd=cwd, env=env, stdout=log_file, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, **detach
    )
    return process.pid


def start_training(valve: Valve, smoke: bool, foreground: bool, dry_run: bool, overrides: dict[str, Any]) -> None:
    """Build the training command of one run and run it.

    Args:
        valve: Valve to train on.
        smoke: Use the smoke-test settings.
        foreground: Run in this terminal instead of in the background.
        dry_run: Only print the command.
        overrides: Command line overrides of the train settings.

    Raises:
        typer.Exit: If a required file is missing or a foreground run fails.
    """
    paths = load_paths()
    config = load_config()
    settings = train_settings(config, smoke, overrides)
    crops_dir = paths.crops(valve.value)
    command = [str(paths.python), str(HERE / "train_glovita.py"), *build_train_args(config, settings, crops_dir)]
    if dry_run:
        console.print(f"cd {paths.glovita_dir}\n{shlex.join(command)}", soft_wrap=True)
        return
    require(paths.python, "Set up GloViTa first (REMOTE_TRAINING.md, step 3).")
    require(crops_dir / "dataset.json", "Export the crops first (REMOTE_TRAINING.md, step 2).")
    require(paths.glovita_dir / config["encoders"][settings["encoder"]], "Download the pretrained weights (step 3).")
    if foreground:
        code = launch(command, paths.glovita_dir, None)
        if code:
            raise typer.Exit(code)
        return
    log_path = paths.logs_dir / f"train_{valve.value}_{datetime.now():%Y%m%d_%H%M%S}.log"
    pid = launch(command, paths.glovita_dir, log_path)
    console.print(f"Started {valve.value} training in the background (pid {pid}).\nLog: {log_path}")
    console.print(f"Follow it with:  uv run side_project/valve_planes/valve.py logs {valve.value}")


@app.command()
def train(
    valve: Valve = typer.Argument(..., help="Valve to train on."),
    lr: Optional[float] = typer.Option(None, help="Learning rate."),
    epochs: Optional[int] = typer.Option(None, help="Number of epochs."),
    batch_size: Optional[int] = typer.Option(None, help="Batch size."),
    encoder: Optional[Encoder] = typer.Option(None, help="Pretrained encoder."),
    linear_probe: Optional[bool] = typer.Option(
        None, "--linear-probe/--full-finetune", help="Train only the head, or all weights."
    ),
    precision: Optional[str] = typer.Option(None, help="bf16, fp16 or no."),
    note: Optional[str] = typer.Option(None, help="Short note saved with the run and shown by `runs`."),
    foreground: bool = typer.Option(False, help="Run in this terminal instead of in the background."),
    dry_run: bool = typer.Option(False, help="Only print the GloViTa command."),
) -> None:
    """Train a classifier for one valve with the settings in valve_config.yaml (in the background by default)."""
    overrides = {
        "lr": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "encoder": encoder.value if encoder else None,
        "linear_probe": linear_probe,
        "precision": precision,
        "note": note,
    }
    start_training(valve, smoke=False, foreground=foreground, dry_run=dry_run, overrides=overrides)


@app.command()
def smoke(
    valve: Valve = typer.Argument(Valve.mitral, help="Valve to train on."),
    precision: Optional[str] = typer.Option(None, help="bf16, fp16 or no."),
    dry_run: bool = typer.Option(False, help="Only print the GloViTa command."),
) -> None:
    """Run a short training (the `smoke` settings) in this terminal to check that everything works."""
    overrides = {"precision": precision, "note": "smoke test"}
    start_training(valve, smoke=True, foreground=True, dry_run=dry_run, overrides=overrides)


@app.command()
def logs(
    valve: Valve = typer.Argument(..., help="Valve of the training run."),
    lines: int = typer.Option(40, help="Number of earlier lines to show."),
    follow: bool = typer.Option(True, help="Keep printing new lines (stop with Ctrl+C)."),
) -> None:
    """Show the log of the latest background training run of a valve."""
    paths = load_paths()
    log_paths = sorted(paths.logs_dir.glob(f"train_{valve.value}_*.log"))
    if not log_paths:
        console.print(f"[red]No {valve.value} training logs in {paths.logs_dir}.[/red]")
        raise typer.Exit(1)
    log_path = log_paths[-1]
    console.print(f"[dim]{log_path}[/dim]")
    with open(log_path) as file:
        sys.stdout.write("".join(file.readlines()[-lines:]))
        if not follow:
            return
        try:
            while True:
                line = file.readline()
                if line:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                else:
                    time.sleep(1)
        except KeyboardInterrupt:
            pass


@app.command()
def runs(valve: Optional[Valve] = typer.Argument(None, help="Only show runs of this valve.")) -> None:
    """List all training runs with their settings and validation (and test) metrics."""
    paths = load_paths()
    summaries = [summarize_run(run) for run in find_runs(paths.experiments_dir, valve.value if valve else None)]
    if not summaries:
        console.print(f"No runs in {paths.experiments_dir}.")
        return
    table = Table("run", "valve", "encoder", "peft", "lr", "epochs", "best val AUROC (epoch)", "final val F1 / bal acc")
    table.add_column("test F1 / AUROC")
    table.add_column("note")
    for summary in summaries:
        final = summary["final_val"] or {}
        test = summary["test"] or {}
        best = format_metric(summary["best_val_auroc"])
        table.add_row(
            summary["run"],
            summary["valve"],
            summary["encoder"],
            summary["peft"],
            f"{summary['lr']:g}" if summary["lr"] is not None else "-",
            summary["epochs"],
            f"{best} ({summary['best_epoch']})" if summary["best_epoch"] is not None else best,
            f"{format_metric(final.get('final_val_F1'))} / {format_metric(final.get('final_val_Balanced_Accuracy'))}",
            f"{format_metric(test.get('f1_macro'))} / {format_metric(test.get('auroc'))}" if test else "-",
            str(summary["note"]),
        )
    console.print(table)


@app.command()
def test(
    valve: Valve = typer.Argument(..., help="Valve of the run."),
    run: Optional[str] = typer.Option(None, help="Run folder name or a unique part of it; default: the latest run."),
    force: bool = typer.Option(False, help="Evaluate again even if the run was already tested."),
) -> None:
    """Evaluate the best checkpoint of a run on the test split, once the settings are final."""
    paths = load_paths()
    selected = select_run(paths, valve.value, run)
    metrics_path = selected.path / "test_metrics.json"
    console.print(f"Run: {selected.name}")
    if metrics_path.exists() and not force:
        print_metrics("Test results (already evaluated; --force to run again)", json.loads(metrics_path.read_text()))
        return
    pred_path = selected.path / "test_predictions.pt"
    csv_path = selected.path / "test_predictions.csv"
    infer = [
        str(paths.python), "-c", INFER_ENTRY,
        "--exp_dir", str(selected.group_dir), "--checkpoint", "best", "--metrics", "acc", "f1",
        "--pred_output", str(pred_path),
        "data:generic3d_dataset_config", "--data.data_root_dir", str(paths.crops(valve.value)),
    ]  # fmt: skip
    if launch(infer, paths.glovita_dir, None):
        raise typer.Exit(1)
    if launch([str(paths.python), str(HERE / "predictions_to_csv.py"), str(pred_path), str(csv_path)], HERE, None):
        raise typer.Exit(1)
    metrics = compute_metrics(read_predictions(csv_path))
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print_metrics("Test results", metrics)
    console.print(f"\nPer-scan predictions: {csv_path}\nMisclassified scans:  valve.py errors {valve.value}")


@app.command()
def errors(
    valve: Valve = typer.Argument(..., help="Valve of the run."),
    run: Optional[str] = typer.Option(None, help="Run folder name or a unique part of it; default: the latest run."),
    show_all: bool = typer.Option(False, "--all", help="List every test scan, not only the misclassified ones."),
    show_phase: bool = typer.Option(
        False, help="Unblind: add patient and phase from blinded_key.csv and the metrics per phase."
    ),
) -> None:
    """List the misclassified test scans of a run, sorted by predicted probability of open."""
    paths = load_paths()
    selected = select_run(paths, valve.value, run)
    csv_path = selected.path / "test_predictions.csv"
    require(csv_path, f"Run `valve.py test {valve.value}` first.")
    rows = sorted(read_predictions(csv_path), key=lambda row: row["p_open"])
    key = load_blinded_key(paths) if show_phase else {}
    columns = ["scan", "true", "pred", "P(open)"] + (["patient", "phase"] if show_phase else [])
    table = Table(*columns, title=f"{valve.value} test, run {selected.short_name}")
    for row in rows:
        wrong = row["label"] != row["pred"]
        if not (wrong or show_all):
            continue
        cells = [row["case_id"], LABEL_NAMES[row["label"]], LABEL_NAMES[row["pred"]], f"{row['p_open']:.3f}"]
        if show_phase:
            info = key.get(row["case_id"], {})
            cells += [info.get("patient_id", "?"), info.get("phase", "?")]
        table.add_row(*cells, style="red" if wrong else None)
    console.print(table)
    console.print(f"{sum(row['label'] != row['pred'] for row in rows)} of {len(rows)} test scans misclassified.")
    if show_phase:
        for phase in sorted({key.get(row["case_id"], {}).get("phase", "?") for row in rows}):
            phase_rows = [row for row in rows if key.get(row["case_id"], {}).get("phase", "?") == phase]
            print_metrics(f"Phase {phase}", compute_metrics(phase_rows))


def load_blinded_key(paths: Paths) -> dict[str, dict[str, str]]:
    """Read ``blinded_key.csv`` (scan id to patient id and phase).

    Args:
        paths: Data and GloViTa locations.

    Returns:
        ``{scan_id: {"scan_id", "patient_id", "phase"}}``.

    Raises:
        typer.Exit: If the key does not exist.
    """
    key_path = paths.data_dir / "blinded_key.csv"
    require(key_path, "The blinded key is needed to show the phase.")
    with open(key_path, newline="") as file:
        return {row["scan_id"]: row for row in csv.DictReader(file)}


def gallery_rows(rows: list[dict[str, Any]], show_all: bool, reference: bool) -> list[tuple[str, dict[str, Any]]]:
    """Choose the test scans for the gallery, sorted by predicted probability of open.

    Args:
        rows: Per-scan predictions from ``read_predictions``.
        show_all: Include every test scan instead of only the misclassified ones.
        reference: Add the most confidently correct closed and open scan for comparison.

    Returns:
        ``(kind, row)`` pairs with kind ``misclassified``, ``correct`` or ``reference``.
    """
    chosen = [
        ("misclassified" if row["label"] != row["pred"] else "correct", row)
        for row in rows
        if show_all or row["label"] != row["pred"]
    ]
    if reference and not show_all:
        correct = [row for row in rows if row["label"] == row["pred"]]
        closed = [row for row in correct if row["label"] == 0]
        opened = [row for row in correct if row["label"] == 1]
        if closed:
            chosen.insert(0, ("reference", min(closed, key=lambda row: row["p_open"])))
        if opened:
            chosen.append(("reference", max(opened, key=lambda row: row["p_open"])))
    return chosen


@app.command()
def plots(
    valve: Valve = typer.Argument(..., help="Valve of the run."),
    run: Optional[str] = typer.Option(None, help="Run folder name or a unique part of it; default: the latest run."),
    show_all: bool = typer.Option(False, "--all", help="Gallery of every test scan, not only the misclassified ones."),
    reference: bool = typer.Option(True, help="Add the most confidently correct closed and open scan to the gallery."),
    show_phase: bool = typer.Option(False, help="Unblind: add patient and phase from blinded_key.csv to the gallery."),
) -> None:
    """Plot the training curves and, after `test`, the confusion matrix and a gallery of the misclassified scans."""
    import blosc2
    import numpy as np
    from PIL import Image

    import valve_plots as figures

    paths = load_paths()
    selected = select_run(paths, valve.value, run)
    summary = summarize_run(selected)
    out_dir = selected.path / "plots"
    out_dir.mkdir(exist_ok=True)
    settings = f"{summary['encoder']} {summary['peft'].replace('_', ' ')}, lr {summary['lr']:g}, run {summary['run']}"
    if summary["note"]:
        settings += f" ({summary['note']})"
    written = []

    history = read_json(selected.path / "metrics.json") or {}
    best_metric = selected.resolved.get("training", {}).get("best_checkpoint_metric") or "val_AUROC"
    curves_path = out_dir / "training_curves.png"
    figures.plot_training_curves(
        history, best_metric, f"{valve.value.capitalize()} valve: training", settings, curves_path
    )
    written.append(curves_path)

    csv_path = selected.path / "test_predictions.csv"
    if not csv_path.exists():
        console.print(
            f"[yellow]No test predictions yet; run `valve.py test {valve.value}` for the other plots.[/yellow]"
        )
    else:
        rows = sorted(read_predictions(csv_path), key=lambda row: row["p_open"])
        confusion_path = out_dir / "confusion_matrix.png"
        title = f"{valve.value.capitalize()} valve: test confusion matrix"
        figures.plot_confusion_matrix(compute_metrics(rows), title, confusion_path)
        written.append(confusion_path)

        key = load_blinded_key(paths) if show_phase else {}
        entries = []
        for kind, row in gallery_rows(rows, show_all, reference):
            case_id = row["case_id"]
            text = f"{case_id} · manual label {LABEL_NAMES[row['label']]} · predicted {LABEL_NAMES[row['pred']]}"
            text += f" · P(open) {row['p_open']:.2f}"
            if case_id in key:
                text += f" · patient {key[case_id]['patient_id']}, {key[case_id]['phase']}"
            views = []
            for view in ("en_face", "long_axis_a", "long_axis_b"):
                png = paths.planes_dir / f"{case_id}_{valve.value}_{view}.png"
                views.append(np.asarray(Image.open(png)) if png.exists() else None)
            crop_path = paths.crops(valve.value) / "images" / f"{case_id}.b2nd"
            crop_slices = figures.crop_mid_slices(blosc2.open(str(crop_path))[:]) if crop_path.exists() else []
            flag, color = {
                "misclassified": ("misclassified", figures.CRITICAL),
                "correct": ("correct", figures.GOOD),
                "reference": ("reference", figures.INK_MUTED),
            }[kind]
            entries.append(figures.GalleryEntry(text, flag, color, views, crop_slices))
        if entries and (show_all or any(kind == "misclassified" for kind, _ in gallery_rows(rows, False, False))):
            gallery_path = out_dir / ("test_scans.png" if show_all else "misclassified_scans.png")
            n_wrong = sum(row["label"] != row["pred"] for row in rows)
            subtitle = f"{n_wrong} of {len(rows)} test scans misclassified · {settings}"
            if not any(view is not None for entry in entries for view in entry.views):
                console.print(f"[yellow]No labelling PNGs found in {paths.planes_dir} (set VALVE_PLANES_DIR).[/yellow]")
            figures.plot_scan_gallery(entries, f"{valve.value.capitalize()} valve: test scans", subtitle, gallery_path)
            written.append(gallery_path)
        else:
            console.print("No misclassified test scans, so no gallery.")
    console.print("Plots written:\n" + "\n".join(f"  {path}" for path in written))


@app.command()
def check() -> None:
    """Check the paths, GloViTa's environment and GPU, the pretrained weights and the crop datasets."""
    paths = load_paths()
    config = load_config()
    ok = True

    def report(passed: bool, message: str) -> None:
        nonlocal ok
        ok &= passed
        console.print(f"{'[green]OK  [/green]' if passed else '[red]FAIL[/red]'} {message}")

    report(paths.data_dir.is_dir(), f"VALVE_DATA_DIR  {paths.data_dir}")
    report(paths.glovita_dir.is_dir(), f"GLOVITA_DIR     {paths.glovita_dir}")
    report(paths.python.exists(), f"GloViTa Python  {paths.python}")
    if paths.python.exists():
        result = subprocess.run([str(paths.python), "-c", TORCH_CHECK], capture_output=True, text=True)
        if result.returncode == 0:
            torch_info = json.loads(result.stdout.strip().splitlines()[-1])
            report(torch_info["cuda"], f"torch {torch_info['version']}, CUDA GPU: {torch_info['device']}")
        else:
            report(False, f"torch cannot be imported:\n{result.stderr.strip().splitlines()[-1]}")
    for name, checkpoint in config["encoders"].items():
        report((paths.glovita_dir / checkpoint).exists(), f"{name} weights     {checkpoint}")
    for valve in Valve:
        crops_dir = paths.crops(valve.value)
        files = [crops_dir / name for name in ("dataset.json", "labels.json", "splits.json")]
        report(all(file.exists() for file in files), f"{valve.value} crops    {crops_dir}")
        if all(file.exists() for file in files):
            counts = split_counts(crops_dir)
            summary = ", ".join(f"{split} {counts[split][0]} closed / {counts[split][1]} open" for split in SPLITS)
            console.print(f"       {summary}")
    if not ok:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
