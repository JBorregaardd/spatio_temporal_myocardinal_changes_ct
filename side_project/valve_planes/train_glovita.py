"""Run ``glovita_train`` with train-time mirroring restricted to chosen axes (by default the valve plane, axes 0 and 1).

GloViTa's ``AugmentationConfig.train_kwargs`` defaults to an empty dict, so its CLI exposes no option for it and
``mirror_axes`` cannot be set from the command line. This wrapper takes the usual ``glovita_train`` arguments plus an
optional ``--mirror_axes 0 1``, sets ``mirror_axes`` in ``train_kwargs`` and starts training. It must run with
GloViTa's Python, from the GloViTa clone. ``valve.py train`` builds and runs this command; to run it by hand:

    cd ~/image_classification
    .venv/bin/python ~/spatio_temporal_myocardinal_changes_ct/side_project/valve_planes/train_glovita.py \
        --mirror_axes 0 1 <glovita_train arguments>

It also replaces GloViTa's checkpoint saving (see ``save_lean_checkpoint``): GloViTa writes the weights plus the AdamW
state of every epoch to its own file and never deletes them, about 8 GB per epoch for the ViT, which fills the disk.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from glovita.configs.root import RootConfig

MIRROR_AXES = [0, 1]


def split_mirror_axes(args: list[str]) -> tuple[list[int], list[str]]:
    """Remove ``--mirror_axes a b ...`` from the arguments.

    Args:
        args: Command line arguments, possibly containing ``--mirror_axes`` followed by one or more axis indices.

    Returns:
        Tuple of (mirror axes, remaining arguments). The axes default to ``MIRROR_AXES`` when the option is absent.
    """
    if "--mirror_axes" not in args:
        return list(MIRROR_AXES), list(args)
    start = args.index("--mirror_axes")
    end = start + 1
    while end < len(args) and args[end].isdigit():
        end += 1
    return [int(axis) for axis in args[start + 1 : end]], args[:start] + args[end:]


def build_config(args: list[str]) -> RootConfig:
    """Parse the ``glovita_train`` arguments and set ``mirror_axes`` in the train augmentation kwargs.

    Args:
        args: ``glovita_train`` arguments, optionally with ``--mirror_axes``.

    Returns:
        The parsed GloViTa root config with ``mirror_axes`` set in the train augmentation kwargs.
    """
    from glovita.configs.cli import parse_root_cli
    from glovita.configs.root import RootConfig

    mirror_axes, glovita_args = split_mirror_axes(args)
    config = parse_root_cli(RootConfig, glovita_args)
    augmentation = config.data.augmentation
    augmentation = augmentation.model_copy(
        update={"train_kwargs": {**augmentation.train_kwargs, "mirror_axes": mirror_axes}}
    )
    data = config.data.model_copy(update={"augmentation": augmentation})
    return config.model_copy(update={"data": data})


def checkpoint_files(epoch: int, n_epochs: int, is_best: bool) -> list[str]:
    """Return the checkpoint files to write after a save request of GloViTa's trainer.

    GloViTa asks for a save with ``is_best=True`` when the validation metric improves and with ``is_best=False`` after
    every epoch. Only ``best.pt`` (used by ``valve.py test``) and the final ``last.pt`` are kept.

    Args:
        epoch: Zero-based epoch of the request.
        n_epochs: Number of epochs of the run.
        is_best: Whether the request comes from an improved validation metric.

    Returns:
        File names to write, possibly none.
    """
    return ["best.pt"] if is_best else ["last.pt"] if epoch == n_epochs - 1 else []


def save_atomically(state: dict[str, Any], path: Path) -> None:
    """Write ``state`` with ``torch.save`` to a temporary file and move it into place, so a failed write (e.g. a full
    disk) never leaves a half-written checkpoint behind."""
    import torch

    temporary = path.with_name(f"{path.name}.tmp")
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_lean_checkpoint(self: Any, model: Any, optimizer: Any, epoch: int, is_best: bool = False) -> None:
    """Replacement for ``Trainer._save_checkpoint`` that keeps only ``best.pt`` and the final ``last.pt``.

    The files hold the epoch, the model weights and the nnFoundation adaptation plan, as GloViTa's own checkpoints do,
    but not the optimizer state, which only matters for resuming a run (``valve.py`` never does) and is twice the size
    of the weights. ``glovita_infer`` reads only the weights and the plan.

    Args:
        self: The GloViTa ``Trainer``.
        model: Model being trained (possibly wrapped by Accelerate).
        optimizer: Optimizer; unused.
        epoch: Zero-based epoch.
        is_best: Whether the validation metric improved.
    """
    files = checkpoint_files(epoch, self.cfg.epochs, is_best)
    if not files:
        return
    checkpoint_dir = Path(self.log_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = self.accelerator.unwrap_model(model)
    state: dict[str, Any] = {"epoch": epoch, "model": unwrapped.state_dict()}
    adaptation_plan = getattr(getattr(unwrapped, "encoder", None), "adaptation_plan", None)
    if adaptation_plan is not None:
        state["nnssl_adaptation_plan"] = adaptation_plan
    for name in files:
        save_atomically(state, checkpoint_dir / name)


if __name__ == "__main__":
    from glovita.cli.train import main
    from glovita.training.trainer import Trainer

    Trainer._save_checkpoint = save_lean_checkpoint
    main(build_config(sys.argv[1:]))
