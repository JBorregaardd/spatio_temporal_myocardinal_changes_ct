"""Run ``glovita_train`` with train-time mirroring restricted to chosen axes (by default the valve plane, axes 0 and 1).

GloViTa's ``AugmentationConfig.train_kwargs`` defaults to an empty dict, so its CLI exposes no option for it and
``mirror_axes`` cannot be set from the command line. This wrapper takes the usual ``glovita_train`` arguments plus an
optional ``--mirror_axes 0 1``, sets ``mirror_axes`` in ``train_kwargs`` and starts training. It must run with
GloViTa's Python, from the GloViTa clone. ``valve.py train`` builds and runs this command; to run it by hand:

    cd ~/image_classification
    .venv/bin/python ~/spatio_temporal_myocardinal_changes_ct/side_project/valve_planes/train_glovita.py \
        --mirror_axes 0 1 <glovita_train arguments>
"""
from __future__ import annotations

import sys
from typing import TYPE_CHECKING

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


if __name__ == "__main__":
    from glovita.cli.train import main

    main(build_config(sys.argv[1:]))
