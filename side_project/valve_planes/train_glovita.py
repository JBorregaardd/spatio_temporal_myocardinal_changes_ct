"""Run ``glovita_train`` with train-time mirroring restricted to the valve plane (axes 0 and 1).

GloViTa's ``AugmentationConfig.train_kwargs`` defaults to an empty dict, so its CLI exposes no option for it and
``mirror_axes`` cannot be set from the command line. This wrapper parses the usual ``glovita_train`` arguments, sets
``mirror_axes`` in ``train_kwargs`` and starts training. Run it from the GloViTa clone with GloViTa's environment:

    cd $GLOVITA
    uv run --no-sync python $REPO/side_project/valve_planes/train_glovita.py <glovita_train arguments>
"""
from __future__ import annotations

from glovita.cli.train import main
from glovita.configs.cli import parse_root_cli
from glovita.configs.root import RootConfig

MIRROR_AXES = [0, 1]


def build_config() -> RootConfig:
    """Parse the ``glovita_train`` command line and restrict train-time mirroring to ``MIRROR_AXES``.

    Returns:
        The parsed GloViTa root config with ``mirror_axes`` set in the train augmentation kwargs.
    """
    config = parse_root_cli(RootConfig)
    augmentation = config.data.augmentation
    augmentation = augmentation.model_copy(
        update={"train_kwargs": {**augmentation.train_kwargs, "mirror_axes": MIRROR_AXES}}
    )
    data = config.data.model_copy(update={"augmentation": augmentation})
    return config.model_copy(update={"data": data})


if __name__ == "__main__":
    main(build_config())
