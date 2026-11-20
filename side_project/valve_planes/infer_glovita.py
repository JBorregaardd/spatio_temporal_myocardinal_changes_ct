"""Run ``glovita_infer`` and print a progress line after every batch, for ``valve.py classify``.

GloViTa's inference prints nothing while it predicts, which takes long for a large dataset. This wrapper takes the
usual ``glovita_infer`` arguments and wraps the test data loader so every batch prints a line
``@progress <pass> <batch> <batches>``: ``pass`` counts the fold models from 0 (each goes through the whole loader) and
``batch`` the batches done in that pass. ``valve.py`` turns these lines into a progress bar. It must run with GloViTa's
Python, from the GloViTa clone:

    cd ~/image_classification
    .venv/bin/python ~/spatio_temporal_myocardinal_changes_ct/side_project/valve_planes/infer_glovita.py \
        <glovita_infer arguments>
"""
from __future__ import annotations

import sys
from collections.abc import Callable, Iterator
from typing import Any

PROGRESS_PREFIX = "@progress"


def progress_line(pass_index: int, batch: int, batches: int) -> str:
    """Format the progress of one batch as ``@progress <pass> <batch> <batches>``."""
    return f"{PROGRESS_PREFIX} {pass_index} {batch} {batches}"


def parse_progress_line(line: str) -> tuple[int, int, int] | None:
    """Read a line written by ``progress_line``.

    Args:
        line: One line of output.

    Returns:
        Tuple of (pass, batch, batches), or ``None`` if the line is other output.
    """
    parts = line.split()
    if len(parts) != 4 or parts[0] != PROGRESS_PREFIX or not all(part.isdigit() for part in parts[1:]):
        return None
    return int(parts[1]), int(parts[2]), int(parts[3])


class ProgressLoader:
    """Data loader wrapper that prints a progress line before the first batch and after every batch of each pass.

    Other attributes (e.g. ``dataset``, which GloViTa reads the case ids from) are passed on to the wrapped loader.
    """

    def __init__(self, loader: Any) -> None:
        """Wrap ``loader``.

        Args:
            loader: The test data loader.
        """
        self.loader = loader
        self.passes = 0

    def __len__(self) -> int:
        """Number of batches."""
        return len(self.loader)

    def __getattr__(self, name: str) -> Any:
        """Pass other attributes on to the wrapped loader."""
        return getattr(self.loader, name)

    def __iter__(self) -> Iterator[Any]:
        """Yield the batches; the line after a batch is printed once the caller asks for the next one, so it counts
        batches that were predicted, not only loaded."""
        pass_index, batches = self.passes, len(self.loader)
        self.passes += 1
        print(progress_line(pass_index, 0, batches), flush=True)
        for batch_index, batch in enumerate(self.loader, start=1):
            yield batch
            print(progress_line(pass_index, batch_index, batches), flush=True)


def with_progress(build_dataloaders: Callable[..., tuple[Any, Any, Any]]) -> Callable[..., tuple[Any, Any, Any]]:
    """Wrap GloViTa's ``build_dataloaders`` so the test loader it returns prints its progress.

    Args:
        build_dataloaders: GloViTa's function returning the (train, validation, test) loaders.

    Returns:
        The wrapped function.
    """

    def build(*args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
        train, val, test = build_dataloaders(*args, **kwargs)
        return train, val, None if test is None else ProgressLoader(test)

    return build


if __name__ == "__main__":
    import glovita.cli.infer as infer

    infer.build_dataloaders = with_progress(infer.build_dataloaders)
    sys.exit(infer.main())
