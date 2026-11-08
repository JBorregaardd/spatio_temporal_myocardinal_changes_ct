"""Convert the ``--pred_output`` file of ``glovita_infer`` (a torch ``.pt`` dict) to a CSV with one row per scan.

Columns: ``case_id``, ``label`` and ``pred`` (0 = closed, 1 = open) and ``p_open`` (predicted probability of open).
Needs torch, so it runs with GloViTa's Python; ``valve.py test`` calls it after inference.

Example:
    ~/image_classification/.venv/bin/python predictions_to_csv.py test_predictions.pt test_predictions.csv
"""
from __future__ import annotations

import csv
import sys

import torch


def main(pred_path: str, csv_path: str) -> None:
    """Write the predictions in ``pred_path`` to ``csv_path``.

    Args:
        pred_path: ``.pt`` file written by ``glovita_infer --pred_output``.
        csv_path: CSV file to write.
    """
    outputs = torch.load(pred_path, map_location="cpu", weights_only=False)
    probs = outputs["probs"]
    p_open = probs[:, 1] if probs.ndim == 2 else probs
    labels = outputs["labels"].tolist()
    case_ids = outputs.get("case_ids") or [str(index) for index in range(len(labels))]
    with open(csv_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["case_id", "label", "pred", "p_open"])
        for case_id, label, pred, p in zip(case_ids, labels, outputs["preds"].tolist(), p_open.tolist()):
            writer.writerow([case_id, label, pred, f"{p:.6f}"])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
