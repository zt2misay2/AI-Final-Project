from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
RANDOM_STATE = 42


def ensure_dirs() -> None:
    for path in [DATA_RAW, DATA_PROCESSED, FIGURES, RESULTS, MODELS]:
        path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def required_raw_files() -> list[Path]:
    return [DATA_RAW / "adult.data", DATA_RAW / "adult.test", DATA_RAW / "adult.names"]


def check_raw_files() -> None:
    missing = [path for path in required_raw_files() if not path.exists()]
    if missing:
        expected = "\n".join(str(path) for path in required_raw_files())
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Missing UCI Adult raw files:\n"
            f"{missing_text}\n\nPlace adult.data, adult.test, and adult.names under:\n{DATA_RAW}\n\n"
            "Official source: https://archive.ics.uci.edu/ml/machine-learning-databases/adult/\n"
            f"Expected paths:\n{expected}"
        )


def save_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
    }


def confusion_as_frame(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=[0, 1])
