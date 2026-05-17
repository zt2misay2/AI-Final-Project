from __future__ import annotations

from pathlib import Path

import pandas as pd

from common import RESULTS
from train_mlp import plot_confusion, plot_training


def main() -> None:
    log_path = RESULTS / "mlp_training_log.csv"
    cm_path = RESULTS / "mlp_confusion_matrix.csv"
    if log_path.exists():
        plot_training(pd.read_csv(log_path))
    if cm_path.exists():
        cm = pd.read_csv(cm_path, index_col=0).to_numpy()
        plot_confusion(cm)

    produced = sorted(str(path) for path in Path("figures").glob("*.png"))
    print("Figures:")
    for path in produced:
        print(path)


if __name__ == "__main__":
    main()
