from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from common import FIGURES, RESULTS, ensure_dirs
from train_mlp import train_model


def main() -> None:
    ensure_dirs()
    rows = []
    for lr in [0.1, 0.01, 0.001, 0.0001]:
        log, test_metrics, _ = train_model(epochs=25, lr=lr, dropout=0.3, save_prefix=f"hparam_lr_{lr:g}", show_progress=True)
        best = log.sort_values("val_f1_macro", ascending=False).iloc[0]
        rows.append(
            {
                "parameter": "learning_rate",
                "value": lr,
                "best_epoch": int(best["epoch"]),
                "val_accuracy": best["val_accuracy"],
                "val_f1_macro": best["val_f1_macro"],
                "test_accuracy": test_metrics["accuracy"],
                "test_f1_macro": test_metrics["f1_macro"],
            }
        )

    for dropout in [0.1, 0.3, 0.5]:
        log, test_metrics, _ = train_model(
            epochs=25, lr=0.001, dropout=dropout, save_prefix=f"hparam_dropout_{dropout:g}", show_progress=True
        )
        best = log.sort_values("val_f1_macro", ascending=False).iloc[0]
        rows.append(
            {
                "parameter": "dropout",
                "value": dropout,
                "best_epoch": int(best["epoch"]),
                "val_accuracy": best["val_accuracy"],
                "val_f1_macro": best["val_f1_macro"],
                "test_accuracy": test_metrics["accuracy"],
                "test_f1_macro": test_metrics["f1_macro"],
            }
        )

    result = pd.DataFrame(rows)
    result["is_best_within_parameter"] = result.groupby("parameter")["val_f1_macro"].transform("max").eq(result["val_f1_macro"])
    result.to_csv(RESULTS / "hparam_results.csv", index=False)

    for parameter, figure_name in [("learning_rate", "hparam_lr.png"), ("dropout", "hparam_dropout.png")]:
        subset = result[result["parameter"] == parameter].sort_values("value")
        plt.figure(figsize=(7, 4))
        plt.plot(subset["value"], subset["val_f1_macro"], marker="o", label="Validation Macro-F1")
        plt.plot(subset["value"], subset["test_f1_macro"], marker="s", label="Test Macro-F1")
        if parameter == "learning_rate":
            plt.xscale("log")
        plt.xlabel(parameter)
        plt.ylabel("Macro-F1")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURES / figure_name, dpi=180)
        plt.close()

    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
