from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from common import DATA_PROCESSED, RESULTS, RANDOM_STATE, classification_metrics, ensure_dirs, set_seed


def load_split(name: str) -> tuple[np.ndarray, np.ndarray]:
    path = DATA_PROCESSED / f"{name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run: python src/prepare_data.py")
    data = np.load(path)
    return data["X"], data["y"]


def main() -> None:
    ensure_dirs()
    set_seed()
    X_train, y_train = load_split("train")
    X_test, y_test = load_split("test")

    rows = []
    majority_class = int(pd.Series(y_train).mode()[0])
    majority_pred = np.full_like(y_test, majority_class)
    rows.append({"model": "Majority Class", **classification_metrics(y_test, majority_pred)})

    prior = np.bincount(y_train) / len(y_train)
    rng = np.random.default_rng(RANDOM_STATE)
    random_pred = rng.choice([0, 1], size=len(y_test), p=prior)
    rows.append({"model": "Random Prior", **classification_metrics(y_test, random_pred)})

    clf = LogisticRegression(max_iter=1000, class_weight=None, random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)
    logistic_pred = clf.predict(X_test)
    rows.append({"model": "Logistic Regression", **classification_metrics(y_test, logistic_pred)})

    result = pd.DataFrame(rows)
    result.to_csv(RESULTS / "baseline_results.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
