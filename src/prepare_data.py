from __future__ import annotations

import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import DATA_PROCESSED, DATA_RAW, FIGURES, RESULTS, RANDOM_STATE, check_raw_files, ensure_dirs, save_json


COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]

NUMERIC_FEATURES = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CATEGORICAL_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


def read_adult() -> pd.DataFrame:
    check_raw_files()
    data = pd.read_csv(DATA_RAW / "adult.data", names=COLUMNS, header=None, na_values="?", skipinitialspace=True)
    test = pd.read_csv(
        DATA_RAW / "adult.test",
        names=COLUMNS,
        header=None,
        skiprows=1,
        na_values="?",
        skipinitialspace=True,
    )
    df = pd.concat([data, test], ignore_index=True)
    for col in COLUMNS:
        if df[col].dtype == "object":
            df[col] = df[col].str.strip()
    df["income"] = df["income"].str.replace(".", "", regex=False)
    return df


def plot_missing(missing: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 5))
    plt.bar(missing["column"], missing["missing_count"], color="#4C78A8")
    plt.xticks(rotation=55, ha="right")
    plt.ylabel("Missing count")
    plt.tight_layout()
    plt.savefig(FIGURES / "missing_values.png", dpi=180)
    plt.close()


def plot_label_distribution(df: pd.DataFrame) -> None:
    counts = df["income"].value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    plt.bar(counts.index, counts.values, color=["#72B7B2", "#E45756"])
    plt.ylabel("Samples")
    plt.title("Income label distribution")
    for idx, value in enumerate(counts.values):
        plt.text(idx, value, str(value), ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(FIGURES / "label_distribution.png", dpi=180)
    plt.close()


def plot_numeric_histograms(df: pd.DataFrame) -> None:
    cols = ["age", "education-num", "hours-per-week"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, col in zip(axes, cols):
        ax.hist(df[col], bins=30, color="#59A14F", edgecolor="white")
        ax.set_title(col)
        ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(FIGURES / "numeric_feature_histograms.png", dpi=180)
    plt.close(fig)


def plot_categorical_distribution(df: pd.DataFrame) -> None:
    cols = ["workclass", "education", "occupation"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, col in zip(axes, cols):
        counts = df[col].value_counts().head(10).sort_values()
        ax.barh(counts.index, counts.values, color="#F28E2B")
        ax.set_title(f"Top categories: {col}")
        ax.set_xlabel("Samples")
    fig.tight_layout()
    fig.savefig(FIGURES / "categorical_feature_distribution.png", dpi=180)
    plt.close(fig)


def save_split(name: str, x: np.ndarray, y: np.ndarray) -> None:
    np.savez_compressed(DATA_PROCESSED / f"{name}.npz", X=x.astype(np.float32), y=y.astype(np.int64))


def main() -> None:
    ensure_dirs()
    df = read_adult()
    raw_rows = len(df)
    missing = (
        df.isna()
        .sum()
        .rename("missing_count")
        .reset_index()
        .rename(columns={"index": "column"})
    )
    missing["missing_ratio"] = missing["missing_count"] / raw_rows
    missing.to_csv(RESULTS / "missing_values.csv", index=False)
    plot_missing(missing)

    duplicates = int(df.duplicated().sum())
    clean_df = df.dropna().copy()
    clean_rows = len(clean_df)
    clean_df["label"] = clean_df["income"].map({"<=50K": 0, ">50K": 1}).astype(int)

    plot_label_distribution(clean_df)
    plot_numeric_histograms(clean_df)
    plot_categorical_distribution(clean_df)

    label_distribution = clean_df["income"].value_counts().rename_axis("income").reset_index(name="count")
    label_distribution["ratio"] = label_distribution["count"] / len(clean_df)
    label_distribution.to_csv(RESULTS / "label_distribution.csv", index=False)

    X = clean_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = clean_df["label"].to_numpy()
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )
    X_train_p = preprocessor.fit_transform(X_train)
    X_val_p = preprocessor.transform(X_val)
    X_test_p = preprocessor.transform(X_test)

    save_split("train", X_train_p, y_train)
    save_split("val", X_val_p, y_val)
    save_split("test", X_test_p, y_test)
    clean_df.to_csv(DATA_PROCESSED / "adult_clean.csv", index=False)
    with (DATA_PROCESSED / "preprocessor.pkl").open("wb") as f:
        pickle.dump(preprocessor, f)

    metadata = {
        "raw_rows": raw_rows,
        "clean_rows": clean_rows,
        "dropped_rows": raw_rows - clean_rows,
        "duplicate_rows_before_cleaning": duplicates,
        "train_rows": int(len(y_train)),
        "val_rows": int(len(y_val)),
        "test_rows": int(len(y_test)),
        "input_dim": int(X_train_p.shape[1]),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "label_mapping": {"<=50K": 0, ">50K": 1},
        "random_state": RANDOM_STATE,
    }
    save_json(RESULTS / "data_summary.json", metadata)
    print(metadata)


if __name__ == "__main__":
    main()
