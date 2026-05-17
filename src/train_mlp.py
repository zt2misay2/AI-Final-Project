from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from common import (
    DATA_PROCESSED,
    FIGURES,
    MODELS,
    RESULTS,
    classification_metrics,
    confusion_as_frame,
    ensure_dirs,
    set_seed,
)


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden: tuple[int, ...] = (128, 64), dropout: float = 0.3) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = input_dim
        for width in hidden:
            layers.extend([nn.Linear(current, width), nn.ReLU(), nn.Dropout(dropout)])
            current = width
        layers.append(nn.Linear(current, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_split(name: str) -> tuple[np.ndarray, np.ndarray]:
    path = DATA_PROCESSED / f"{name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run: python src/prepare_data.py")
    data = np.load(path)
    return data["X"], data["y"]


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer | None) -> dict:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    preds = []
    labels = []
    for xb, yb in loader:
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        if is_train:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.item()) * len(yb)
        preds.append(logits.argmax(dim=1).detach().numpy())
        labels.append(yb.numpy())
    y_true = np.concatenate(labels)
    y_pred = np.concatenate(preds)
    return {"loss": total_loss / len(y_true), **classification_metrics(y_true, y_pred)}


def train_model(
    epochs: int = 50,
    lr: float = 0.001,
    dropout: float = 0.3,
    batch_size: int = 64,
    save_prefix: str = "mlp",
    show_progress: bool = True,
) -> tuple[pd.DataFrame, dict, np.ndarray]:
    set_seed()
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    X_test, y_test = load_split("test")

    train_loader = make_loader(X_train, y_train, batch_size, True)
    val_loader = make_loader(X_val, y_val, batch_size, False)
    test_loader = make_loader(X_test, y_test, batch_size, False)

    model = MLP(input_dim=X_train.shape[1], dropout=dropout)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_state = None
    best_f1 = -1.0
    logs = []
    iterator = range(1, epochs + 1)
    if show_progress:
        iterator = tqdm(iterator, desc=f"{save_prefix} lr={lr:g} dropout={dropout:g}")
    for epoch in iterator:
        train_metrics = run_epoch(model, train_loader, criterion, optimizer)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, criterion, None)
        row = {"epoch": epoch}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        logs.append(row)
        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    torch.save(
        {"model_state_dict": model.state_dict(), "input_dim": X_train.shape[1], "dropout": dropout, "lr": lr},
        MODELS / f"{save_prefix}_best.pt",
    )

    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for xb, yb in test_loader:
            logits = model(xb)
            all_preds.append(logits.argmax(dim=1).numpy())
            all_labels.append(yb.numpy())
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    test_metrics = classification_metrics(y_true, y_pred)
    cm = confusion_as_frame(y_true, y_pred)
    return pd.DataFrame(logs), test_metrics, cm


def plot_training(log: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(log["epoch"], log["train_loss"], label="Train")
    axes[0].plot(log["epoch"], log["val_loss"], label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[1].plot(log["epoch"], log["train_accuracy"], label="Train")
    axes[1].plot(log["epoch"], log["val_accuracy"], label="Validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "mlp_training_curves.png", dpi=180)
    plt.close(fig)


def plot_confusion(cm: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["<=50K", ">50K"])
    ax.set_yticks([0, 1], labels=["<=50K", ">50K"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(FIGURES / "mlp_confusion_matrix.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    ensure_dirs()
    log, test_metrics, cm = train_model(args.epochs, args.lr, args.dropout, args.batch_size, "mlp", True)
    log.to_csv(RESULTS / "mlp_training_log.csv", index=False)
    pd.DataFrame([{**{"model": "MLP"}, **test_metrics}]).to_csv(RESULTS / "mlp_test_metrics.csv", index=False)
    pd.DataFrame(cm, index=["true_<=50K", "true_>50K"], columns=["pred_<=50K", "pred_>50K"]).to_csv(
        RESULTS / "mlp_confusion_matrix.csv"
    )
    plot_training(log)
    plot_confusion(cm)
    print(pd.DataFrame([{**{"model": "MLP"}, **test_metrics}]).to_string(index=False))


if __name__ == "__main__":
    main()
