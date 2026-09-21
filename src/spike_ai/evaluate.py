"""Chỉ số đánh giá offline.

Dùng macro-F1 thay vì accuracy: dữ liệu có rất nhiều frame 'None', model đoán toàn
'None' vẫn có accuracy cao nhưng macro-F1 thấp.
"""

from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def compute_metrics(y_true, y_pred, labels) -> dict:
    labels = list(labels)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "labels": [str(l) for l in labels],
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "report": classification_report(y_true, y_pred, labels=labels, zero_division=0, output_dict=True),
    }


def plot_confusion_matrix(metrics: dict, title: str, out_path: Path) -> None:
    """Lưu confusion matrix (chuẩn hoá theo hàng) thành ảnh PNG cho báo cáo."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.array(metrics["confusion_matrix"], dtype=float)
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sum, out=np.zeros_like(cm), where=row_sum > 0)
    labels = metrics["labels"]

    fig, ax = plt.subplots(figsize=(1.0 + 0.6 * len(labels), 0.8 + 0.6 * len(labels)))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Thực tế")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j] > 0:
                ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if cm_norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
