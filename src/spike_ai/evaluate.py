"""Chỉ số đánh giá offline.

Dùng macro-F1 thay vì accuracy: dữ liệu có rất nhiều frame 'None', model đoán toàn
'None' vẫn có accuracy cao nhưng macro-F1 thấp.

macro-F1 chỉ lấy trung bình trên các lớp CÓ MẶT trong tập đo (support > 0). Một lớp vắng mặt
không thể đo được: nếu vẫn tính nó với F1 = 0 thì điểm bị kéo xuống dù model không sai gì.
Model đoán ra một lớp vắng mặt vẫn bị phạt, vì dự đoán đó làm giảm recall của lớp thật.
"""

from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

# Lớp có ít mẫu hơn ngưỡng này trong tập đo thì F1 của nó rất nhiễu -> cảnh báo
MIN_SUPPORT = 10


def compute_metrics(y_true, y_pred, label_order) -> dict:
    """label_order: thứ tự lớp chuẩn (lấy từ schema), dùng để sắp xếp kết quả."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    true_set, pred_set = set(y_true.tolist()), set(y_pred.tolist())

    measured = [c for c in label_order if c in true_set]  # lớp tính vào macro-F1
    shown = [c for c in label_order if c in true_set | pred_set]  # lớp hiện trong confusion matrix
    support = {str(c): int((y_true == c).sum()) for c in measured}

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=measured, average="macro", zero_division=0)),
        "measured_labels": [str(c) for c in measured],
        "support": support,
        "rare_labels": [c for c, n in support.items() if n < MIN_SUPPORT],
        "labels": [str(c) for c in shown],
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=shown).tolist(),
        "report": classification_report(y_true, y_pred, labels=measured, zero_division=0, output_dict=True),
    }


def summarize_folds(fold_metrics: list[dict]) -> dict:
    """Gộp kết quả nhiều fold (cross-validation): trung bình ± độ lệch chuẩn."""
    acc = np.array([m["accuracy"] for m in fold_metrics])
    f1 = np.array([m["macro_f1"] for m in fold_metrics])
    return {
        "accuracy": float(acc.mean()),
        "accuracy_std": float(acc.std()),
        "macro_f1": float(f1.mean()),
        "macro_f1_std": float(f1.std()),
        "n_folds": len(fold_metrics),
        "folds": fold_metrics,
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
                ax.text(
                    j,
                    i,
                    f"{cm_norm[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if cm_norm[i, j] > 0.5 else "black",
                )
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
