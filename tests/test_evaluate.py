"""Test cách tính chỉ số: macro-F1 chỉ trên lớp có mặt, support, lớp hiếm, gộp fold."""

import pytest

from spike_ai.evaluate import MIN_SUPPORT, compute_metrics, summarize_folds

ORDER = ["None", "Serve", "SpikeLight", "SpikeStrong"]


def test_absent_class_does_not_drag_macro_f1_down():
    """Regression: lớp vắng mặt trong tập đo từng bị tính F1 = 0 (0.33 thay vì 0.495)."""
    y_true = ["None"] * 98 + ["Serve"] * 2
    y_pred = ["None"] * 100
    m = compute_metrics(y_true, y_pred, ORDER)
    assert m["measured_labels"] == ["None", "Serve"]
    # F1(None) = 2 * 0.98 * 1 / (0.98 + 1), F1(Serve) = 0; hai lớp SpikeLight/SpikeStrong vắng mặt không tính
    f1_none = 2 * 0.98 / 1.98
    assert m["macro_f1"] == pytest.approx((f1_none + 0.0) / 2)


def test_predicting_an_absent_class_is_still_penalised():
    y_true = ["None"] * 10
    good = compute_metrics(y_true, ["None"] * 10, ORDER)
    bad = compute_metrics(y_true, ["None"] * 5 + ["SpikeStrong"] * 5, ORDER)
    assert good["macro_f1"] == 1.0
    assert bad["macro_f1"] < 1.0
    # lớp bị đoán nhầm vẫn hiện trong confusion matrix để thấy lỗi
    assert "SpikeStrong" in bad["labels"] and "SpikeStrong" not in bad["measured_labels"]


def test_support_and_rare_labels():
    y = ["None"] * 50 + ["Serve"] * (MIN_SUPPORT - 1) + ["SpikeLight"] * MIN_SUPPORT
    m = compute_metrics(y, y, ORDER)
    assert m["support"] == {"None": 50, "Serve": MIN_SUPPORT - 1, "SpikeLight": MIN_SUPPORT}
    assert m["rare_labels"] == ["Serve"]


def test_labels_follow_schema_order_for_int_targets():
    m = compute_metrics([1, 0, -1, 1], [1, 0, 0, 1], [-1, 0, 1])
    assert m["labels"] == ["-1", "0", "1"]
    assert len(m["confusion_matrix"]) == 3


def test_summarize_folds():
    folds = [{"accuracy": 0.8, "macro_f1": 0.4}, {"accuracy": 0.6, "macro_f1": 0.6}]
    s = summarize_folds(folds)
    assert s["macro_f1"] == pytest.approx(0.5)
    assert s["macro_f1_std"] == pytest.approx(0.1)
    assert s["n_folds"] == 2
