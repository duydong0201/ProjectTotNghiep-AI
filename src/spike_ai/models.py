"""Danh sách model có thể train. Thêm model mới: viết thêm 1 hàm vào REGISTRY."""

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def _majority(seed, **params):
    # Baseline "ngu": luôn đoán nhãn phổ biến nhất. Model nào không hơn nó là vô dụng.
    return DummyClassifier(strategy="most_frequent", **params)


def _decision_tree(seed, **params):
    return DecisionTreeClassifier(random_state=seed, **params)


def _random_forest(seed, **params):
    return RandomForestClassifier(random_state=seed, n_jobs=-1, **params)


def _logreg(seed, **params):
    return make_pipeline(StandardScaler(), LogisticRegression(random_state=seed, max_iter=2000, **params))


def _knn(seed, **params):
    return make_pipeline(StandardScaler(), KNeighborsClassifier(**params))


REGISTRY = {
    "majority": _majority,
    "decision_tree": _decision_tree,
    "random_forest": _random_forest,
    "logreg": _logreg,
    "knn": _knn,
}

# Model export được sang header C++ (xem export_cpp.py)
EXPORTABLE = {"decision_tree", "random_forest"}


def create(name: str, seed: int, params: dict | None = None):
    if name not in REGISTRY:
        raise ValueError(f"Model '{name}' chưa có. Có: {list(REGISTRY)}")
    return REGISTRY[name](seed, **(params or {}))
