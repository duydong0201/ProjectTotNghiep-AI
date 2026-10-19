"""Đọc schema JSON - nguồn sự thật về cột log, enum và thứ tự feature."""

import json
from functools import cache

import pandas as pd

from .paths import SCHEMA_DIR

SCHEMA_FILES = {
    "v0": "legacy_v0.json",
    "v1": "feature_spec.v1.json",
    "v2": "feature_spec.v2.json",
}


@cache
def load_spec(version: str) -> dict:
    if version not in SCHEMA_FILES:
        raise ValueError(f"Schema '{version}' không tồn tại. Có: {list(SCHEMA_FILES)}")
    with open(SCHEMA_DIR / SCHEMA_FILES[version], encoding="utf-8") as f:
        return json.load(f)


def raw_column_names(spec: dict) -> list[str]:
    cols = spec["raw_columns"]
    return [c if isinstance(c, str) else c["name"] for c in cols]


def feature_names(spec: dict) -> list[str]:
    return [f["name"] for f in spec["features"]]


def detect_version(columns) -> str:
    """Nhận diện version của một file CSV dựa vào header.

    Chỉ nhìn TÊN cột, không nhìn giá trị, nên v2 được nhận ra bằng cột riêng của nó
    (`rally_last_touch_kind`) chứ không bằng giá trị schema_version. Giá trị đó vẫn được
    kiểm lại trong validate_columns.
    """
    columns = list(columns)
    if columns == raw_column_names(load_spec("v0")):
        return "v0"
    if "schema_version" not in columns:
        raise ValueError(f"Không nhận diện được format log. Header: {columns}")
    # v2 = v1 + cột rally_last_touch_kind (thêm ở cuối, xem DatasetLogger.cpp)
    return "v2" if "rally_last_touch_kind" in columns else "v1"


def validate_columns(df: pd.DataFrame, version: str) -> None:
    """Báo lỗi sớm nếu file log thiếu cột so với spec."""
    missing = [c for c in raw_column_names(load_spec(version)) if c not in df.columns]
    if missing:
        raise ValueError(f"Log thiếu cột theo schema {version}: {missing}")
    expected_value = {"v1": 1, "v2": 2}.get(version)
    if expected_value is not None:
        versions = set(df["schema_version"].unique())
        if versions != {expected_value}:
            raise ValueError(f"schema_version không hợp lệ cho {version}: {versions}, mong đợi {{{expected_value}}}")
