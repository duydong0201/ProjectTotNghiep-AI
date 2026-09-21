"""Đọc schema JSON - nguồn sự thật về cột log, enum và thứ tự feature."""

import json
from functools import lru_cache

import pandas as pd

from .paths import SCHEMA_DIR

SCHEMA_FILES = {
    "v0": "legacy_v0.json",
    "v1": "feature_spec.v1.json",
}


@lru_cache(maxsize=None)
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
    """Nhận diện version của một file CSV dựa vào header."""
    columns = list(columns)
    if columns == raw_column_names(load_spec("v0")):
        return "v0"
    if "schema_version" in columns:
        return "v1"
    raise ValueError(f"Không nhận diện được format log. Header: {columns}")


def validate_columns(df: pd.DataFrame, version: str) -> None:
    """Báo lỗi sớm nếu file log thiếu cột so với spec."""
    missing = [c for c in raw_column_names(load_spec(version)) if c not in df.columns]
    if missing:
        raise ValueError(f"Log thiếu cột theo schema {version}: {missing}")
    if version == "v1":
        versions = set(df["schema_version"].unique())
        if versions != {1}:
            raise ValueError(f"schema_version không hợp lệ: {versions}")
