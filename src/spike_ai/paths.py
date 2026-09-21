"""Đường dẫn chuẩn của project, dùng chung cho mọi module."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SCHEMA_DIR = ROOT / "schema"
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
EXPORTS_DIR = ROOT / "exports"
REPORTS_DIR = ROOT / "reports"


def resolve(path: str | Path) -> Path:
    """Đường dẫn tương đối được hiểu là tương đối với thư mục gốc project."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p
