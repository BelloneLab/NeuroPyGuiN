from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


_TIME_COLUMN_PRIORITY = (
    "time_s",
    "event_time_s",
    "event_time",
    "cue_onset_s",
    "onset_s",
    "onset",
    "timestamp_s",
    "timestamp",
    "time",
)

_LABEL_COLUMN_PRIORITY = (
    "event_type",
    "event",
    "label",
    "condition",
    "trial_type",
    "type",
    "name",
)


def _normalized_column_map(df: pd.DataFrame) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for col in df.columns:
        key = str(col).strip().lower()
        if key and key not in out:
            out[key] = str(col)
    return out


def _numeric_score(series: pd.Series) -> int:
    try:
        numeric = pd.to_numeric(series, errors="coerce")
    except Exception:
        return 0
    return int(numeric.notna().sum())


def detect_event_time_column(df: pd.DataFrame) -> Optional[str]:
    if df is None or df.empty or df.shape[1] == 0:
        return None
    normalized = _normalized_column_map(df)
    for name in _TIME_COLUMN_PRIORITY:
        actual = normalized.get(name)
        if actual and _numeric_score(df[actual]) > 0:
            return actual

    keyword_hits: List[str] = []
    for col in df.columns:
        lower = str(col).strip().lower()
        if any(token in lower for token in ("time", "onset", "timestamp")) and _numeric_score(df[col]) > 0:
            keyword_hits.append(str(col))
    if keyword_hits:
        return keyword_hits[0]

    numeric_cols = [str(col) for col in df.columns if _numeric_score(df[col]) > 0]
    if len(numeric_cols) == 1:
        return numeric_cols[0]
    return numeric_cols[0] if numeric_cols else None


def detect_event_label_column(df: pd.DataFrame, time_column: Optional[str] = None) -> Optional[str]:
    if df is None or df.empty or df.shape[1] == 0:
        return None

    normalized = _normalized_column_map(df)

    def valid_label_column(column: str) -> bool:
        if time_column is not None and str(column) == str(time_column):
            return False
        series = df[column].dropna()
        if series.empty:
            return False
        if pd.api.types.is_numeric_dtype(series):
            return False
        labels = [str(v).strip() for v in series.tolist() if str(v).strip()]
        if not labels:
            return False
        return len(dict.fromkeys(labels)) > 1

    for name in _LABEL_COLUMN_PRIORITY:
        actual = normalized.get(name)
        if actual and valid_label_column(actual):
            return actual

    for col in df.columns:
        actual = str(col)
        if valid_label_column(actual):
            return actual
    return None


def event_label_values(df: pd.DataFrame, label_column: Optional[str]) -> List[str]:
    if not label_column or label_column not in df.columns:
        return []
    values = [str(v).strip() for v in df[label_column].dropna().tolist()]
    return [v for v in dict.fromkeys(values) if v]


def inspect_event_csv(path: str | Path) -> Dict[str, object]:
    csv_path = Path(path)
    df = pd.read_csv(csv_path)
    time_column = detect_event_time_column(df)
    label_column = detect_event_label_column(df, time_column=time_column)
    labels = event_label_values(df, label_column)
    return {
        "path": str(csv_path),
        "dataframe": df,
        "time_column": time_column,
        "label_column": label_column,
        "labels": labels,
    }


def load_event_times(
    path: str | Path,
    *,
    selected_label: Optional[str] = None,
) -> pd.Series:
    info = inspect_event_csv(path)
    df = info["dataframe"]
    time_column = str(info.get("time_column") or "")
    label_column = str(info.get("label_column") or "")
    if not time_column or time_column not in df.columns:
        return pd.Series(dtype=float)
    source = df
    label_value = str(selected_label or "").strip()
    if label_value and label_column and label_column in source.columns:
        source = source[source[label_column].astype(str).str.strip() == label_value]
    return pd.to_numeric(source[time_column], errors="coerce").dropna()
