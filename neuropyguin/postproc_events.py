"""Helpers for inspecting and loading behavioral event CSV files.

Two CSV shapes are supported:

1. **Long-format event tables** - one row per event, with a numeric event-time
   column (seconds) and an optional string label column. Column detection is
   tolerant of varied naming conventions.

2. **Wide binary behavior matrices** - one row per video frame, one column per
   behavior, every cell a 0/1 state (the format produced by DeepLabCut /
   behavior-classifier pipelines). For these, each behavior column is exposed as
   a selectable "label", and event times are derived from 0->1 rising edges
   (bout onsets), 1->0 falling edges (offsets), or bout midpoints. Onset times
   come from the file's own time column when present (so they share the spike
   time base) and otherwise from frame index / frame rate.

The unified entry points are :func:`inspect_event_csv` (reports the detected
shape, time column, and selectable labels) and :func:`load_event_times` (returns
the numeric event times for a chosen label).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
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
    "time_seconds",
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

# Names that hold a frame timestamp / index rather than a behavior state.
_TIME_LIKE_NAMES = {
    "time",
    "time_s",
    "time_seconds",
    "timestamp",
    "timestamp_s",
    "t",
    "sec",
    "seconds",
    "frame",
    "frames",
    "frame_index",
    "index",
}

ALIGNMENT_OPTIONS = ("Onset (rising)", "Offset (falling)", "Bout midpoint")


def _normalized_column_map(df: pd.DataFrame) -> Dict[str, str]:
    """Map each lowercased, stripped column name to its original column name."""
    out: Dict[str, str] = {}
    for col in df.columns:
        key = str(col).strip().lower()
        if key and key not in out:
            out[key] = str(col)
    return out


def _numeric_score(series: pd.Series) -> int:
    """Return the count of values in the series that parse as numbers."""
    try:
        numeric = pd.to_numeric(series, errors="coerce")
    except Exception:
        return 0
    return int(numeric.notna().sum())


def _is_index_like(name: str) -> bool:
    """True for an obvious row-index column (e.g. pandas' 'Unnamed: 0')."""
    low = str(name).strip().lower()
    return low.startswith("unnamed") or low in {"index", "", "frame", "frames", "frame_index"}


def _is_binary_series(series: pd.Series) -> bool:
    """True if the numeric content of a column is restricted to {0, 1}."""
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return False
    uniques = np.unique(numeric.to_numpy().astype(float))
    if uniques.size == 0 or uniques.size > 2:
        return False
    return set(uniques.tolist()).issubset({0.0, 1.0})


def behavior_columns(df: pd.DataFrame) -> List[str]:
    """Return the binary behavior columns of a wide behavior matrix, in order.

    A column qualifies when its numeric content is restricted to {0, 1} and it is
    not an index or timestamp column. Columns that are constant (all 0 or all 1,
    i.e. no transitions) are still reported so the user can see them, but they
    yield no events.
    """
    cols: List[str] = []
    for col in df.columns:
        if _is_index_like(col):
            continue
        if str(col).strip().lower() in _TIME_LIKE_NAMES:
            continue
        if _is_binary_series(df[col]):
            cols.append(str(col))
    return cols


def is_behavior_matrix(df: pd.DataFrame) -> bool:
    """True when the CSV looks like a wide binary behavior matrix (>= 2 0/1 columns)."""
    return len(behavior_columns(df)) >= 2


def matrix_time_column(df: pd.DataFrame) -> Optional[str]:
    """Return the per-frame time column of a behavior matrix (seconds), or None.

    Prefers an explicit seconds column ('time', 'time_seconds', ...). Returns
    None when no monotonic numeric time column is present (the caller then falls
    back to frame index / frame rate).
    """
    normalized = _normalized_column_map(df)
    for name in ("time", "time_seconds", "timestamp_s", "timestamp", "time_s", "t"):
        actual = normalized.get(name)
        if actual and _numeric_score(df[actual]) > 1:
            return actual
    return None


def detect_frame_rate(df: pd.DataFrame, default: float = 30.0) -> float:
    """Infer the frame rate (Hz) from the median spacing of the time column."""
    tcol = matrix_time_column(df)
    if not tcol:
        return float(default)
    t = pd.to_numeric(df[tcol], errors="coerce").dropna().to_numpy()
    if t.size < 3:
        return float(default)
    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0:
        return float(default)
    return float(1.0 / dt)


def behavior_onset_times(
    df: pd.DataFrame,
    column: str,
    *,
    frame_rate: Optional[float] = None,
    time_column: Optional[str] = None,
    alignment: str = "Onset (rising)",
    min_bout_s: float = 0.0,
) -> np.ndarray:
    """Return event times (s) for one behavior column of a wide binary matrix.

    Detects bouts as runs of 1s. ``alignment`` selects the rising edge (bout
    onset), the falling edge (bout offset), or the bout midpoint. ``min_bout_s``
    drops bouts shorter than the given duration. Times come from the file's time
    column when available (sharing the spike time base) and otherwise from frame
    index divided by ``frame_rate``.
    """
    v = (pd.to_numeric(df[column], errors="coerce").fillna(0).to_numpy() > 0.5).astype(np.int8)
    if v.size == 0:
        return np.array([], dtype=float)

    # Bout start/stop frame indices (stop is exclusive, i.e. first 0 after the run).
    padded = np.concatenate(([0], v, [0]))
    diff = np.diff(padded)
    starts = np.flatnonzero(diff == 1)
    stops = np.flatnonzero(diff == -1)
    if starts.size == 0:
        return np.array([], dtype=float)

    tcol = time_column or matrix_time_column(df)
    if tcol and tcol in df.columns:
        t = pd.to_numeric(df[tcol], errors="coerce").to_numpy(dtype=float)
        fps = detect_frame_rate(df, default=float(frame_rate or 30.0))

        def frame_time(idx: int) -> float:
            idx = int(min(max(idx, 0), t.size - 1))
            return float(t[idx])
    else:
        fps = float(frame_rate or 30.0)

        def frame_time(idx: int) -> float:
            return float(idx) / fps

    if min_bout_s > 0.0:
        durations = (stops - starts) / max(fps, 1e-9)
        keep = durations >= float(min_bout_s)
        starts, stops = starts[keep], stops[keep]
        if starts.size == 0:
            return np.array([], dtype=float)

    align = str(alignment or "").strip().lower()
    if align.startswith("offset"):
        idx = np.clip(stops - 1, 0, v.size - 1)
    elif align.startswith("bout"):
        idx = np.clip((starts + (stops - 1)) // 2, 0, v.size - 1)
    else:
        idx = starts
    return np.array([frame_time(int(i)) for i in idx], dtype=float)


def detect_event_time_column(df: pd.DataFrame) -> Optional[str]:
    """Detect the column most likely to contain long-format event times."""
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
    return numeric_cols[0] if numeric_cols else None


def detect_event_label_column(df: pd.DataFrame, time_column: Optional[str] = None) -> Optional[str]:
    """Detect the column most likely to contain long-format event labels."""
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
    """Return the distinct, non-empty label values in order of first appearance."""
    if not label_column or label_column not in df.columns:
        return []
    values = [str(v).strip() for v in df[label_column].dropna().tolist()]
    return [v for v in dict.fromkeys(values) if v]


def inspect_event_csv(path: str | Path) -> Dict[str, object]:
    """Read an event CSV and report its shape, time column, and selectable labels.

    Returns a dict with: ``path``, ``dataframe``, ``kind`` ("behavior_matrix" or
    "events"), ``time_column``, ``label_column`` (None for matrices),
    ``labels`` (behavior columns for matrices, label values for event tables),
    and ``frame_rate`` (for matrices).
    """
    csv_path = Path(path)
    df = pd.read_csv(csv_path)

    if is_behavior_matrix(df):
        beh_cols = behavior_columns(df)
        tcol = matrix_time_column(df)
        return {
            "path": str(csv_path),
            "dataframe": df,
            "kind": "behavior_matrix",
            "time_column": tcol,
            "label_column": None,
            "labels": beh_cols,
            "frame_rate": detect_frame_rate(df),
        }

    time_column = detect_event_time_column(df)
    label_column = detect_event_label_column(df, time_column=time_column)
    labels = event_label_values(df, label_column)
    return {
        "path": str(csv_path),
        "dataframe": df,
        "kind": "events",
        "time_column": time_column,
        "label_column": label_column,
        "labels": labels,
        "frame_rate": None,
    }


def load_event_times(
    path: str | Path,
    *,
    selected_label: Optional[str] = None,
    frame_rate: Optional[float] = None,
    alignment: str = "Onset (rising)",
    min_bout_s: float = 0.0,
) -> pd.Series:
    """Load event times (s) from a CSV, branching on the detected shape.

    - Behavior matrix: ``selected_label`` names a behavior column; its bout
      events (per ``alignment``) are returned. An empty/"all" label pools the
      onsets of every behavior column.
    - Long-format table: rows are optionally filtered by ``selected_label``
      against the detected label column, and the numeric time column is returned.

    Returns an empty float Series when no usable times are found.
    """
    info = inspect_event_csv(path)
    df = info["dataframe"]
    label_value = str(selected_label or "").strip()

    if info.get("kind") == "behavior_matrix":
        cols = [str(c) for c in info.get("labels", [])]
        if label_value and label_value in cols:
            chosen = [label_value]
        elif label_value and label_value.lower() not in {"all events", "all", ""}:
            return pd.Series(dtype=float)
        else:
            chosen = cols  # pool all behaviors
        all_times: List[np.ndarray] = []
        for col in chosen:
            all_times.append(
                behavior_onset_times(
                    df,
                    col,
                    frame_rate=frame_rate,
                    time_column=str(info.get("time_column") or "") or None,
                    alignment=alignment,
                    min_bout_s=min_bout_s,
                )
            )
        if not all_times:
            return pd.Series(dtype=float)
        merged = np.sort(np.concatenate(all_times)) if all_times else np.array([], dtype=float)
        return pd.Series(merged, dtype=float)

    time_column = str(info.get("time_column") or "")
    label_column = str(info.get("label_column") or "")
    if not time_column or time_column not in df.columns:
        return pd.Series(dtype=float)
    source = df
    if label_value and label_column and label_column in source.columns:
        source = source[source[label_column].astype(str).str.strip() == label_value]
    return pd.to_numeric(source[time_column], errors="coerce").dropna()
