"""Consistent, editable figure bundles with data and analysis provenance.

Export defaults are deliberately explicit and configurable here. This module
never changes the plotted numerical values or writes into a recording folder
unless the caller selects that folder as the export destination.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import re

import matplotlib as mpl
from matplotlib.colors import to_rgba
from matplotlib.text import Text

EXPORT_DPI = 600
EXPORT_FORMATS = ("pdf", "png", "svg")
PUBLICATION_RC = {"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"}


@contextmanager
def publication_canvas(figure, *, width_mm=183.0, font_size_pt=7.0):
    """Temporarily put a themed figure on white paper, restoring every change.

    Data colors and colormaps remain intact. Only pale, nearly neutral foreground
    artists on dark axes are adapted to dark ink so exports stay legible.
    """
    changes = []

    def change(artist, getter, setter, value):
        previous = getattr(artist, getter)()
        changes.append((artist, setter, previous))
        getattr(artist, setter)(value)

    def pale_neutral(color):
        try:
            r, g, b, _ = to_rgba(color)
            return min(r, g, b) > 0.48 and max(r, g, b) - min(r, g, b) < 0.25
        except (ValueError, TypeError):
            return False

    try:
        if width_mm is not None:
            previous_size = figure.get_size_inches().copy()
            width_inches = float(width_mm) / 25.4
            height_inches = min(170.0 / 25.4, width_inches * previous_size[1] / previous_size[0])
            change(figure, "get_size_inches", "set_size_inches", [width_inches, height_inches])
            for item in figure.findobj(match=Text):
                change(item, "get_fontsize", "set_fontsize", float(font_size_pt))
            for ax in figure.axes:
                for artist in list(ax.lines) + list(ax.spines.values()):
                    width = artist.get_linewidth()
                    change(artist, "get_linewidth", "set_linewidth", min(1.0, max(0.25, width)))
        dark = sum(to_rgba(figure.get_facecolor())[:3]) / 3 < 0.5
        change(figure, "get_facecolor", "set_facecolor", "white")
        if dark:
            for ax in figure.axes:
                change(ax, "get_facecolor", "set_facecolor", "white")
                for spine in ax.spines.values():
                    change(spine, "get_edgecolor", "set_edgecolor", "#222222")
                for line in list(ax.lines) + list(ax.xaxis.get_ticklines()) + list(ax.yaxis.get_ticklines()):
                    if pale_neutral(line.get_color()):
                        change(line, "get_color", "set_color", "#333333")
                    if pale_neutral(line.get_markeredgecolor()):
                        change(line, "get_markeredgecolor", "set_markeredgecolor", "#333333")
            for item in figure.findobj(match=Text):
                if pale_neutral(item.get_color()):
                    change(item, "get_color", "set_color", "#222222")
        with mpl.rc_context(PUBLICATION_RC):
            yield figure
    finally:
        for artist, setter, previous in reversed(changes):
            getattr(artist, setter)(previous)
        if figure.canvas is not None:
            figure.canvas.draw_idle()


def export_figure_bundle(figure, destination, stem="figure", *, tables=(), metadata=None, width_mm=183.0):
    """Save PDF, 600-dpi PNG, editable SVG, supplied CSVs and a JSON manifest.

    ``tables`` contains ``(filename, pandas.DataFrame)`` pairs. Existing bundles
    are never silently replaced: a numerical suffix is selected when needed.
    Returned paths identify every successfully written artifact.
    """
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    base = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(stem)).strip("._") or "figure"
    candidate, version = base, 1
    while any((root / f"{candidate}.{suffix}").exists() for suffix in (*EXPORT_FORMATS, "json")):
        version += 1
        candidate = f"{base}_{version}"
    paths = []
    with publication_canvas(figure, width_mm=width_mm):
        export_size = list(map(float, figure.get_size_inches()))
        for extension in EXPORT_FORMATS:
            path = root / f"{candidate}.{extension}"
            figure.savefig(path, dpi=EXPORT_DPI, bbox_inches=None if width_mm is not None else "tight", pad_inches=0.04,
                           facecolor="white", edgecolor="none")
            paths.append(path)
    for name, frame in tables:
        if frame is None:
            continue
        suffix = Path(str(name)).stem
        path = root / f"{candidate}_{suffix}.csv"
        frame.to_csv(path, index=False)
        paths.append(path)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "figure_size_inches": export_size,
        "target_width_mm": width_mm,
        "text_size_pt": 7.0 if width_mm is not None else "canvas",
        "raster_dpi": EXPORT_DPI,
        "formats": list(EXPORT_FORMATS),
        "files": [p.name for p in paths],
        "analysis": metadata or {},
        "note": "Figure styling does not establish statistical or biological validity.",
    }
    manifest_path = root / f"{candidate}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    paths.append(manifest_path)
    return paths
