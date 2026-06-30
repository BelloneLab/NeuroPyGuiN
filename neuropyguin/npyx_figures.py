"""Embed real NeuroPyxels (npyx) matplotlib figures inside the Qt GUI.

The post-processing panel's correlogram and waveform views render figures using
npyx's own plotting functions (``npyx.plot.plot_acg`` / ``plot_ccg`` /
``plot_wvf``) so the output is pixel-faithful to what a user gets calling npyx
directly, rather than a pyqtgraph look-alike. Figures are produced with the
non-interactive Agg backend (no windows pop) and wrapped in a
``FigureCanvasQTAgg`` for display.

npyx reads the Kilosort/phy folder directly as its datapath (``dp``); correlograms
are cached under ``<dp>/.NeuroPyxels/`` on first use.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

os.environ.setdefault("QT_API", "pyside6")

import matplotlib

matplotlib.use("Agg")  # figure creation only; we reparent into a Qt canvas
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: E402
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT  # noqa: E402

from PySide6 import QtCore, QtWidgets  # noqa: E402


def _ensure_npyx_plot():
    """Import and return ``npyx.plot`` (vendored copy is on sys.path via the bridge)."""
    import sys

    app_root = Path(__file__).resolve().parents[1]
    if (app_root / "npyx").exists() and str(app_root) not in sys.path:
        sys.path.insert(0, str(app_root))
    import npyx.plot as nplt  # type: ignore

    return nplt


def _grid_shape(n: int) -> tuple[int, int]:
    if n <= 1:
        return 1, 1
    cols = int(math.ceil(math.sqrt(float(n))))
    rows = int(math.ceil(float(n) / max(cols, 1)))
    return rows, cols


def _apply_theme(fig, dark: bool) -> None:
    """Tint a finished npyx figure to the app theme (npyx draws light by default)."""
    if not dark:
        fig.patch.set_facecolor("white")
        return
    bg = "#0b0f14"
    fg = "#e8eef7"
    fig.patch.set_facecolor(bg)
    for ax in fig.get_axes():
        ax.set_facecolor(bg)
        for spine in ax.spines.values():
            spine.set_color(fg)
        ax.tick_params(colors=fg, which="both")
        ax.xaxis.label.set_color(fg)
        ax.yaxis.label.set_color(fg)
        ax.title.set_color(fg)
        leg = ax.get_legend()
        if leg is not None:
            for text in leg.get_texts():
                text.set_color(fg)


class NpyxFigureView(QtWidgets.QWidget):
    """A Qt widget that displays a single npyx matplotlib figure with a toolbar.

    Call :meth:`show_figure` with a freshly built Figure; the previous figure is
    closed to keep memory bounded.
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._layout = QtWidgets.QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)
        self._canvas: Optional[FigureCanvasQTAgg] = None
        self._toolbar: Optional[NavigationToolbar2QT] = None
        self._placeholder = QtWidgets.QLabel(
            "Select unit(s) to render the NeuroPyxels figure."
        )
        self._placeholder.setAlignment(QtCore.Qt.AlignCenter)
        self._placeholder.setWordWrap(True)
        self._placeholder.setObjectName("npyxPlaceholder")
        self._layout.addWidget(self._placeholder)

    def show_message(self, text: str) -> None:
        self._clear()
        self._placeholder.setText(text)
        self._placeholder.setVisible(True)

    def show_figure(self, fig) -> None:
        self._clear()
        self._placeholder.setVisible(False)
        self._canvas = FigureCanvasQTAgg(fig)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        self._layout.addWidget(self._toolbar)
        self._layout.addWidget(self._canvas, 1)
        self._canvas.draw_idle()

    def _clear(self) -> None:
        if self._toolbar is not None:
            self._layout.removeWidget(self._toolbar)
            self._toolbar.setParent(None)
            self._toolbar.deleteLater()
            self._toolbar = None
        if self._canvas is not None:
            self._layout.removeWidget(self._canvas)
            fig = self._canvas.figure
            self._canvas.setParent(None)
            self._canvas.deleteLater()
            self._canvas = None
            try:
                plt.close(fig)
            except Exception:
                pass


# Colourblind-safe palette (Okabe-Ito) so each unit reads as a distinct colour.
OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]

_ACG_YLABEL = {
    "Hertz": "Autocorrelation (spk/s)",
    "Counts": "Autocorrelation (count)",
    "Pearson": "Autocorrelation (Pearson)",
    "zscore": "Autocorrelation (z-score)",
}


def _despine(ax) -> None:
    """Drop the top/right frame for a clean publication look."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _recolor_acg_axis(ax, color: str) -> None:
    """Tint an ACG panel's bars/curve to ``color`` while leaving dashed guide lines grey."""
    for patch in ax.patches:
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
    for line in ax.lines:
        style = line.get_linestyle()
        if style in ("-", "solid"):
            line.set_color(color)
        else:
            # Drop npyx's dashed refractory / zero-lag guide lines (user preference).
            line.set_visible(False)


def acg_grid_figure(
    dp: str,
    units: Sequence[int],
    *,
    cbin: float = 0.5,
    cwin: float = 100.0,
    fs: float = 30000.0,
    normalize: str = "Hertz",
    dark: bool = False,
):
    """Build an npyx ACG grid (one ``plot_acg`` panel per unit, one colour per unit)."""
    nplt = _ensure_npyx_plot()
    plt.close("all")
    units = [int(u) for u in units]
    if not units:
        raise ValueError("Select at least one unit for the ACG.")
    rows, cols = _grid_shape(len(units))
    fig, axes = plt.subplots(
        rows, cols, figsize=(4.2 * cols, 3.0 * rows), squeeze=False, constrained_layout=True
    )
    flat = axes.flatten()
    for i, u in enumerate(units):
        color = OKABE_ITO[i % len(OKABE_ITO)]
        try:
            nplt.plot_acg(
                dp, u, cbin=cbin, cwin=cwin, normalize=normalize, fs=int(fs),
                ax=flat[i], saveFig=False, prettify=True, color=color, ref_per=False,
            )
        except TypeError:
            nplt.plot_acg(
                dp, u, cbin=cbin, cwin=cwin, normalize=normalize, fs=int(fs),
                ax=flat[i], saveFig=False, prettify=True, color=i % 6, ref_per=False,
            )
        _recolor_acg_axis(flat[i], color)
        # Force a consistent, colour-matched per-unit title (npyx's own title is unreliable
        # across grid panels). Per-panel axis labels are replaced by a single shared pair below.
        flat[i].set_title(f"unit {u}", color=color, fontsize=11, fontweight="bold")
        flat[i].set_xlabel("")
        flat[i].set_ylabel("")
    for j in range(len(units), flat.size):
        flat[j].set_visible(False)
    for ax in fig.get_axes():
        _despine(ax)
    fig.supxlabel("Time (ms)", fontsize=11)
    fig.supylabel(_ACG_YLABEL.get(normalize, "Autocorrelation"), fontsize=11)
    _apply_theme(fig, dark)
    return fig


def _ensure_npyx_corr():
    """Import and return ``npyx.corr`` (vendored copy is on sys.path via the bridge)."""
    import sys

    app_root = Path(__file__).resolve().parents[1]
    if (app_root / "npyx").exists() and str(app_root) not in sys.path:
        sys.path.insert(0, str(app_root))
    import npyx.corr as ncorr  # type: ignore

    return ncorr


def _ccg_theme(dark: bool) -> dict:
    """Resolved colours for the CCG grid (white default / dark variant)."""
    if dark:
        return {
            "bg": "#0b0f14", "fg": "#e8eef7", "muted": "#8b97a8",
            "ccg": "#c8d2e0", "zero": "#9aa6b6", "spine": "#e8eef7",
        }
    return {
        "bg": "white", "fg": "#1a1a1a", "muted": "#6b6b6b",
        "ccg": "#444444", "zero": "#9a9a9a", "spine": "#1a1a1a",
    }


def ccg_grid_figure(
    dp: str,
    units: Sequence[int],
    *,
    cbin: float = 0.5,
    cwin: float = 100.0,
    fs: float = 30000.0,
    normalize: str = "Hertz",
    style: str = "bar",
    dark: bool = False,
):
    """Build a clean N x N correlogram grid (ACGs on the diagonal, CCGs off it).

    Data is computed directly with the vendored ``npyx.corr`` (``acg`` /
    ``ccg``), and laid out on a custom matplotlib GridSpec so the grid stays
    readable: diagonal ACGs are filled in each unit's Okabe-Ito colour, off-
    diagonal CCGs are thin dark lines, only the left column carries a y-label and
    only the bottom row carries the time axis, with unit-id headers along the top
    row and left column. Capped to the first 6 selected units.
    """
    corr = _ensure_npyx_corr()
    plt.close("all")
    units = [int(u) for u in units]
    if len(units) < 2:
        raise ValueError("Select at least two distinct units for a CCG grid.")

    n_total = len(units)
    capped = n_total > 8
    units = units[:8]
    n = len(units)
    th = _ccg_theme(dark)

    # Lag axis (ms): shared across all panels.
    lags = np.arange(-cwin / 2.0, cwin / 2.0 + cbin, cbin)

    # Compute every panel up front (diagonal = ACG, off-diagonal = CCG[0,1]).
    panels: dict[tuple[int, int], Optional[np.ndarray]] = {}
    for i, ui in enumerate(units):
        for j, uj in enumerate(units):
            try:
                if i == j:
                    y = np.asarray(corr.acg(dp, ui, cbin, cwin, fs=int(fs),
                                            normalize=normalize), dtype=float)
                else:
                    cc = np.asarray(corr.ccg(dp, [ui, uj], cbin, cwin, fs=int(fs),
                                             normalize=normalize), dtype=float)
                    y = cc[0, 1]
                panels[(i, j)] = y
            except Exception:
                panels[(i, j)] = None

    cell = 1.85
    fig = plt.figure(figsize=(cell * n + 0.6, cell * n + 0.7),
                     constrained_layout=True)
    fig.patch.set_facecolor(th["bg"])
    gs = fig.add_gridspec(n, n)

    y_unit = "Hz" if str(normalize).lower().startswith("hert") else str(normalize)
    axes = np.empty((n, n), dtype=object)
    for i in range(n):
        for j in range(n):
            ax = fig.add_subplot(gs[i, j])
            axes[i, j] = ax
            ax.set_facecolor(th["bg"])
            y = panels[(i, j)]
            if y is None or y.size == 0:
                ax.text(0.5, 0.5, "n/a", transform=ax.transAxes, ha="center",
                        va="center", color=th["muted"], fontsize=7)
            else:
                x = lags[: y.size]
                if i == j:
                    # Diagonal = autocorrelogram: clean bars, no 0-lag marker.
                    color = OKABE_ITO[i % len(OKABE_ITO)]
                    ax.fill_between(x, y, step="mid", color=color, alpha=0.85,
                                    lw=0, zorder=2)
                else:
                    # Off-diagonal = cross-correlogram: keep the 0-lag reference.
                    ax.axvline(0.0, color=th["zero"], ls="--", lw=0.7, zorder=1)
                    ax.fill_between(x, y, step="mid", color=th["ccg"], alpha=0.18,
                                    lw=0, zorder=2)
                    ax.plot(x, y, color=th["ccg"], lw=0.8, zorder=3)
                ax.set_xlim(x[0], x[-1])
                ax.set_ylim(bottom=0)

            # Clean spines / ticks.
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(th["spine"])
                ax.spines[side].set_linewidth(0.7)
            ax.tick_params(colors=th["fg"], labelsize=6, width=0.6, length=2.5,
                           direction="out")

            # Declutter: only bottom row gets the time axis, only left column the y-label.
            if i == n - 1:
                ax.set_xlabel("Time (ms)", color=th["fg"], fontsize=7.5)
            else:
                ax.tick_params(labelbottom=False)
            if j == 0:
                ax.set_ylabel(y_unit, color=th["fg"], fontsize=7.5)
            else:
                ax.tick_params(labelleft=False)

            # Unit-id headers: top row column headers + left column row headers.
            if i == 0:
                ax.set_title(f"u{units[j]}", color=th["fg"], fontsize=8.5,
                             fontweight="bold", pad=4)
            if j == 0:
                ax.annotate(f"u{units[i]}", xy=(0, 0.5), xytext=(-30, 0),
                            textcoords="offset points", xycoords="axes fraction",
                            ha="right", va="center", rotation=90,
                            color=th["fg"], fontsize=8.5, fontweight="bold")

    title_color = "#ffffff" if dark else th["fg"]
    note = f"   |   showing 6 of {n_total} units" if capped else ""
    fig.suptitle(f"Correlogram grid ({y_unit})   |   ACG diagonal, CCG off-diagonal"
                 f"   |   bin {cbin:g} ms, win {cwin:g} ms{note}",
                 color=title_color, fontsize=11, fontweight="bold", x=0.012, ha="left")
    return fig


def waveform_figure(
    dp: str,
    unit: int,
    *,
    n_channels: int = 12,
    n_waveforms: int = 200,
    fs: float = 30000.0,
    dark: bool = False,
):
    """Build an npyx multi-channel mean-waveform figure (``plot_wvf``) for one unit."""
    nplt = _ensure_npyx_plot()
    plt.close("all")
    fig = nplt.plot_wvf(
        dp, int(unit), Nchannels=int(n_channels), n_waveforms=int(n_waveforms),
        saveFig=False, fs=int(fs),
    )
    _apply_theme(fig, dark)
    return fig
