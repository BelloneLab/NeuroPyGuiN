"""Publication-clean, NeuroPyxels-style unit figures for the post-processing GUI.

This module renders two information-dense matplotlib figures that match the look of
NeuroPyxels (npyx) output: thin axes, no top/right spines, sans-serif text, scalebars,
peak-channel waveforms laid out on the real probe geometry, Hertz-normalised
autocorrelograms, and stacked multichannel raw traces with overlaid spike ticks.

Both builders use the non-interactive Agg backend (``matplotlib.use("Agg")``) and
return a bare :class:`matplotlib.figure.Figure`; the caller wraps it in a
``FigureCanvasQTAgg``. No ``pyplot.show`` is ever called. A dark theme tint is
applied when ``dark=True``; the default is a white npyx look.

Public API (hard contract, called by the GUI layer):

    unit_basics_figure(dataset, units, *, window_start_s=0.0, window_s=1.0,
                       ifr_bin_ms=30.0, acg_bin_ms=1.0, acg_win_ms=100.0,
                       isi_max_ms=200.0, show_ifr=True, dark=False) -> Figure

    raw_explorer_figure(dataset, *, t0_s=0.0, dur_s=1.0, n_channels=32,
                        hp_hz=300.0, lp_hz=0.0, center_channel=None,
                        overlay_units=(), y_mode="channel", dark=False) -> Figure
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

os.environ.setdefault("QT_API", "pyside6")

import matplotlib

matplotlib.use("Agg")  # figure creation only; the caller reparents into a Qt canvas

from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402


# --------------------------------------------------------------------------- #
# Theme
# --------------------------------------------------------------------------- #

# npyx-style categorical palette for overlaying several units (colorblind-aware,
# distinct hues, all readable on both white and dark backgrounds).
_UNIT_COLORS = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#8C564B",  # brown
    "#BCBD22",  # olive
]


class _Theme:
    """Resolved color set for either the white (npyx default) or dark look."""

    def __init__(self, dark: bool) -> None:
        self.dark = bool(dark)
        if self.dark:
            self.bg = "#0b0f14"
            self.fg = "#e8eef7"
            self.muted = "#8b97a8"
            self.faint = "#2a323d"
            self.grid = "#1c232c"
            self.accent = "#4ea1ff"
            self.trace = "#c8d2e0"
            self.point = "#9fb2cc"
            self.refband = "#ff6b6b"
        else:
            self.bg = "white"
            self.fg = "#1a1a1a"
            self.muted = "#6b6b6b"
            self.faint = "#cfcfcf"
            self.grid = "#e6e6e6"
            self.accent = "#0072B2"
            self.trace = "#333333"
            self.point = "#3a4a63"
            self.refband = "#d62728"


def _style_axes(ax, theme: _Theme) -> None:
    """Apply the npyx clean look: no top/right spines, outward thin ticks, themed colors."""
    ax.set_facecolor(theme.bg)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.8)
        ax.spines[side].set_color(theme.fg)
    ax.tick_params(
        which="both", direction="out", length=3.0, width=0.8,
        colors=theme.fg, labelsize=7,
    )
    ax.xaxis.label.set_color(theme.fg)
    ax.yaxis.label.set_color(theme.fg)
    ax.xaxis.label.set_size(8)
    ax.yaxis.label.set_size(8)
    ax.title.set_color(theme.fg)
    ax.title.set_size(8.5)


def _scalebar(ax, theme: _Theme, x0, y0, dx, dy, x_label, y_label, *, lw=1.2):
    """Draw an L-shaped uV+ms scalebar in data coordinates (no axes/ticks needed)."""
    ax.plot([x0, x0], [y0, y0 + dy], color=theme.fg, lw=lw, solid_capstyle="butt",
            clip_on=False, zorder=10)
    ax.plot([x0, x0 + dx], [y0, y0], color=theme.fg, lw=lw, solid_capstyle="butt",
            clip_on=False, zorder=10)
    ax.text(x0 + dx * 0.5, y0 - dy * 0.06, x_label, ha="center", va="top",
            fontsize=6.5, color=theme.fg, clip_on=False)
    ax.text(x0 - dx * 0.12, y0 + dy * 0.5, y_label, ha="right", va="center",
            fontsize=6.5, color=theme.fg, rotation=90, clip_on=False)


def _nice_round(value: float) -> float:
    """Round a positive value to a clean 1/2/5 x 10^n figure for scalebars."""
    if not np.isfinite(value) or value <= 0:
        return 1.0
    exp = np.floor(np.log10(value))
    frac = value / (10 ** exp)
    if frac < 1.5:
        nice = 1.0
    elif frac < 3.5:
        nice = 2.0
    elif frac < 7.5:
        nice = 5.0
    else:
        nice = 10.0
    return float(nice * (10 ** exp))


# --------------------------------------------------------------------------- #
# Per-unit data helpers
# --------------------------------------------------------------------------- #

def _session_duration_s(dataset) -> float:
    st = np.asarray(dataset.spike_times)
    if st.size == 0:
        return 1.0
    return float(st.max()) / float(dataset.sample_rate)


def _peak_channel(dataset, unit: int) -> Tuple[Optional[int], Optional[int]]:
    """Return (template-channel index, mapped channel id) of the unit's peak channel."""
    w = dataset.mean_template_waveform(unit)
    if w is None or w.ndim != 2 or w.shape[1] == 0:
        return None, None
    idx = int(np.nanargmax(np.nanmax(np.abs(w), axis=0)))
    cmap = dataset.channel_map
    if cmap is not None and np.asarray(cmap).squeeze().size > idx:
        ch = int(np.asarray(cmap).squeeze()[idx])
    else:
        ch = idx
    return idx, ch


def _unit_amplitudes(dataset, unit: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (spike_times_s, per-spike amplitude) for one unit, aligned to amplitudes.npy."""
    amp_path = Path(dataset.ks_folder) / "amplitudes.npy"
    sc = np.asarray(dataset.spike_clusters)
    mask = sc == int(unit)
    t = np.asarray(dataset.spike_times)[mask].astype(float) / float(dataset.sample_rate)
    if not amp_path.exists():
        return t, np.array([])
    try:
        amp = np.load(amp_path, mmap_mode="r").squeeze()
    except Exception:
        return t, np.array([])
    if amp.shape[0] != sc.shape[0]:
        return t, np.array([])
    a = np.asarray(amp[mask], dtype=float)
    return t, a


def _refractory_violation_pct(t_s: np.ndarray, refractory_ms: float = 2.0) -> float:
    """Fraction (in %) of inter-spike-intervals shorter than the refractory period."""
    if t_s.size < 2:
        return 0.0
    isi_ms = np.diff(np.sort(t_s)) * 1000.0
    return 100.0 * float(np.mean(isi_ms < refractory_ms))


def _extract_spike_waveforms(dataset, unit: int, channel_indices: np.ndarray, *,
                             half_ms: float = 1.3, max_spikes: int = 300):
    """Extract real spike snippets from the AP binary for ``channel_indices``.

    Reads a +/- ``half_ms`` window around up to ``max_spikes`` randomly sampled
    spikes of ``unit`` directly from the memory-mapped int16 SpikeGLX file and
    scales to microvolts via ``dataset.bit_uV``. Spikes too close to the file
    edges are skipped. Returns ``(t_ms, mean, sem, n_used)`` where ``mean`` /
    ``sem`` have shape ``(n_window_samples, len(channel_indices))``, or
    ``(None, None, None, 0)`` when the binary is unavailable / unreadable.

    ``channel_indices`` are file-channel indices (0..n_channels-1), i.e. the
    template/channel-map indices used everywhere else in this module.
    """
    ap = getattr(dataset, "ap_bin_path", None)
    if ap is None or not Path(ap).exists():
        return None, None, None, 0
    ch = np.asarray(channel_indices, dtype=np.int64)
    if ch.size == 0:
        return None, None, None, 0

    fs = float(dataset.sample_rate)
    n_ch_file = int(dataset.n_channels)
    half = int(round(half_ms * 1e-3 * fs))
    win = 2 * half + 1

    sc = np.asarray(dataset.spike_clusters)
    spk = np.asarray(dataset.spike_times)[sc == int(unit)].astype(np.int64)
    if spk.size == 0:
        return None, None, None, 0

    try:
        data = np.memmap(ap, dtype=np.int16, mode="r")
    except Exception:
        return None, None, None, 0
    n_samples = data.size // n_ch_file
    if n_samples <= win:
        return None, None, None, 0

    # Random subsample, then keep only spikes with a full window inside the file.
    if spk.size > max_spikes:
        idx = np.random.default_rng(0).choice(spk.size, size=max_spikes, replace=False)
        spk = spk[idx]
    spk = spk[(spk - half >= 0) & (spk + half + 1 <= n_samples)]
    if spk.size < 3:
        return None, None, None, 0

    snippets = np.empty((spk.size, win, ch.size), dtype=np.float32)
    try:
        for i, s in enumerate(spk):
            block = data[(s - half) * n_ch_file: (s + half + 1) * n_ch_file]
            block = block.reshape(win, n_ch_file)
            snippets[i] = block[:, ch].astype(np.float32)
    except Exception:
        return None, None, None, 0
    finally:
        del data

    snippets *= float(dataset.bit_uV)
    # Remove a per-snippet, per-channel pre-trough baseline so traces overlay cleanly.
    snippets -= np.median(snippets[:, :max(3, half // 3), :], axis=1, keepdims=True)

    mean = snippets.mean(axis=0)
    sem = snippets.std(axis=0, ddof=1) / np.sqrt(snippets.shape[0])
    t_ms = (np.arange(win) - half) / fs * 1000.0
    return t_ms, mean, sem, int(snippets.shape[0])


# --------------------------------------------------------------------------- #
# Panel: waveform on probe geometry
# --------------------------------------------------------------------------- #

def _draw_waveform_geometry(ax, dataset, units, theme: _Theme, *, n_neighbors=11):
    """Mean waveform(s) drawn at real probe (x,y) positions around the peak channel."""
    primary = int(units[0])
    w0 = dataset.mean_template_waveform(primary)
    cp = dataset.channel_positions
    if w0 is None or cp is None or w0.ndim != 2:
        _style_axes(ax, theme)
        ax.text(0.5, 0.5, "no template waveform", transform=ax.transAxes,
                ha="center", va="center", color=theme.muted, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title("Waveform on probe", color=theme.fg, fontsize=8.5)
        return

    cp = np.asarray(cp, dtype=float)
    fs = float(dataset.sample_rate)

    peak_idx, peak_ch = _peak_channel(dataset, primary)
    if peak_idx is None:
        peak_idx = int(np.nanargmax(np.nanmax(np.abs(w0), axis=0)))

    # Choose neighbors by physical distance to the peak channel.
    px, py = cp[peak_idx]
    dist = np.sqrt((cp[:, 0] - px) ** 2 + (cp[:, 1] - py) ** 2)
    n_take = min(int(n_neighbors), cp.shape[0])
    sel = np.argsort(dist)[:n_take]

    # Real-spike mean +/- SEM for the primary unit on the selected channels.
    rt_ms, r_mean, r_sem, n_used = _extract_spike_waveforms(
        dataset, primary, sel, half_ms=1.3, max_spikes=300)
    have_band = r_mean is not None

    # Choose the trace abscissa / amplitude source: real snippets if available,
    # otherwise the template (no band).
    if have_band:
        t_ms = rt_ms
        # r_mean columns are aligned to ``sel``; build a per-sel-index lookup.
        sel_mean = {int(ci): r_mean[:, j] for j, ci in enumerate(sel)}
        sel_sem = {int(ci): r_sem[:, j] for j, ci in enumerate(sel)}
        peak_amp = float(np.nanmax(np.abs(sel_mean.get(int(peak_idx),
                                                        r_mean[:, 0]))))
    else:
        n_samp = w0.shape[0]
        t_ms = (np.arange(n_samp) - n_samp // 2) / fs * 1000.0
        peak_amp = float(np.nanmax(np.abs(w0[:, peak_idx])))
    if not np.isfinite(peak_amp) or peak_amp <= 0:
        peak_amp = 1.0

    # Geometry-aware scaling: traces span ~80% of the channel pitch.
    ux = np.unique(cp[:, 0])
    uy = np.unique(cp[:, 1])
    dx_pitch = float(np.min(np.diff(ux))) if ux.size > 1 else 32.0
    dy_pitch = float(np.min(np.diff(uy))) if uy.size > 1 else 20.0
    t_span = t_ms.max() - t_ms.min()
    x_scale = (dx_pitch * 0.82) / max(t_span, 1e-6)
    y_scale = (dy_pitch * 1.7) / (2.0 * peak_amp)

    overlay_colors = _UNIT_COLORS
    # Overlay units use the template line only (no band) to keep the panel legible.
    overlay_waveforms = []
    for k, u in enumerate(units[1:], start=1):
        wk = dataset.mean_template_waveform(int(u))
        if wk is not None and wk.shape == w0.shape:
            overlay_waveforms.append((int(u), wk, overlay_colors[k % len(overlay_colors)]))
    n_tmpl = w0.shape[0]
    t_tmpl_ms = (np.arange(n_tmpl) - n_tmpl // 2) / fs * 1000.0

    for ci in sel:
        cx, cy = cp[ci]
        is_peak = ci == peak_idx
        if is_peak:
            # Emphasize the peak channel with a soft highlight box (drawn first).
            ax.add_patch(mpatches.Rectangle(
                (cx - dx_pitch * 0.46, cy - dy_pitch * 1.0),
                dx_pitch * 0.92, dy_pitch * 2.0,
                facecolor=theme.accent, alpha=0.07 if not theme.dark else 0.16,
                edgecolor=theme.accent, lw=0.6, zorder=1))
        # Primary unit: mean line with shaded +/- SEM band (or template fallback).
        if have_band:
            m = sel_mean[int(ci)]
            s = sel_sem[int(ci)]
            xx = cx + t_ms * x_scale
            ax.fill_between(xx, cy + (m - s) * y_scale, cy + (m + s) * y_scale,
                            color=theme.accent, alpha=0.28 if not theme.dark else 0.34,
                            lw=0, zorder=3 if not is_peak else 4)
            ax.plot(xx, cy + m * y_scale, color=theme.accent,
                    lw=1.3 if is_peak else 0.95, alpha=1.0 if is_peak else 0.85,
                    solid_capstyle="round", zorder=5 if is_peak else 4)
        else:
            ax.plot(cx + t_tmpl_ms * x_scale, cy + w0[:, ci] * y_scale,
                    color=theme.accent, lw=1.3 if is_peak else 0.95,
                    alpha=1.0 if is_peak else 0.85, solid_capstyle="round",
                    zorder=5 if is_peak else 4)
        # Overlay units (template lines only).
        for (u, wk, col) in overlay_waveforms:
            ax.plot(cx + t_tmpl_ms * x_scale, cy + wk[:, ci] * y_scale,
                    color=col, lw=0.9, alpha=0.7, solid_capstyle="round",
                    zorder=3, clip_on=True)

    # Scalebar in the lower-left of the panel.
    x_lo = cp[sel, 0].min() - dx_pitch * 0.55
    y_lo = cp[sel, 1].min() - dy_pitch * 1.2
    sb_t = _nice_round(0.4 * t_span)
    sb_uv = _nice_round(0.5 * peak_amp)
    _scalebar(ax, theme, x_lo, y_lo, sb_t * x_scale, sb_uv * y_scale,
              f"{sb_t:g} ms", f"{sb_uv:g} uV")

    ax.set_xlim(cp[sel, 0].min() - dx_pitch * 0.9,
                cp[sel, 0].max() + dx_pitch * 0.7)
    ax.set_ylim(y_lo - dy_pitch * 0.6, cp[sel, 1].max() + dy_pitch * 1.5)
    ax.set_aspect("auto")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_facecolor(theme.bg)
    if have_band:
        sub = f"{n_used} spk, mean+/-SEM"
    else:
        sub = "template, no SEM"
    title = f"Waveform - ch {peak_ch}"
    ax.set_title(title, color=theme.fg, fontsize=8.5, pad=15, loc="left")
    # Place the SEM/template note as a small left-aligned sub-line under the title.
    ax.text(0.0, 1.012, sub, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=6.5, color=theme.muted)


# --------------------------------------------------------------------------- #
# Panel: ACG (Hertz)
# --------------------------------------------------------------------------- #

def _ensure_npyx_plot():
    app_root = Path(__file__).resolve().parents[1]
    if (app_root / "npyx").exists() and str(app_root) not in sys.path:
        sys.path.insert(0, str(app_root))
    try:
        import npyx.plot as nplt  # type: ignore
        return nplt
    except Exception:
        return None


def _draw_acg(ax, dataset, units, theme: _Theme, *, bin_ms, win_ms):
    """Hertz-normalised autocorrelogram (clean bars; no refractory band or 0-lag marker)."""
    _style_axes(ax, theme)
    primary = int(units[0])
    nplt = _ensure_npyx_plot()
    drew_with_npyx = False
    if nplt is not None:
        try:
            sc = np.asarray(dataset.spike_clusters)
            train = np.asarray(dataset.spike_times)[sc == primary].astype(np.int64)
            if train.size >= 2:
                nplt.plot_acg(
                    str(dataset.ks_folder), primary, cbin=float(bin_ms),
                    cwin=float(win_ms), normalize="Hertz",
                    fs=int(dataset.sample_rate), ax=ax, train=train,
                    saveFig=False, prettify=True, title="",
                )
                drew_with_npyx = True
        except Exception:
            drew_with_npyx = False

    if not drew_with_npyx:
        # Manual fallback using the engine correlogram.
        centers, vals = dataset.correlogram(
            primary, primary, bin_ms=float(bin_ms), win_ms=float(win_ms),
            remove_zero=True, normalize="Hertz")
        if centers.size:
            ax.bar(centers, vals, width=float(bin_ms) * 0.95, color=theme.accent,
                   edgecolor="none", zorder=3)
            ax.set_xlim(-float(win_ms) / 2, float(win_ms) / 2)
            ax.set_ylim(bottom=0)

    # Re-apply clean styling on top of whatever npyx drew (npyx sets its own
    # facecolor / spines / tick colors, so we reset everything to the theme look).
    _style_axes(ax, theme)
    ax.set_facecolor(theme.bg)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_color(theme.fg)
        lab.set_fontsize(7)
    ax.set_xlabel("lag (ms)")
    ax.set_ylabel("firing rate (Hz)")
    ax.set_title("Autocorrelogram", color=theme.fg, fontsize=8.5)


# --------------------------------------------------------------------------- #
# Panel: ISI histogram (log x)
# --------------------------------------------------------------------------- #

def _draw_isi(ax, dataset, unit, theme: _Theme, *, max_ms, refractory_ms=2.0):
    """ISI histogram with a log-spaced x-axis and a refractory-violation annotation."""
    _style_axes(ax, theme)
    t = dataset.unit_spike_times_s(int(unit))
    if t.size < 2:
        ax.text(0.5, 0.5, "too few spikes", transform=ax.transAxes,
                ha="center", va="center", color=theme.muted, fontsize=8)
        ax.set_title("ISI distribution", color=theme.fg, fontsize=8.5)
        return
    isi_ms = np.diff(np.sort(t)) * 1000.0
    isi_ms = isi_ms[isi_ms > 0]
    lo = max(0.2, float(np.nanpercentile(isi_ms, 0.1)) if isi_ms.size else 0.2)
    hi = float(max(max_ms, np.nanpercentile(isi_ms, 99.5))) if isi_ms.size else max_ms
    bins = np.logspace(np.log10(lo), np.log10(max(hi, lo * 10)), 60)
    ax.hist(isi_ms, bins=bins, color=theme.accent, edgecolor="none", alpha=0.9, zorder=3)
    ax.set_xscale("log")
    ax.axvline(refractory_ms, color=theme.refband, ls="--", lw=0.9, zorder=4)
    rpv = _refractory_violation_pct(t, refractory_ms)
    ax.set_xlabel("inter-spike interval (ms)")
    ax.set_ylabel("count")
    ax.set_title("ISI distribution", color=theme.fg, fontsize=8.5)
    ax.text(0.97, 0.94,
            f"refractory < {refractory_ms:g} ms\n{rpv:.2f}% violations",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.5,
            color=theme.refband)


# --------------------------------------------------------------------------- #
# Panel: amplitude vs time (drift / stability)
# --------------------------------------------------------------------------- #

def _draw_amplitudes(ax, dataset, unit, theme: _Theme, *, max_points=20000):
    """Per-spike amplitude vs spike time over the whole session (drift / stability)."""
    _style_axes(ax, theme)
    t, a = _unit_amplitudes(dataset, int(unit))
    if t.size == 0 or a.size == 0:
        ax.text(0.5, 0.5, "no per-spike amplitudes", transform=ax.transAxes,
                ha="center", va="center", color=theme.muted, fontsize=8)
        ax.set_title("Amplitude over session", color=theme.fg, fontsize=8.5)
        return
    n = t.size
    if n > max_points:
        idx = np.random.default_rng(0).choice(n, size=max_points, replace=False)
        idx.sort()
        ts, ams = t[idx], a[idx]
    else:
        ts, ams = t, a
    ax.scatter(ts, ams, s=2.0, c=theme.point, alpha=0.25, edgecolors="none", zorder=3)
    # Robust y-limits to keep outliers from flattening the cloud.
    ylo = float(np.nanpercentile(a, 0.5))
    yhi = float(np.nanpercentile(a, 99.8))
    if yhi > ylo:
        ax.set_ylim(max(0.0, ylo - 0.05 * (yhi - ylo)), yhi + 0.08 * (yhi - ylo))
    ax.set_xlim(0, _session_duration_s(dataset))
    ax.set_xlabel("time (s)")
    ax.set_ylabel("amplitude (a.u.)")
    ax.set_title("Amplitude over session", color=theme.fg, fontsize=8.5)


# --------------------------------------------------------------------------- #
# Panel: firing rate over session
# --------------------------------------------------------------------------- #

def _draw_firing_rate(ax, dataset, units, theme: _Theme, *, bin_s=1.0, show_ifr=True,
                      ifr_bin_ms=30.0):
    """Binned firing rate across the whole session, optionally with an IFR trace."""
    _style_axes(ax, theme)
    dur = _session_duration_s(dataset)
    edges = np.arange(0.0, dur + bin_s, bin_s)
    centers = 0.5 * (edges[:-1] + edges[1:])
    overlay_colors = _UNIT_COLORS
    any_plotted = False
    for k, u in enumerate(units):
        t = dataset.unit_spike_times_s(int(u))
        if t.size == 0:
            continue
        counts, _ = np.histogram(t, bins=edges)
        rate = counts.astype(float) / bin_s
        color = theme.accent if (len(units) == 1) else overlay_colors[k % len(overlay_colors)]
        ax.plot(centers, rate, color=color, lw=0.9, alpha=0.95, zorder=3,
                label=f"unit {int(u)}")
        any_plotted = True
    if not any_plotted:
        ax.text(0.5, 0.5, "no spikes", transform=ax.transAxes,
                ha="center", va="center", color=theme.muted, fontsize=8)
    ax.set_xlim(0, dur)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("time (s)")
    ax.set_ylabel(f"firing rate (Hz, {bin_s:g} s bins)")
    ax.set_title("Firing rate over session", color=theme.fg, fontsize=8.5)
    if len(units) > 1:
        leg = ax.legend(loc="upper right", fontsize=6, frameon=False,
                        handlelength=1.2, borderaxespad=0.2)
        for txt in leg.get_texts():
            txt.set_color(theme.fg)


# --------------------------------------------------------------------------- #
# Public: unit basics card
# --------------------------------------------------------------------------- #

def unit_basics_figure(dataset, units, *, window_start_s=0.0, window_s=1.0,
                       ifr_bin_ms=30.0, acg_bin_ms=1.0, acg_win_ms=100.0,
                       isi_max_ms=200.0, show_ifr=True, dark=False):
    """Return a matplotlib Figure: a clean multi-panel 'unit card' for ``units[0]``.

    Panels: mean waveform on the real probe geometry (peak channel highlighted),
    a Hertz-normalised autocorrelogram with refractory band, a log-x ISI histogram
    with refractory-violation %, per-spike amplitude over the whole session
    (drift / stability), and the binned session firing rate. When ``len(units) > 1``
    the additional units' waveforms (where they share geometry) and firing-rate
    traces are overlaid in distinct colors so they stay legible.

    Parameters mirror the GUI controls; ``window_start_s`` / ``window_s`` /
    ``ifr_bin_ms`` are accepted for API compatibility (the session-wide panels are
    preferred over a fixed short raster, which is empty for low-rate units).
    """
    theme = _Theme(dark)
    units = [int(u) for u in (units if units is not None else [])]
    if not units:
        fig = Figure(figsize=(11.0, 7.0), dpi=110)
        fig.patch.set_facecolor(theme.bg)
        ax = fig.add_subplot(111)
        ax.set_facecolor(theme.bg)
        ax.axis("off")
        ax.text(0.5, 0.5, "No unit selected.", ha="center", va="center",
                color=theme.muted, fontsize=12)
        return fig

    primary = units[0]
    fig = Figure(figsize=(12.4, 7.4), dpi=110, constrained_layout=True)
    fig.patch.set_facecolor(theme.bg)
    fig.set_constrained_layout_pads(w_pad=0.06, h_pad=0.10, wspace=0.05, hspace=0.07)

    # Layout: left column = waveform (tall); right block = 2x2 of ACG / ISI / amp / FR.
    # constrained_layout keeps panel titles from overlapping neighbouring axes.
    gs = fig.add_gridspec(2, 3, width_ratios=[1.08, 1.0, 1.0], height_ratios=[1.0, 1.0])
    ax_wf = fig.add_subplot(gs[:, 0])
    ax_acg = fig.add_subplot(gs[0, 1])
    ax_isi = fig.add_subplot(gs[0, 2])
    ax_amp = fig.add_subplot(gs[1, 1])
    ax_fr = fig.add_subplot(gs[1, 2])

    _draw_waveform_geometry(ax_wf, dataset, units, theme)
    _draw_acg(ax_acg, dataset, units, theme, bin_ms=acg_bin_ms, win_ms=acg_win_ms)
    _draw_isi(ax_isi, dataset, primary, theme, max_ms=isi_max_ms)
    _draw_amplitudes(ax_amp, dataset, primary, theme)
    _draw_firing_rate(ax_fr, dataset, units, theme, bin_s=1.0, show_ifr=show_ifr,
                      ifr_bin_ms=ifr_bin_ms)

    # Single concise title with key metrics for the primary unit (no grey subtitle:
    # the panel titles already describe each panel).
    t = dataset.unit_spike_times_s(primary)
    dur = _session_duration_s(dataset)
    n_spk = int(t.size)
    mean_fr = n_spk / dur if dur > 0 else 0.0
    rpv = _refractory_violation_pct(t)
    _, peak_ch = _peak_channel(dataset, primary)
    peak_str = "n/a" if peak_ch is None else str(peak_ch)
    extra = f"  (+{len(units) - 1} overlaid)" if len(units) > 1 else ""
    title = (f"Unit {primary}{extra}  |  peak ch {peak_str}  |  "
             f"{n_spk:,} spikes  |  {mean_fr:.1f} Hz  |  {rpv:.2f}% RPV")
    title_color = "#ffffff" if theme.dark else theme.fg
    fig.suptitle(title, color=title_color, fontsize=11, fontweight="bold",
                 x=0.012, ha="left")
    return fig


# --------------------------------------------------------------------------- #
# Public: raw explorer (npyx plot_raw_units style)
# --------------------------------------------------------------------------- #

def _no_raw_figure(theme: _Theme, message: str):
    fig = Figure(figsize=(11.0, 7.0), dpi=110)
    fig.patch.set_facecolor(theme.bg)
    ax = fig.add_subplot(111)
    ax.set_facecolor(theme.bg)
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center",
            color=theme.muted, fontsize=12, wrap=True)
    return fig


def raw_explorer_figure(dataset, *, t0_s=0.0, dur_s=1.0, n_channels=32,
                        hp_hz=300.0, lp_hz=0.0, center_channel=None,
                        overlay_units=(), y_mode="channel", dark=False):
    """Return a matplotlib Figure: npyx ``plot_raw_units``-style multichannel traces.

    Stacked filtered raw traces for ``n_channels`` around the peak channel of
    ``overlay_units[0]`` (or ``center_channel``), vertically offset, with a uV+ms
    scalebar. The y-axis is labelled by channel id (``y_mode="channel"``) or depth
    in mm (``y_mode="depth"``). Spike times of each unit in ``overlay_units`` within
    ``[t0_s, t0_s + dur_s]`` are overlaid as colored ticks on that unit's peak
    channel, with a small legend. Returns a clear message figure when the raw
    binary is unavailable.
    """
    theme = _Theme(dark)
    overlay_units = [int(u) for u in (overlay_units or [])]

    if getattr(dataset, "ap_bin_path", None) is None:
        return _no_raw_figure(theme, "Raw binary not available\n(no .ap.bin found from params/meta).")

    # Resolve the center channel (template index) for the chunk window.
    center_idx = center_channel
    peak_idx_by_unit = {}
    for u in overlay_units:
        pidx, _ = _peak_channel(dataset, u)
        if pidx is not None:
            peak_idx_by_unit[u] = pidx
    if center_idx is None and overlay_units:
        center_idx = peak_idx_by_unit.get(overlay_units[0])
    if center_idx is None:
        center_idx = int(dataset.n_channels) // 2

    try:
        t, x, channel_ids, channel_order = dataset.raw_explorer_chunk(
            t0_s=float(t0_s), dur_s=float(dur_s), max_channels=int(n_channels),
            hp_hz=float(hp_hz), lp_hz=float(lp_hz), downsample=1,
            center_channel=int(center_idx),
        )
    except Exception as exc:  # pragma: no cover - depends on disk state
        return _no_raw_figure(theme, f"Could not read raw chunk:\n{exc}")

    if x is None or x.size == 0:
        return _no_raw_figure(theme, "Raw chunk is empty for the requested window.")

    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    t_rel_ms = (t - t[0]) * 1000.0
    n_ch = x.shape[1]
    span_ms = float(t_rel_ms[-1] - t_rel_ms[0]) if t_rel_ms.size > 1 else 1.0
    left_pad = max(span_ms * 0.012, 0.3)

    # Center each trace and clip extreme samples to its own offset slot so a big
    # spike on one channel does not bleed across neighbours (npyx-style spacing).
    x = x - np.nanmedian(x, axis=0, keepdims=True)
    spread = float(np.nanpercentile(np.abs(x), 95))
    spread = max(spread, 1e-3)
    offset = spread * 3.4
    clip = offset * 0.60
    x_disp = np.clip(x, -clip, clip)

    fig = Figure(figsize=(11.6, 8.0), dpi=110)
    fig.patch.set_facecolor(theme.bg)
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.90, bottom=0.075)
    _style_axes(ax, theme)

    cp = dataset.channel_positions
    cp = np.asarray(cp, dtype=float) if cp is not None else None

    order_to_row = {int(channel_order[i]): i for i in range(n_ch)}
    peak_rows = {u: order_to_row.get(peak_idx_by_unit.get(u))
                 for u in overlay_units if peak_idx_by_unit.get(u) in order_to_row}
    row_color = {}
    for k, u in enumerate(overlay_units):
        r = peak_rows.get(u)
        if r is not None:
            row_color[r] = _UNIT_COLORS[k % len(_UNIT_COLORS)]

    # Plot channels bottom (low order index) to top; highlight overlay peak rows.
    y_positions = []
    y_tick_labels = []
    for i in range(n_ch):
        y_base = i * offset
        y_positions.append(y_base)
        is_peak_row = i in row_color
        col = row_color.get(i, theme.trace)
        lw = 0.65 if is_peak_row else 0.4
        z = 4 if is_peak_row else 2
        ax.plot(t_rel_ms, x_disp[:, i] + y_base, color=col, lw=lw,
                solid_capstyle="round", zorder=z, rasterized=True)
        ch_id = int(channel_ids[i])
        order_i = int(channel_order[i])
        if y_mode == "depth" and cp is not None and order_i < cp.shape[0]:
            y_tick_labels.append(f"{cp[order_i, 1] / 1000.0:.3f}")
        else:
            y_tick_labels.append(str(ch_id))

    # Tick a readable subset of channels.
    n_ticks = min(n_ch, 18)
    tick_step = max(1, n_ch // n_ticks)
    tick_idx = list(range(0, n_ch, tick_step))
    ax.set_yticks([y_positions[i] for i in tick_idx])
    ax.set_yticklabels([y_tick_labels[i] for i in tick_idx], fontsize=6.5)
    ax.set_ylabel("depth (mm)" if y_mode == "depth" else "channel")
    ax.set_xlabel("time (ms)")

    # Overlay spike ticks per unit just above its peak-channel trace.
    legend_handles = []
    t1 = float(t[0])
    t2 = float(t[-1])
    for k, u in enumerate(overlay_units):
        row = peak_rows.get(u)
        color = _UNIT_COLORS[k % len(_UNIT_COLORS)]
        if row is None:
            legend_handles.append(Line2D([0], [0], color=color, lw=2.0,
                                         label=f"unit {u} (off-window)"))
            continue
        st = dataset.unit_spike_times_s(u)
        st_win = st[(st >= t1) & (st <= t2)]
        if st_win.size == 0:
            legend_handles.append(Line2D([0], [0], color=color, lw=2.0,
                                         label=f"unit {u} (0 spk here)"))
            continue
        st_rel_ms = (st_win - t1) * 1000.0
        y_base = row * offset
        # Marker row sits just above the trace band, plus a thin guide tick down.
        marker_y = y_base + offset * 0.78
        ax.plot(st_rel_ms, np.full_like(st_rel_ms, marker_y), marker="v",
                ms=5, color=color, ls="none", zorder=8, clip_on=True,
                markeredgecolor="none")
        ax.vlines(st_rel_ms, y_base - clip, y_base + clip, color=color,
                  lw=0.6, alpha=0.35, zorder=6)
        legend_handles.append(Line2D([0], [0], marker="v", color=color, lw=0,
                                     markersize=6, markeredgecolor="none",
                                     label=f"unit {u}: {st_win.size} spk"))

    # uV + ms scalebar in the lower-right corner (traces are already in uV).
    sb_ms = _nice_round(0.10 * span_ms)
    sb_uv = _nice_round(spread)
    x_lo = t_rel_ms[-1] - sb_ms * 1.25
    y_lo = -offset * 0.7
    ax.plot([x_lo, x_lo], [y_lo, y_lo + sb_uv], color=theme.fg, lw=1.4,
            clip_on=False, zorder=10, solid_capstyle="butt")
    ax.plot([x_lo, x_lo + sb_ms], [y_lo, y_lo], color=theme.fg, lw=1.4,
            clip_on=False, zorder=10, solid_capstyle="butt")
    ax.text(x_lo + sb_ms * 0.5, y_lo - offset * 0.18, f"{sb_ms:g} ms",
            ha="center", va="top", fontsize=7, color=theme.fg, clip_on=False)
    ax.text(x_lo - sb_ms * 0.06, y_lo + sb_uv * 0.5, f"{sb_uv:g} uV",
            ha="right", va="center", fontsize=7, color=theme.fg, rotation=90,
            clip_on=False)

    ax.set_xlim(t_rel_ms[0] - left_pad, t_rel_ms[-1])
    ax.set_ylim(-offset * 1.1, (n_ch - 1) * offset + offset * 1.2)

    filt_txt = []
    if hp_hz and hp_hz > 0:
        filt_txt.append(f"HP {hp_hz:g} Hz")
    if lp_hz and lp_hz > 0:
        filt_txt.append(f"LP {lp_hz:g} Hz")
    filt_str = ", ".join(filt_txt) if filt_txt else "wideband"
    center_ch_id = int(channel_ids[order_to_row[center_idx]]) if center_idx in order_to_row else center_idx
    title_color = "#ffffff" if theme.dark else theme.fg
    fig.text(0.075, 0.955,
             f"Raw traces  |  {n_ch} ch around ch {center_ch_id}  |  "
             f"t = {t0_s:.3f}-{t0_s + dur_s:.3f} s  |  {filt_str}",
             color=title_color, fontsize=11, fontweight="bold", ha="left", va="center")
    fig.text(0.075, 0.922,
             "Filtered AP-band traces, vertically offset by channel; colored markers "
             "are spike times of the overlaid units on their peak channel.",
             color=theme.muted, fontsize=7.5, ha="left", va="center")

    if legend_handles:
        leg = ax.legend(handles=legend_handles, loc="lower left",
                        bbox_to_anchor=(0.0, 1.005), ncol=max(1, len(legend_handles)),
                        fontsize=8, frameon=False, handlelength=1.0,
                        borderaxespad=0.0, columnspacing=1.4, labelspacing=0.3)
        for txt in leg.get_texts():
            txt.set_color(theme.fg)

    return fig


# --------------------------------------------------------------------------- #
# Public: condition PSTH
# --------------------------------------------------------------------------- #

def _resolve_trial_slice(n_trials: int, trial_from: int, trial_to: int) -> slice:
    """Mirror postprocessing_tab._condition_trial_slice (1-based, inclusive)."""
    total = max(0, int(n_trials))
    if total == 0:
        return slice(0, 0)
    start = max(1, int(trial_from))
    stop = total if int(trial_to) <= 0 else max(1, int(trial_to))
    if stop < start:
        start, stop = stop, start
    start = min(start, total)
    stop = min(stop, total)
    return slice(start - 1, stop)


def _baseline_subtract(mat: np.ndarray, t_ms: np.ndarray) -> np.ndarray:
    """Subtract each row's pre-event (t<0) mean firing rate."""
    mat = np.asarray(mat, dtype=float)
    if mat.size == 0 or mat.ndim != 2:
        return mat
    pre = np.asarray(t_ms, dtype=float) < 0.0
    if not np.any(pre):
        return mat
    return mat - np.nanmean(mat[:, pre], axis=1, keepdims=True)


def _condition_unit_means(cond: dict, t_ms: np.ndarray, *, trial_from: int,
                          trial_to: int, baseline: bool):
    """Per-unit trial-mean PSTH for one condition.

    Returns ``(unit_ids, units x bins mean, per_trial)`` where ``per_trial`` is a
    parallel list of the sliced ``(n_trials, n_bins)`` matrices (used for the
    across-trial SEM in per-unit mode).
    """
    unit_ids = list(cond.get("unit_ids", []))
    mats = cond.get("unit_trial_mats", [])
    rows = []
    kept_ids = []
    per_trial = []
    for uid, m in zip(unit_ids, mats):
        m = np.asarray(m, dtype=float)
        if m.ndim != 2 or m.size == 0:
            continue
        sl = _resolve_trial_slice(m.shape[0], trial_from, trial_to)
        sub = m[sl]
        if sub.size == 0:
            continue
        if baseline:
            sub = _baseline_subtract(sub, t_ms)
        rows.append(np.nanmean(sub, axis=0))
        per_trial.append(sub)
        kept_ids.append(int(uid))
    if not rows:
        return [], np.zeros((0, t_ms.size), dtype=float), []
    return kept_ids, np.vstack(rows), per_trial


def _psth_color(idx: int) -> str:
    return _UNIT_COLORS[idx % len(_UNIT_COLORS)]


def _empty_psth_figure(theme: _Theme, message: str):
    fig = Figure(figsize=(10.5, 7.0), dpi=110)
    fig.patch.set_facecolor(theme.bg)
    ax = fig.add_subplot(111)
    ax.set_facecolor(theme.bg)
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", color=theme.muted,
            fontsize=12, wrap=True)
    return fig


def _draw_psth_event_decor(ax, t_ms, theme: _Theme):
    """Dashed t=0 line and a light shaded pre-event window."""
    t0 = 0.0
    tmin = float(np.min(t_ms)) if t_ms.size else -1.0
    if tmin < 0:
        ax.axvspan(tmin, t0, color=theme.muted, alpha=0.08, lw=0, zorder=0)
    ax.axvline(t0, color=theme.fg, ls="--", lw=0.8, alpha=0.7, zorder=2)


def condition_psth_figure(psth_results, *, mode="average", baseline=False,
                          trial_from=1, trial_to=0, dark=False):
    """Return a matplotlib.figure.Figure rendering a multi-unit condition PSTH.

    ``psth_results`` is the dict produced by ``_compute_psth`` in
    ``postprocessing_tab.py`` (see that method for the canonical shape):
    ``t_ms`` (bin centres, ms, 0 = event), and ``conditions`` -- a list of dicts
    each with ``condition``/``selected_label``/``source_csv``, ``unit_ids``,
    ``unit_trial_mats`` (one ``(n_trials, n_bins)`` Hz matrix per unit) and
    ``trial_count``.

    Trial slicing (1-based, inclusive) and baseline subtraction mirror the GUI's
    ``_condition_trial_slice`` / ``_psth_baseline_subtract``. With ``baseline=True``
    each row's pre-event (t<0) mean is removed and the y-axis becomes Delta Rate.

    - ``mode="average"``: per condition, the across-units mean with a shaded +/-SEM
      band plus faint individual unit-mean traces, over a per-unit-row heatmap.
    - ``mode="per_unit"``: a grid of small panels (capped at 12 units, noted when
      capped), one per unit, each showing that unit's across-trial mean +/-SEM with
      conditions overlaid by colour, over the same heatmap.

    The heatmap shares the time axis; multiple conditions stack as labelled
    row-blocks. A perceptually-uniform colormap is used (magma for rate, a
    diverging RdBu_r centred on 0 when ``baseline=True``).
    """
    theme = _Theme(dark)
    if not psth_results or not isinstance(psth_results, dict):
        return _empty_psth_figure(theme, "No PSTH results to display.")
    t_ms = np.asarray(psth_results.get("t_ms", []), dtype=float)
    conditions = list(psth_results.get("conditions", []))
    if t_ms.size == 0 or not conditions:
        return _empty_psth_figure(theme, "No PSTH results to display.")

    y_label = "Delta Rate (Hz)" if baseline else "Rate (Hz)"
    bin_ms = float(psth_results.get("bin_ms", 0.0) or 0.0)

    # Pre-compute per-condition per-unit trial means (units x bins) once.
    cond_data = []
    for ci, cond in enumerate(conditions):
        ids, mat, per_trial = _condition_unit_means(
            cond, t_ms, trial_from=trial_from, trial_to=trial_to, baseline=baseline)
        if mat.shape[0] == 0:
            continue
        name = str(cond.get("condition", f"cond {ci + 1}"))
        sel = str(cond.get("selected_label", "") or "")
        label = f"{name}" + (f" [{sel}]" if sel else "")
        cond_data.append({
            "label": label, "unit_ids": ids, "mat": mat, "_per_trial": per_trial,
            "color": _psth_color(len(cond_data)),
            "trial_count": int(cond.get("trial_count", 0)),
        })
    if not cond_data:
        return _empty_psth_figure(theme, "PSTH results contain no usable trials.")

    if mode == "per_unit":
        return _psth_per_unit_figure(cond_data, t_ms, theme, baseline=baseline,
                                     y_label=y_label, bin_ms=bin_ms)
    return _psth_average_figure(cond_data, t_ms, theme, baseline=baseline,
                                y_label=y_label, bin_ms=bin_ms)


def _heatmap_cmap_norm(values: np.ndarray, baseline: bool):
    """Return (cmap, norm) for the unit-row heatmap (diverging if baseline)."""
    finite = values[np.isfinite(values)]
    if baseline:
        vmax = float(np.nanpercentile(np.abs(finite), 99)) if finite.size else 1.0
        vmax = max(vmax, 1e-6)
        return "RdBu_r", mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    vmax = float(np.nanpercentile(finite, 99)) if finite.size else 1.0
    vmin = float(np.nanpercentile(finite, 1)) if finite.size else 0.0
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return "magma", mcolors.Normalize(vmin=vmin, vmax=vmax)


def _draw_psth_heatmap(ax, cax, cond_data, t_ms, theme: _Theme, *, baseline, y_label):
    """Stacked per-unit-row heatmap (conditions as labelled row-blocks)."""
    blocks = [c["mat"] for c in cond_data]
    stacked = np.vstack(blocks)
    cmap, norm = _heatmap_cmap_norm(stacked, baseline)
    extent = [float(t_ms[0]), float(t_ms[-1]), stacked.shape[0], 0]
    im = ax.imshow(stacked, aspect="auto", origin="upper", cmap=cmap, norm=norm,
                   extent=extent, interpolation="nearest")

    # Block boundaries + condition labels on the left.
    y0 = 0
    yticks = []
    yticklabels = []
    for c in cond_data:
        h = c["mat"].shape[0]
        ymid = y0 + h / 2.0
        if len(cond_data) > 1:
            yticks.append(ymid)
            yticklabels.append(c["label"])
        y0 += h
        if y0 < stacked.shape[0]:
            ax.axhline(y0, color=theme.bg, lw=1.4, zorder=3)
            ax.axhline(y0, color=theme.fg, lw=0.5, alpha=0.5, zorder=3)
    ax.axvline(0.0, color=theme.fg, ls="--", lw=0.8, alpha=0.8, zorder=4)

    _style_axes(ax, theme)
    ax.set_xlabel("time from event (ms)")
    if len(cond_data) > 1:
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels, fontsize=7)
        ax.set_ylabel("condition / unit")
    else:
        ax.set_ylabel("unit")
        # Tick each unit row when there is only one condition and few units.
        ids = cond_data[0]["unit_ids"]
        if len(ids) <= 20:
            ax.set_yticks(np.arange(len(ids)) + 0.5)
            ax.set_yticklabels([str(i) for i in ids], fontsize=6.5)
    ax.set_title("Trial-averaged PSTH by unit", color=theme.fg, fontsize=9)

    cb = ax.figure.colorbar(im, cax=cax)
    cb.set_label(y_label, color=theme.fg, fontsize=8)
    cb.ax.tick_params(colors=theme.fg, labelsize=6.5, width=0.6, length=2.5)
    cb.outline.set_edgecolor(theme.fg)
    cb.outline.set_linewidth(0.6)


def _psth_average_figure(cond_data, t_ms, theme, *, baseline, y_label, bin_ms):
    fig = Figure(figsize=(10.8, 8.4), dpi=110, constrained_layout=True)
    fig.patch.set_facecolor(theme.bg)
    fig.set_constrained_layout_pads(w_pad=0.06, h_pad=0.10, hspace=0.10)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.028], height_ratios=[1.0, 1.15])
    ax_line = fig.add_subplot(gs[0, 0])
    ax_heat = fig.add_subplot(gs[1, 0], sharex=ax_line)
    cax = fig.add_subplot(gs[1, 1])
    gs_line_spacer = fig.add_subplot(gs[0, 1]); gs_line_spacer.axis("off")

    _style_axes(ax_line, theme)
    _draw_psth_event_decor(ax_line, t_ms, theme)
    # With a single unit, "average" degenerates to that unit's mean +/- SEM across
    # TRIALS (the population SEM is undefined for n=1).
    single_unit = all(c["mat"].shape[0] == 1 for c in cond_data)
    legend_handles = []
    for c in cond_data:
        mat = c["mat"]
        color = c["color"]
        n_units = mat.shape[0]
        if single_unit:
            raw = c["_per_trial"][0] if c["_per_trial"] else mat
            mean = np.nanmean(raw, axis=0) if raw.ndim == 2 else mat[0]
            n_tr = raw.shape[0] if raw.ndim == 2 else 0
            if n_tr > 1:
                sem = np.nanstd(raw, axis=0, ddof=1) / np.sqrt(n_tr)
                ax_line.fill_between(t_ms, mean - sem, mean + sem, color=color,
                                     alpha=0.25, lw=0, zorder=3)
            ax_line.plot(t_ms, mean, color=color, lw=1.8, zorder=4)
            legend_handles.append(Line2D([0], [0], color=color, lw=1.8,
                                         label=f"{c['label']}  (unit {c['unit_ids'][0]})"))
        else:
            for row in mat:  # faint individual unit means
                ax_line.plot(t_ms, row, color=color, lw=0.5, alpha=0.22, zorder=2)
            mean = np.nanmean(mat, axis=0)
            sem = np.nanstd(mat, axis=0, ddof=1) / np.sqrt(n_units)
            ax_line.fill_between(t_ms, mean - sem, mean + sem, color=color,
                                 alpha=0.25, lw=0, zorder=3)
            ax_line.plot(t_ms, mean, color=color, lw=1.8, zorder=4)
            legend_handles.append(Line2D([0], [0], color=color, lw=1.8,
                                         label=f"{c['label']}  (n={n_units})"))
    ax_line.set_ylabel(y_label)
    ax_line.set_xlim(float(t_ms[0]), float(t_ms[-1]))
    ax_line.tick_params(labelbottom=False)
    if single_unit:
        band_txt = "mean +/- SEM across trials (single unit)"
    else:
        band_txt = "mean +/- SEM across units; faint lines = individual units"
    ax_line.set_title(band_txt, color=theme.muted, fontsize=8, loc="left")
    if legend_handles:
        leg = ax_line.legend(handles=legend_handles, loc="upper right", fontsize=8,
                             frameon=False, handlelength=1.4, labelspacing=0.3)
        for txt in leg.get_texts():
            txt.set_color(theme.fg)

    _draw_psth_heatmap(ax_heat, cax, cond_data, t_ms, theme, baseline=baseline,
                       y_label=y_label)

    n_cond = len(cond_data)
    title_color = "#ffffff" if theme.dark else theme.fg
    fig.suptitle(f"Condition PSTH (average)   |   {n_cond} condition"
                 f"{'s' if n_cond != 1 else ''}   |   bin {bin_ms:g} ms"
                 + ("   |   baseline-subtracted" if baseline else ""),
                 color=title_color, fontsize=12, fontweight="bold", x=0.012, ha="left")
    return fig


def _psth_per_unit_figure(cond_data, t_ms, theme, *, baseline, y_label, bin_ms,
                          max_units=12):
    # Union of unit ids across conditions, preserving order of first appearance.
    ordered_ids = []
    for c in cond_data:
        for uid in c["unit_ids"]:
            if uid not in ordered_ids:
                ordered_ids.append(uid)
    capped = len(ordered_ids) > max_units
    shown_ids = ordered_ids[:max_units]
    n = len(shown_ids)
    if n == 0:
        return _empty_psth_figure(theme, "PSTH results contain no usable units.")

    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))

    # One FLAT gridspec (no nesting) so constrained_layout fills every panel: the
    # unit grid occupies the top ``nrows`` rows; the heatmap spans the bottom block
    # of rows across all columns, with a slim extra column for the colorbar.
    row_h = 1.55          # inches per grid row
    heat_rows = max(2, nrows)  # heatmap height in grid-row units
    heat_h = heat_rows * row_h
    fig_h = nrows * row_h + heat_h * 0.62 + 1.0
    fig = Figure(figsize=(11.8, fig_h), dpi=110, constrained_layout=True)
    fig.patch.set_facecolor(theme.bg)
    fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.09, hspace=0.10, wspace=0.10)
    total_rows = nrows + 1
    height_ratios = [1.0] * nrows + [heat_h * 0.62 / row_h]
    gs = fig.add_gridspec(total_rows, ncols + 1,
                          width_ratios=[1.0] * ncols + [0.04],
                          height_ratios=height_ratios)
    ax_heat = fig.add_subplot(gs[nrows, 0:ncols])
    cax = fig.add_subplot(gs[nrows, ncols])

    # Shared y-limits across small panels for honest comparison.
    all_means = []
    per_unit_cache = {}
    for uid in shown_ids:
        for c in cond_data:
            if uid in c["unit_ids"]:
                row = c["mat"][c["unit_ids"].index(uid)]
                all_means.append(row)
                per_unit_cache.setdefault(uid, []).append((c, row))
    if all_means:
        stacked = np.vstack(all_means)
        ylo = float(np.nanpercentile(stacked, 1))
        yhi = float(np.nanpercentile(stacked, 99))
        pad = 0.08 * (yhi - ylo + 1e-6)
        ylo, yhi = ylo - pad, yhi + pad
    else:
        ylo, yhi = 0.0, 1.0

    axes = []
    for k, uid in enumerate(shown_ids):
        r, cc = divmod(k, ncols)
        ax = fig.add_subplot(gs[r, cc])
        axes.append(ax)
        _style_axes(ax, theme)
        _draw_psth_event_decor(ax, t_ms, theme)
        # Each condition overlaid: this unit's mean +/- SEM across trials.
        for entry in cond_data:
            if uid not in entry["unit_ids"]:
                continue
            j = entry["unit_ids"].index(uid)
            raw = entry["_per_trial"][j]
            mean = entry["mat"][j]
            n_tr = raw.shape[0] if raw.ndim == 2 else 0
            if n_tr > 1:
                sem = np.nanstd(raw, axis=0, ddof=1) / np.sqrt(n_tr)
                ax.fill_between(t_ms, mean - sem, mean + sem, color=entry["color"],
                                alpha=0.22, lw=0, zorder=3)
            ax.plot(t_ms, mean, color=entry["color"], lw=1.3, zorder=4)
        ax.set_title(f"unit {uid}", color=theme.fg, fontsize=8, pad=2)
        ax.set_ylim(ylo, yhi)
        ax.set_xlim(float(t_ms[0]), float(t_ms[-1]))
        if cc != 0:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel(y_label, fontsize=7)
        if r != nrows - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("time (ms)", fontsize=7)
        ax.tick_params(labelsize=6)

    _draw_psth_heatmap(ax_heat, cax, cond_data, t_ms, theme, baseline=baseline,
                       y_label=y_label)

    # A compact condition legend at the figure level.
    legend_handles = [Line2D([0], [0], color=c["color"], lw=1.8, label=c["label"])
                      for c in cond_data]
    leg = fig.legend(handles=legend_handles, loc="upper right",
                     bbox_to_anchor=(0.995, 0.995), fontsize=8, frameon=False,
                     ncol=min(len(legend_handles), 4), handlelength=1.4)
    for txt in leg.get_texts():
        txt.set_color(theme.fg)

    title_color = "#ffffff" if theme.dark else theme.fg
    cap = f"   |   showing {n} of {len(ordered_ids)} units" if capped else ""
    fig.suptitle(f"Condition PSTH (per unit){cap}   |   bin {bin_ms:g} ms"
                 + ("   |   baseline-subtracted" if baseline else ""),
                 color=title_color, fontsize=12, fontweight="bold", x=0.012, ha="left")
    return fig


# --------------------------------------------------------------------------- #
# Public: network / connectivity
# --------------------------------------------------------------------------- #

def _empty_message_figure(theme: _Theme, message: str, *, figsize=(10.0, 6.5)):
    fig = Figure(figsize=figsize, dpi=110)
    fig.patch.set_facecolor(theme.bg)
    ax = fig.add_subplot(111)
    ax.set_facecolor(theme.bg)
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", color=theme.muted,
            fontsize=12, wrap=True)
    return fig


def _signed_norm(values: np.ndarray):
    """Symmetric diverging normalisation centred on 0 from finite data."""
    finite = np.asarray(values)[np.isfinite(values)]
    vmax = float(np.nanpercentile(np.abs(finite), 99)) if finite.size else 1.0
    vmax = max(vmax, 1e-6)
    return mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)


def _matrix_ticklabels(ax, labels, theme: _Theme, *, max_ticks=40):
    """Put unit-id ticks on both axes of a square matrix when there are not too many."""
    n = len(labels)
    if n <= max_ticks:
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(labels, fontsize=5.5, rotation=90, color=theme.fg)
        ax.set_yticklabels(labels, fontsize=5.5, color=theme.fg)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f"unit ({n})", color=theme.fg, fontsize=8)


def _style_colorbar(cb, label, theme: _Theme):
    cb.set_label(label, color=theme.fg, fontsize=8)
    cb.ax.tick_params(colors=theme.fg, labelsize=6.5, width=0.6, length=2.5)
    cb.outline.set_edgecolor(theme.fg)
    cb.outline.set_linewidth(0.6)


def network_figure(results, *, dark=False):
    """Return a matplotlib.figure.Figure rendering population network metrics.

    ``results`` is the engine-computed dict::

        { "units":[int...], "labels":[str...],          # sorted display order
          "corr_matrix": (n,n) Pearson spike-count correlation (sorted),
          "corr_bin_ms": float,
          "population_coupling": (n,) Okun coupling, z-scored (sorted),
          "depths_um": (n,) | None,                     # unit depth, sorted
          "connections": (n,n) | None,                  # signed CCG connection z
          "n_significant": int }

    Composes (left) the sorted spike-count correlation matrix as an RdBu_r heatmap
    centred on 0; (top-right) population coupling either as a coupling-vs-depth
    scatter (when depths are given) or a sorted horizontal bar; and (bottom-right,
    when ``connections`` is present) a second diverging "putative connections (z)"
    matrix, annotated with the number of significant pairs. Degrades gracefully for
    fewer than two units / empty input.
    """
    theme = _Theme(dark)
    if not results or not isinstance(results, dict):
        return _empty_message_figure(theme, "No network results to display.")
    labels = [str(x) for x in results.get("labels", results.get("units", []))]
    corr = np.asarray(results.get("corr_matrix", []), dtype=float)
    n = len(labels)
    if n < 2 or corr.ndim != 2 or corr.shape != (n, n):
        return _empty_message_figure(
            theme, "Select at least two units to compute network metrics.")

    coupling = np.asarray(results.get("population_coupling", []), dtype=float)
    depths = results.get("depths_um", None)
    depths = np.asarray(depths, dtype=float) if depths is not None else None
    connections = results.get("connections", None)
    has_conn = connections is not None and np.asarray(connections).shape == (n, n)
    if has_conn:
        connections = np.asarray(connections, dtype=float)
    n_sig = int(results.get("n_significant", 0))
    corr_bin = float(results.get("corr_bin_ms", 0.0) or 0.0)

    fig = Figure(figsize=(12.8, 6.8), dpi=110, constrained_layout=True)
    fig.patch.set_facecolor(theme.bg)
    fig.set_constrained_layout_pads(w_pad=0.07, h_pad=0.09, wspace=0.06, hspace=0.12)

    # Left: correlation matrix (+ slim colorbar). Right block: coupling on top,
    # connections matrix below (or coupling spanning the whole right when absent).
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 0.03, 1.05],
                          height_ratios=[1.0, 1.0])
    ax_corr = fig.add_subplot(gs[:, 0])
    cax_corr = fig.add_subplot(gs[:, 1])

    # (a) Correlation matrix.
    norm = _signed_norm(corr[~np.eye(n, dtype=bool)] if n > 1 else corr)
    im = ax_corr.imshow(corr, cmap="RdBu_r", norm=norm, interpolation="nearest")
    _style_axes(ax_corr, theme)
    ax_corr.spines[["left", "bottom"]].set_visible(True)
    _matrix_ticklabels(ax_corr, labels, theme)
    bin_txt = f" ({corr_bin:g} ms bins)" if corr_bin else ""
    ax_corr.set_title(f"Spike-count correlation{bin_txt}", color=theme.fg, fontsize=9.5)
    cb = fig.colorbar(im, cax=cax_corr)
    _style_colorbar(cb, "correlation (r)", theme)

    if has_conn:
        ax_couple = fig.add_subplot(gs[0, 2])
        ax_conn = fig.add_subplot(gs[1, 2])
    else:
        ax_couple = fig.add_subplot(gs[:, 2])
        ax_conn = None

    # (b) Population coupling.
    _style_axes(ax_couple, theme)
    if coupling.size == n and depths is not None and depths.size == n:
        ax_couple.scatter(coupling, depths, s=26, c=coupling, cmap="viridis",
                          edgecolors=theme.fg, linewidths=0.4, zorder=3)
        ax_couple.axvline(0.0, color=theme.muted, ls="--", lw=0.7, zorder=1)
        ax_couple.invert_yaxis()  # depth descending (deeper lower)
        ax_couple.set_xlabel("population coupling (z)", fontsize=8)
        ax_couple.set_ylabel("depth (um)", fontsize=8)
        ax_couple.set_title("Population coupling vs depth", color=theme.fg, fontsize=9.5)
    elif coupling.size == n:
        order = np.argsort(coupling)
        ypos = np.arange(n)
        colors = [theme.accent if v >= 0 else _UNIT_COLORS[1] for v in coupling[order]]
        ax_couple.barh(ypos, coupling[order], color=colors, edgecolor="none", zorder=3)
        ax_couple.axvline(0.0, color=theme.muted, ls="-", lw=0.7, zorder=2)
        if n <= 40:
            ax_couple.set_yticks(ypos)
            ax_couple.set_yticklabels([labels[i] for i in order], fontsize=5.5,
                                      color=theme.fg)
        else:
            ax_couple.set_yticks([])
        ax_couple.set_xlabel("population coupling (z)", fontsize=8)
        ax_couple.set_ylim(-0.6, n - 0.4)
        ax_couple.set_title("Population coupling", color=theme.fg, fontsize=9.5)
    else:
        ax_couple.axis("off")
        ax_couple.text(0.5, 0.5, "coupling unavailable", transform=ax_couple.transAxes,
                       ha="center", va="center", color=theme.muted, fontsize=9)

    # (c) Putative connections matrix.
    if ax_conn is not None:
        cnorm = _signed_norm(connections[~np.eye(n, dtype=bool)])
        im2 = ax_conn.imshow(connections, cmap="PuOr_r", norm=cnorm,
                             interpolation="nearest")
        _style_axes(ax_conn, theme)
        ax_conn.spines[["left", "bottom"]].set_visible(True)
        _matrix_ticklabels(ax_conn, labels, theme)
        ax_conn.set_title(f"Putative connections (z)   |   {n_sig} significant",
                          color=theme.fg, fontsize=9.5)
        cb2 = fig.colorbar(im2, ax=ax_conn, fraction=0.046, pad=0.03)
        _style_colorbar(cb2, "connection (z)", theme)

    title_color = "#ffffff" if theme.dark else theme.fg
    fig.suptitle(f"Network metrics   |   {n} units"
                 + (f"   |   {n_sig} significant connections" if has_conn else ""),
                 color=title_color, fontsize=12, fontweight="bold", x=0.012, ha="left")
    return fig


# --------------------------------------------------------------------------- #
# Public: C4 cell-type classification
# --------------------------------------------------------------------------- #

def _c4_colors():
    """Return a {class_name: hex} map from npyx.c4 COLORS_DICT (0-255 RGB -> hex).

    ``unlabelled`` maps to the C4 ``background`` colour; any class missing from the
    C4 dict falls back to Okabe-Ito so the figure never crashes on an unknown label.
    """
    out = {}
    try:
        from npyx.c4.dataset_init import COLORS_DICT  # type: ignore
        for name, rgb in COLORS_DICT.items():
            r, g, b = [int(v) for v in rgb[:3]]
            out[str(name)] = "#{:02x}{:02x}{:02x}".format(r, g, b)
    except Exception:
        out = {}
    if "background" in out:
        out.setdefault("unlabelled", out["background"])
    return out


def _c4_color_for(name: str, palette: dict, fallback_idx: int) -> str:
    if name in palette:
        return palette[name]
    low = str(name).lower()
    if low in ("unlabelled", "unlabeled", "unknown", "none", "background"):
        return palette.get("background", "#999999")
    return _UNIT_COLORS[fallback_idx % len(_UNIT_COLORS)]


def c4_figure(results, *, dark=False):
    """Return a matplotlib.figure.Figure rendering C4 cell-type predictions.

    ``results`` is the C4-runner dict::

        { "units":[int...], "predicted_type":[str...], "confidence":[float...],
          "probabilities": (n_units, n_classes), "class_names":[str...],
          "model_type": str, "error": str|None }

    Composes (top) a per-unit confidence bar coloured by predicted cell type with
    unit id + label annotated, (bottom-left) a units x classes probability heatmap
    (0..1) with a colorbar, and (bottom-right) a colour->cell-type legend. Cell-type
    colours come from ``npyx.c4.dataset_init.COLORS_DICT`` (Okabe-Ito fallback for
    unknown classes). Renders a clean centred message when ``error`` is set or the
    input is empty, and carries a discreet note that the C4 model is cerebellum-
    trained (labels are indicative on non-cerebellar regions).
    """
    theme = _Theme(dark)
    err = (results or {}).get("error") if isinstance(results, dict) else "no results"
    units = list((results or {}).get("units", [])) if isinstance(results, dict) else []
    if err:
        return _empty_message_figure(theme, f"C4 model not available:\n{err}")
    if not units:
        return _empty_message_figure(theme, "No C4 predictions to display.")

    predicted = [str(x) for x in results.get("predicted_type", [])]
    confidence = np.asarray(results.get("confidence", []), dtype=float)
    probs = np.asarray(results.get("probabilities", []), dtype=float)
    class_names = [str(c) for c in results.get("class_names", [])]
    model_type = str(results.get("model_type", "C4"))
    n = len(units)
    if confidence.size != n:
        confidence = np.full(n, np.nan)

    palette = _c4_colors()
    class_color = {c: _c4_color_for(c, palette, i) for i, c in enumerate(class_names)}
    # Ensure predicted labels not in class_names still get a colour.
    for i, p in enumerate(predicted):
        class_color.setdefault(p, _c4_color_for(p, palette, len(class_names) + i))
    bar_colors = [class_color.get(p, theme.muted) for p in predicted]

    fig = Figure(figsize=(12.2, 7.6), dpi=110, constrained_layout=True)
    fig.patch.set_facecolor(theme.bg)
    fig.set_constrained_layout_pads(w_pad=0.07, h_pad=0.10, wspace=0.06, hspace=0.12)
    # Row 2 is a slim caption strip so the note never collides with the heatmap's
    # rotated x-tick labels.
    gs = fig.add_gridspec(3, 3, width_ratios=[1.0, 0.03, 0.32],
                          height_ratios=[0.9, 1.0, 0.06])
    ax_caption = fig.add_subplot(gs[2, :])
    ax_caption.axis("off")
    ax_caption.text(0.0, 0.2,
                    "Note: the C4 model is cerebellum-trained; predictions are "
                    "indicative only on non-cerebellar regions.",
                    color=theme.muted, fontsize=7.5, ha="left", va="center",
                    transform=ax_caption.transAxes)

    # (a) Confidence bar coloured by predicted type.
    ax_bar = fig.add_subplot(gs[0, :])
    _style_axes(ax_bar, theme)
    xpos = np.arange(n)
    ax_bar.bar(xpos, confidence, color=bar_colors, edgecolor="none", zorder=3,
               width=0.78)
    ax_bar.set_ylim(0, 1.0)
    ax_bar.set_xlim(-0.6, n - 0.4)
    ax_bar.set_ylabel("confidence", fontsize=8)
    ax_bar.set_xticks(xpos)
    ax_bar.set_xticklabels([f"u{u}" for u in units], fontsize=6.5, color=theme.fg)
    ax_bar.set_title("Predicted cell type & confidence", color=theme.fg, fontsize=9.5)
    for x, p, c in zip(xpos, predicted, confidence):
        if np.isfinite(c):
            ax_bar.text(x, min(c + 0.02, 0.98), p, ha="center", va="bottom",
                        fontsize=6, color=theme.fg, rotation=90)

    # (b) Probability heatmap (units x classes).
    ax_heat = fig.add_subplot(gs[1, 0])
    cax = fig.add_subplot(gs[1, 1])
    if probs.ndim == 2 and probs.shape[0] == n and probs.shape[1] == len(class_names) \
            and len(class_names) > 0:
        im = ax_heat.imshow(probs, aspect="auto", cmap="magma",
                            norm=mcolors.Normalize(0, 1), interpolation="nearest")
        _style_axes(ax_heat, theme)
        ax_heat.set_xticks(np.arange(len(class_names)))
        ax_heat.set_xticklabels(class_names, fontsize=7, rotation=35, ha="right",
                                color=theme.fg)
        ax_heat.set_yticks(np.arange(n))
        ax_heat.set_yticklabels([f"u{u}" for u in units], fontsize=6.5, color=theme.fg)
        ax_heat.set_title("Class probabilities", color=theme.fg, fontsize=9.5)
        cb = fig.colorbar(im, cax=cax)
        _style_colorbar(cb, "probability", theme)
    else:
        ax_heat.axis("off")
        cax.axis("off")
        ax_heat.text(0.5, 0.5, "probabilities unavailable", transform=ax_heat.transAxes,
                     ha="center", va="center", color=theme.muted, fontsize=9)

    # (c) Colour -> cell-type legend.
    ax_leg = fig.add_subplot(gs[1, 2])
    ax_leg.axis("off")
    legend_classes = class_names if class_names else list(dict.fromkeys(predicted))
    handles = [mpatches.Patch(facecolor=class_color.get(c, theme.muted),
                              edgecolor="none", label=c) for c in legend_classes]
    if handles:
        leg = ax_leg.legend(handles=handles, loc="center left", frameon=False,
                            fontsize=8, handlelength=1.0, title="cell type")
        leg.get_title().set_color(theme.fg)
        leg.get_title().set_fontsize(8.5)
        for txt in leg.get_texts():
            txt.set_color(theme.fg)

    title_color = "#ffffff" if theme.dark else theme.fg
    fig.suptitle(f"C4 cell-type classification   |   {n} units   |   model: {model_type}",
                 color=title_color, fontsize=12, fontweight="bold", x=0.012, ha="left")
    return fig


# --------------------------------------------------------------------------- #
# Public: single-unit waveform + ACG card (one per unit, for PDF batches)
# --------------------------------------------------------------------------- #

def unit_waveform_acg_figure(dataset, unit, *, acg_bin_ms=0.5, acg_win_ms=100.0,
                             dark=False):
    """Return a clean two-panel card for ONE unit, sized for a PDF page.

    Left panel: the mean waveform laid out on the real probe geometry (peak channel
    plus its physical neighbours from ``channel_positions``), with a uV/ms scalebar
    and a shaded +/-SEM band built from real spike snippets read off the AP binary.
    This reuses :func:`_draw_waveform_geometry` (the exact drawer used by
    ``unit_basics_figure``), so a unit with no template waveform degrades to a clean
    "no template waveform" note while the ACG is still drawn.

    Right panel: the Hertz-normalised autocorrelogram with a shaded 2 ms refractory
    band and a dashed zero-lag line, via :func:`_draw_acg` (npyx ``plot_acg`` path
    with an engine-``correlogram`` fallback at
    ``bin_ms=acg_bin_ms, win_ms=acg_win_ms``).

    The bold figure title reads
    ``Unit {id} | peak ch {ch} | {n_spikes} spikes | {mean_fr:.1f} Hz``.

    The figure uses ``constrained_layout`` (no overlapping titles), top/right
    despining, and the shared white / dark theme. It is fully self-contained so the
    caller can build one per unit in a batch loop and write each to a PDF page.
    """
    theme = _Theme(dark)
    unit = int(unit)

    fig = Figure(figsize=(8.5, 4.2), dpi=110, constrained_layout=True)
    fig.patch.set_facecolor(theme.bg)
    fig.set_constrained_layout_pads(w_pad=0.07, h_pad=0.12, wspace=0.07)
    # Reserve a top band for the bold title and side margins so neither the title
    # text nor the ACG's right spine is clipped on this short single-row canvas
    # (constrained_layout does not reserve suptitle space on its own here).
    try:
        fig.get_layout_engine().set(rect=(0.012, 0.0, 0.976, 0.88))
    except Exception:
        pass

    # Slightly favour the waveform panel; it carries the multichannel geometry.
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0])
    ax_wf = fig.add_subplot(gs[0, 0])
    ax_acg = fig.add_subplot(gs[0, 1])

    # Left: waveform on probe geometry (single unit -> mean +/- SEM, peak highlight).
    _draw_waveform_geometry(ax_wf, dataset, [unit], theme)
    # Right: Hertz ACG with refractory band + zero-lag line.
    _draw_acg(ax_acg, dataset, [unit], theme, bin_ms=acg_bin_ms, win_ms=acg_win_ms)
    # npyx's plot_acg (prettify) resets the figure-level facecolor to white; re-assert
    # the theme background so the dark look (and a white title) survive.
    fig.patch.set_facecolor(theme.bg)

    # Title metrics.
    t = dataset.unit_spike_times_s(unit)
    dur = _session_duration_s(dataset)
    n_spk = int(t.size)
    mean_fr = (n_spk / dur) if dur > 0 else 0.0
    _, peak_ch = _peak_channel(dataset, unit)
    peak_str = "n/a" if peak_ch is None else str(peak_ch)
    title_color = "#ffffff" if theme.dark else theme.fg
    fig.suptitle(
        f"Unit {unit}  |  peak ch {peak_str}  |  {n_spk:,} spikes  |  {mean_fr:.1f} Hz",
        color=title_color, fontsize=11, fontweight="bold", x=0.012, y=0.955,
        ha="left", va="center",
    )
    return fig


# --------------------------------------------------------------------------- #
# Public: Bombcell region-specific cell-type classification
# --------------------------------------------------------------------------- #

# Fixed, distinct colours per Bombcell class. "Unknown" is always grey. Chosen from
# the same colorblind-aware family as ``_UNIT_COLORS`` so the cards read on white
# and dark, and so the strip + scatter share one legend.
_BOMBCELL_GREY = "#9aa3b0"
_BOMBCELL_CORTEX_COLORS = {
    "Wide-spiking": "#0072B2",   # blue  (broad / regular-spiking, putative pyramidal)
    "Narrow-spiking": "#D55E00",  # vermillion (fast-spiking, putative interneuron)
    "Unknown": _BOMBCELL_GREY,
}
_BOMBCELL_STRIATUM_COLORS = {
    "MSN": "#0072B2",   # blue  - medium spiny neuron
    "FSI": "#D55E00",   # vermillion - fast-spiking interneuron
    "TAN": "#009E73",   # bluish green - tonically active (cholinergic)
    "UIN": "#CC79A7",   # reddish purple - unidentified interneuron
    "Unknown": _BOMBCELL_GREY,
}


def _bombcell_palette(region: str) -> dict:
    return (_BOMBCELL_STRIATUM_COLORS if str(region).lower() == "striatum"
            else _BOMBCELL_CORTEX_COLORS)


def _bombcell_color_for(label: str, palette: dict, fallback_idx: int) -> str:
    if label in palette:
        return palette[label]
    if str(label).lower() in ("unknown", "unlabelled", "unlabeled", "none", "na", "n/a"):
        return _BOMBCELL_GREY
    return _UNIT_COLORS[fallback_idx % len(_UNIT_COLORS)]


def _bombcell_legend(ax_leg, class_names, class_color, theme: _Theme, *, title="cell type"):
    ax_leg.axis("off")
    handles = [mpatches.Patch(facecolor=class_color.get(c, _BOMBCELL_GREY),
                              edgecolor="none", label=c) for c in class_names]
    if not handles:
        return
    leg = ax_leg.legend(handles=handles, loc="center left", frameon=False,
                        fontsize=8, handlelength=1.0, title=title,
                        labelspacing=0.5, borderaxespad=0.1)
    leg.get_title().set_color(theme.fg)
    leg.get_title().set_fontsize(8.5)
    for txt in leg.get_texts():
        txt.set_color(theme.fg)


def _bombcell_strip(ax, units, predicted, class_color, theme: _Theme):
    """Per-unit coloured strip (one bar per unit) annotated with id + predicted label."""
    _style_axes(ax, theme)
    n = len(units)
    xpos = np.arange(n)
    colors = [class_color.get(p, _BOMBCELL_GREY) for p in predicted]
    ax.bar(xpos, np.ones(n), color=colors, edgecolor=theme.bg, linewidth=0.6,
           width=0.92, zorder=3)
    ax.set_ylim(0, 1.0)
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"u{u}" for u in units], fontsize=6.5, color=theme.fg,
                       rotation=90)
    ax.set_title("Units by predicted cell type", color=theme.fg, fontsize=9.5)
    # Predicted label printed inside each bar (vertical, dark text reads on every hue).
    for x, p in zip(xpos, predicted):
        ax.text(x, 0.5, p, ha="center", va="center", fontsize=6,
                color="#15181d", rotation=90, zorder=5)


def _bombcell_scatter(ax, x, y, predicted, class_color, theme: _Theme, *,
                      xlabel, ylabel, vlines=(), hlines=(), log_y=False, title=""):
    """One feature scatter coloured by predicted type, with dashed threshold lines."""
    _style_axes(ax, theme)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    colors = [class_color.get(p, _BOMBCELL_GREY) for p in predicted]
    if log_y:
        ax.set_yscale("log")
    # Threshold guide lines first (behind the points).
    for xv, lab in vlines:
        ax.axvline(xv, color=theme.muted, ls="--", lw=0.9, alpha=0.85, zorder=1)
        ax.text(xv, 1.0, f" {lab}", transform=ax.get_xaxis_transform(),
                ha="left", va="bottom", fontsize=6, color=theme.muted, rotation=0)
    for yv, lab in hlines:
        ax.axhline(yv, color=theme.muted, ls="--", lw=0.9, alpha=0.85, zorder=1)
        ax.text(1.0, yv, f"{lab} ", transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=6, color=theme.muted)
    ax.scatter(x, y, s=34, c=colors, edgecolors=theme.fg, linewidths=0.45,
               alpha=0.92, zorder=3)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    if title:
        ax.set_title(title, color=theme.fg, fontsize=9.5)


def bombcell_celltype_figure(results, *, dark=False):
    """Return a matplotlib.figure.Figure of Bombcell region-specific cell types.

    ``results`` is the Bombcell-runner dict::

        { "units":[int...], "predicted_type":[str...], "class_names":[str...],
          "region": "cortex"|"striatum", "method":"bombcell",
          "metrics": {"waveform_duration_us":[float...],
                      "post_spike_suppression_ms":[float...],
                      "prop_long_isi":[float...], "firing_rate_hz":[float...]},
          "skipped_units":[int...], "error": str|None }

    All ``metrics`` arrays are aligned element-wise to ``units`` / ``predicted_type``.

    Composes (constrained_layout, GridSpec):

    - (a) a per-unit coloured strip: one bar per unit tinted by predicted cell type,
      annotated with the unit id and its label, plus a colour -> cell-type legend.
    - (b) the Bombcell feature scatter coloured by type:
        * ``region == "cortex"``: ``waveform_duration_us`` (x) vs ``firing_rate_hz``
          (y, log scale) with a dashed vertical line at the 400 us
          Wide-vs-Narrow threshold.
        * ``region == "striatum"``: two feature subpanels,
          [``waveform_duration_us`` vs ``post_spike_suppression_ms``] and
          [``waveform_duration_us`` vs ``prop_long_isi``], with dashed thresholds at
          400 us, 40 ms and 0.1 respectively.

    Fixed distinct colours per class ("Unknown" = grey). The title reads
    ``Bombcell cell types ({region}) | {N} units`` and a discreet caption notes the
    threshold-based MATLAB ``classifyCells`` provenance. When ``results["error"]`` is
    set or there are no units, a clean centred message is rendered instead.
    """
    theme = _Theme(dark)
    if not results or not isinstance(results, dict):
        return _empty_message_figure(theme, "No Bombcell results to display.")
    err = results.get("error")
    if err:
        return _empty_message_figure(theme, f"Bombcell classification failed:\n{err}")

    units = [int(u) for u in results.get("units", [])]
    if not units:
        return _empty_message_figure(theme, "No Bombcell predictions to display.")

    region = str(results.get("region", "cortex")).lower()
    is_striatum = region == "striatum"
    predicted = [str(p) for p in results.get("predicted_type", [])]
    if len(predicted) != len(units):
        predicted = (predicted + ["Unknown"] * len(units))[:len(units)]
    n = len(units)

    metrics = results.get("metrics", {}) or {}

    def _metric(name):
        arr = np.asarray(metrics.get(name, []), dtype=float)
        if arr.size != n:
            arr = np.full(n, np.nan)
        return arr

    dur_us = _metric("waveform_duration_us")
    pss_ms = _metric("post_spike_suppression_ms")
    long_isi = _metric("prop_long_isi")
    fr_hz = _metric("firing_rate_hz")

    palette = _bombcell_palette(region)
    class_names = [str(c) for c in results.get("class_names", [])]
    if not class_names:
        # Preserve palette order, then append any extra predicted labels.
        class_names = list(palette.keys())
    class_color = {c: _bombcell_color_for(c, palette, i)
                   for i, c in enumerate(class_names)}
    for i, p in enumerate(predicted):
        class_color.setdefault(p, _bombcell_color_for(p, palette, len(class_names) + i))
    # Legend lists palette classes first, then any predicted label not already shown.
    legend_classes = list(class_names)
    for p in predicted:
        if p not in legend_classes:
            legend_classes.append(p)

    fig = Figure(figsize=(12.4, 7.4), dpi=110, constrained_layout=True)
    fig.patch.set_facecolor(theme.bg)
    fig.set_constrained_layout_pads(w_pad=0.07, h_pad=0.12, wspace=0.07, hspace=0.16)

    # Row 0: per-unit strip (+ legend column). Row 1: feature scatter(s).
    # Row 2: a slim caption strip kept clear of the scatter axis labels.
    gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 0.30],
                          height_ratios=[0.62, 1.0, 0.07])

    # (a) per-unit strip + legend.
    ax_strip = fig.add_subplot(gs[0, 0])
    _bombcell_strip(ax_strip, units, predicted, class_color, theme)
    ax_leg = fig.add_subplot(gs[0, 1])
    _bombcell_legend(ax_leg, legend_classes, class_color, theme)

    # (b) feature scatter(s).
    if is_striatum:
        gs_feat = gs[1, :].subgridspec(1, 2, wspace=0.28)
        ax_f1 = fig.add_subplot(gs_feat[0, 0])
        ax_f2 = fig.add_subplot(gs_feat[0, 1])
        _bombcell_scatter(
            ax_f1, dur_us, pss_ms, predicted, class_color, theme,
            xlabel="waveform duration (us)", ylabel="post-spike suppression (ms)",
            vlines=[(400.0, "400 us")], hlines=[(40.0, "40 ms")],
            title="Duration vs post-spike suppression")
        _bombcell_scatter(
            ax_f2, dur_us, long_isi, predicted, class_color, theme,
            xlabel="waveform duration (us)", ylabel="proportion long ISI",
            vlines=[(400.0, "400 us")], hlines=[(0.1, "0.1")],
            title="Duration vs long-ISI proportion")
    else:
        ax_feat = fig.add_subplot(gs[1, :])
        _bombcell_scatter(
            ax_feat, dur_us, fr_hz, predicted, class_color, theme,
            xlabel="waveform duration (us)", ylabel="firing rate (Hz)",
            vlines=[(400.0, "400 us (Wide / Narrow)")], log_y=True,
            title="Waveform duration vs firing rate")

    # (c) discreet caption strip.
    skipped = list(results.get("skipped_units", []) or [])
    ax_cap = fig.add_subplot(gs[2, :])
    ax_cap.axis("off")
    cap = ("Threshold-based region-specific classes (Bombcell / MATLAB classifyCells)."
           + (f"  {len(skipped)} unit(s) skipped." if skipped else ""))
    ax_cap.text(0.0, 0.2, cap, color=theme.muted, fontsize=7.5, ha="left",
                va="center", transform=ax_cap.transAxes)

    title_color = "#ffffff" if theme.dark else theme.fg
    fig.suptitle(f"Bombcell cell types ({region})   |   {n} units",
                 color=title_color, fontsize=12, fontweight="bold", x=0.012, ha="left")
    return fig
