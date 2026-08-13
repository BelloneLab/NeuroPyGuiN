"""Match histology slices to Allen CCF planes -> ``histology_ccf``.

Native port of ``match_histology_atlas.m``. The interactive camera/scroll model
of the MATLAB 3D view is replaced (in the GUI) by an intuitive AP-position +
tilt model, but the underlying plane sampling is identical
(:meth:`atlas.AllenCCFAtlas.grab_atlas_slice`), so the resulting ``histology_ccf``
is compatible with the rest of the pipeline.

This module provides the non-interactive helpers: building the full-resolution
``histology_ccf`` for every slice and rendering an atlas slice (TV / AV-overlay /
TV-AV colour) for display on the canvas.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Sequence

import numpy as np
from scipy import ndimage as ndi

from .atlas import AllenCCFAtlas, coronal_camera_vector, coronal_slice_point
from . import acceleration


def build_histology_ccf(
    atlas: AllenCCFAtlas,
    slice_specs: Sequence[Dict[str, np.ndarray]],
    spacing: int = 1,
) -> List[Dict[str, np.ndarray]]:
    """Grab full-resolution CCF planes for every matched slice.

    ``slice_specs[i]`` = ``{"slice_point": (3,), "camera_vector": (3,)}`` (the
    plane chosen for histology slice ``i``). Returns the ``histology_ccf`` list
    of dicts ready for :func:`io_formats.save_histology_ccf`.
    """
    out: List[Dict[str, np.ndarray]] = []
    for spec in slice_specs:
        out.append(atlas.grab_atlas_slice(spec["slice_point"], spec["camera_vector"], spacing))
    return out


def _normalize_uint8(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Scale ``arr`` from the ``[lo, hi]`` range into ``[0, 255]`` uint8.

    NaNs are treated as 0 and values outside ``[lo, hi]`` are clipped. The
    ``hi - lo`` divisor is floored at a tiny epsilon to avoid division by zero.
    """
    a = np.nan_to_num(arr.astype(np.float64), nan=0.0)
    a = np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1)
    return (a * 255).astype(np.uint8)


def render_atlas_slice(
    slice_dict: Dict[str, np.ndarray],
    atlas: AllenCCFAtlas,
    mode: str = "TV",
    tv_max: float = 516.0,
) -> np.ndarray:
    """Render an atlas slice as an RGB uint8 image for display.

    ``mode`` is one of ``"TV"`` (template grayscale), ``"AV"`` (template with red
    region boundaries) or ``"TV-AV"`` (regions coloured by the Allen palette).
    """
    tv = slice_dict["tv_slices"]
    av = slice_dict["av_slices"]
    brain = np.isfinite(tv)

    if mode == "TV":
        g = _normalize_uint8(tv, 0, tv_max)
        rgb = np.dstack([g, g, g])
    elif mode == "AV":
        g = _normalize_uint8(tv, 0, tv_max)
        rgb = np.dstack([g, g, g])
        bound = _av_boundaries(av)
        rgb[bound] = [255, 40, 40]
    else:  # TV-AV
        rgb = np.zeros((*av.shape, 3), dtype=np.uint8)
        idx = np.nan_to_num(av, nan=0.0).astype(np.int64)
        valid = brain & (idx >= 1) & (idx <= len(atlas.structure_tree))
        cmap = (atlas.st_rgb * 255).astype(np.uint8)
        flat_idx = idx[valid] - 1
        rgb[valid] = cmap[flat_idx]
    rgb[~brain] = 0
    return rgb


def _av_boundaries(av: np.ndarray) -> np.ndarray:
    """Return a boolean mask of region boundaries in an annotation slice.

    A 2x2 uniform filter is rounded and compared against the original label
    values; pixels where the smoothed value differs sit on a region edge.
    """
    av0 = np.nan_to_num(av, nan=0.0)
    smoothed = np.round(ndi.uniform_filter(av0, size=2))
    return smoothed != av0


# ---------------------------------------------------------------------------
# Lightweight shape-based atlas matching
# ---------------------------------------------------------------------------

def _largest_component(mask: np.ndarray) -> np.ndarray:
    labels, n_labels = ndi.label(np.asarray(mask, bool))
    if n_labels <= 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = ndi.sum(np.ones_like(labels, dtype=float), labels, index=np.arange(1, n_labels + 1))
    return labels == int(np.argmax(sizes) + 1)


def _major_components(mask: np.ndarray, min_fraction: float = 0.04, max_components: int = 6) -> np.ndarray:
    """Keep major tissue islands, not only the largest one.

    Coronal slices can split around the midline during mounting, and one side can
    be incomplete. Keeping several large components makes the matcher use the
    observed tissue instead of accidentally throwing away a hemisphere.
    """
    labels, n_labels = ndi.label(np.asarray(mask, bool))
    if n_labels <= 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.asarray(
        ndi.sum(np.ones_like(labels, dtype=float), labels, index=np.arange(1, n_labels + 1)),
        dtype=float,
    )
    if sizes.size == 0 or sizes.max() <= 0:
        return np.zeros_like(mask, dtype=bool)
    order = np.argsort(sizes)[::-1]
    keep = []
    min_size = max(32.0, float(sizes.max()) * float(min_fraction))
    for comp_idx in order[:max_components]:
        if sizes[comp_idx] >= min_size:
            keep.append(int(comp_idx) + 1)
    return np.isin(labels, keep)


def histology_shape_mask(image: np.ndarray) -> np.ndarray:
    """Estimate the tissue silhouette from a histology slice image.

    The matcher needs only a coarse filled outline. For RGB images this favors
    blue/cyan tissue signal over red background fluorescence; for grayscale it
    falls back to a robust intensity threshold.
    """
    arr = np.asarray(image)
    if arr.ndim < 2 or arr.size == 0:
        return np.zeros(arr.shape[:2], dtype=bool)
    if arr.ndim == 3:
        rgb = arr[..., :3].astype(np.float64)
        scale = 255.0 if rgb.max(initial=0) > 1.5 else 1.0
        rgb = rgb / scale
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        value = np.max(rgb, axis=2)
        blue_signal = np.clip(b - 0.45 * r - 0.20 * g, 0, None)
        chroma = np.max(rgb, axis=2) - np.min(rgb, axis=2)
        score = 0.70 * blue_signal + 0.20 * chroma + 0.10 * value
    else:
        score = arr.astype(np.float64)
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    finite = score[np.isfinite(score)]
    if finite.size == 0:
        return np.zeros(arr.shape[:2], dtype=bool)
    hi = float(np.percentile(finite, 99.0))
    lo = float(np.percentile(finite, 10.0))
    if hi <= lo:
        hi = float(finite.max())
        lo = float(finite.min())
    norm = np.clip((score - lo) / max(hi - lo, 1e-9), 0, 1)
    threshold = max(0.08, float(np.percentile(norm[norm > 0], 62.0)) if np.any(norm > 0) else 0.5)
    threshold = min(threshold, 0.85)
    mask = norm >= threshold
    min_side = max(3, int(min(mask.shape) * 0.012))
    mask = ndi.binary_opening(mask, iterations=1)
    mask = ndi.binary_closing(mask, iterations=min_side)
    mask = _major_components(mask)
    mask = ndi.binary_fill_holes(mask)
    mask = ndi.binary_closing(mask, iterations=max(1, min_side // 2))
    return _major_components(mask)


def atlas_shape_mask(tv_slice: np.ndarray) -> np.ndarray:
    """Estimate the atlas brain silhouette from a template slice.

    AP_histology-style atlas slices can be finite across the whole rectangular
    sampling plane even where the visible template is pure zero. Using
    ``np.isfinite`` alone therefore makes AP 1 look like a valid full rectangle.
    The matcher wants the visible positive template tissue instead.
    """
    tv = np.nan_to_num(np.asarray(tv_slice, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if tv.ndim != 2 or tv.size == 0:
        return np.zeros(tv.shape[:2], dtype=bool)
    positive = tv[tv > 0]
    if positive.size == 0:
        return np.zeros(tv.shape, dtype=bool)
    max_value = float(positive.max())
    if max_value <= 5.0:
        threshold = max_value * 0.10
    else:
        threshold = max(5.0, max_value * 0.02)
    mask = tv > threshold
    if mask.sum() < 64 and positive.size >= 64:
        # Synthetic/unit-test atlases may use small 0/1 masks; keep them valid.
        mask = tv > 0
    if not mask.any():
        return np.zeros(tv.shape, dtype=bool)
    min_side = max(1, int(min(mask.shape) * 0.006))
    mask = ndi.binary_closing(mask, iterations=min_side)
    mask = ndi.binary_fill_holes(mask)
    return _major_components(mask, min_fraction=0.02, max_components=3)


def _histology_signal(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3:
        rgb = arr[..., :3].astype(np.float64)
        scale = 255.0 if rgb.max(initial=0) > 1.5 else 1.0
        rgb = rgb / scale
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        value = np.max(rgb, axis=2)
        blue_signal = np.clip(b - 0.45 * r - 0.20 * g, 0, None)
        chroma = np.max(rgb, axis=2) - np.min(rgb, axis=2)
        signal = 0.72 * blue_signal + 0.18 * chroma + 0.10 * value
    else:
        signal = arr.astype(np.float64)
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    finite = signal[np.isfinite(signal)]
    if finite.size == 0:
        return np.zeros(signal.shape[:2], dtype=float)
    lo, hi = np.percentile(finite, [5.0, 99.5])
    if hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    return np.clip((signal - lo) / max(float(hi - lo), 1e-9), 0, 1)


def _internal_edge_mask(signal: np.ndarray, tissue_mask: np.ndarray, percentile: float = 87.0) -> np.ndarray:
    signal = np.nan_to_num(np.asarray(signal, dtype=float), nan=0.0)
    tissue_mask = np.asarray(tissue_mask, bool)
    if signal.ndim != 2 or not tissue_mask.any():
        return np.zeros(signal.shape[:2], dtype=bool)
    erode_iter = max(2, int(round(min(signal.shape) * 0.025)))
    inner = ndi.binary_erosion(tissue_mask, iterations=erode_iter)
    if inner.sum() < 64:
        inner = tissue_mask
    smooth = ndi.gaussian_filter(signal, sigma=max(0.8, min(signal.shape) * 0.0025))
    grad = np.hypot(ndi.sobel(smooth, axis=0), ndi.sobel(smooth, axis=1))
    values = grad[inner & np.isfinite(grad)]
    if values.size < 32 or float(values.max()) <= 0:
        return np.zeros(signal.shape, dtype=bool)
    threshold = float(np.percentile(values, percentile))
    edges = (grad >= threshold) & inner
    edges = ndi.binary_opening(edges, iterations=1)
    return np.asarray(edges, bool)


def _histology_void_mask(signal: np.ndarray, tissue_mask: np.ndarray) -> np.ndarray:
    tissue_mask = np.asarray(tissue_mask, bool)
    if not tissue_mask.any():
        return np.zeros(signal.shape[:2], dtype=bool)
    outer = ndi.binary_closing(tissue_mask, iterations=max(2, int(min(tissue_mask.shape) * 0.018)))
    outer = ndi.binary_fill_holes(outer)
    inner = ndi.binary_erosion(outer, iterations=max(2, int(min(tissue_mask.shape) * 0.018)))
    values = signal[outer & tissue_mask]
    if values.size < 32:
        return np.zeros(signal.shape[:2], dtype=bool)
    low = float(np.percentile(values, 18.0))
    void = outer & inner & (signal <= low)
    return _major_components(void, min_fraction=0.015, max_components=6)


def _atlas_internal_edges(tv_slice: np.ndarray, av_slice: np.ndarray, atlas_mask: np.ndarray) -> np.ndarray:
    atlas_mask = np.asarray(atlas_mask, bool)
    if not atlas_mask.any():
        return np.zeros(atlas_mask.shape, dtype=bool)
    tv = np.nan_to_num(np.asarray(tv_slice, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    positive = tv[atlas_mask & (tv > 0)]
    if positive.size:
        lo, hi = np.percentile(positive, [2.0, 99.0])
        tv_norm = np.clip((tv - lo) / max(float(hi - lo), 1e-9), 0, 1)
    else:
        tv_norm = np.zeros_like(tv, dtype=float)
    erode_iter = max(1, int(round(min(tv.shape) * 0.012)))
    inner = ndi.binary_erosion(atlas_mask, iterations=erode_iter)
    grad = np.hypot(ndi.sobel(ndi.gaussian_filter(tv_norm, sigma=1.0), axis=0),
                    ndi.sobel(ndi.gaussian_filter(tv_norm, sigma=1.0), axis=1))
    values = grad[inner & np.isfinite(grad)]
    tv_edges = np.zeros_like(atlas_mask, dtype=bool)
    if values.size >= 32 and float(values.max()) > 0:
        tv_edges = (grad >= float(np.percentile(values, 88.0))) & inner
    try:
        av_edges = _av_boundaries(av_slice) & inner
        av_edges = ndi.binary_dilation(av_edges, iterations=1)
    except Exception:
        av_edges = np.zeros_like(atlas_mask, dtype=bool)
    return np.asarray((tv_edges | av_edges) & inner, bool)


def _atlas_void_mask(tv_slice: np.ndarray, atlas_mask: np.ndarray) -> np.ndarray:
    atlas_mask = np.asarray(atlas_mask, bool)
    if not atlas_mask.any():
        return np.zeros(atlas_mask.shape, dtype=bool)
    tv = np.nan_to_num(np.asarray(tv_slice, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    positive = tv[atlas_mask & (tv > 0)]
    if positive.size < 32:
        return np.zeros(atlas_mask.shape, dtype=bool)
    low = float(np.percentile(positive, 18.0))
    inner = ndi.binary_erosion(atlas_mask, iterations=max(1, int(min(atlas_mask.shape) * 0.018)))
    void = inner & (tv <= low)
    return _major_components(void, min_fraction=0.015, max_components=8)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    rows = np.flatnonzero(np.any(mask, axis=1))
    cols = np.flatnonzero(np.any(mask, axis=0))
    if rows.size == 0 or cols.size == 0:
        return None
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def _canonical_mask(mask: np.ndarray, size: int = 128, pad_frac: float = 0.08) -> np.ndarray:
    mask = np.asarray(mask, bool)
    box = _bbox(mask)
    canvas = np.zeros((size, size), dtype=bool)
    if box is None:
        return canvas
    r0, r1, c0, c1 = box
    crop = mask[r0:r1, c0:c1]
    h, w = crop.shape
    inner = max(8, int(round(size * (1.0 - 2.0 * pad_frac))))
    scale = min(inner / max(h, 1), inner / max(w, 1))
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    zoomed = ndi.zoom(crop.astype(float), (new_h / h, new_w / w), order=0) > 0.5
    y0 = (size - zoomed.shape[0]) // 2
    x0 = (size - zoomed.shape[1]) // 2
    canvas[y0:y0 + zoomed.shape[0], x0:x0 + zoomed.shape[1]] = zoomed
    return canvas


def _canonical_image(image: np.ndarray, mask: np.ndarray, size: int = 128, pad_frac: float = 0.08) -> np.ndarray:
    image = np.nan_to_num(np.asarray(image, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    mask = np.asarray(mask, bool)
    box = _bbox(mask)
    canvas = np.zeros((size, size), dtype=float)
    if box is None or image.ndim != 2:
        return canvas
    r0, r1, c0, c1 = box
    crop = image[r0:r1, c0:c1]
    h, w = crop.shape
    inner = max(8, int(round(size * (1.0 - 2.0 * pad_frac))))
    scale = min(inner / max(h, 1), inner / max(w, 1))
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    zoomed = ndi.zoom(crop, (new_h / h, new_w / w), order=1)
    y0 = (size - zoomed.shape[0]) // 2
    x0 = (size - zoomed.shape[1]) // 2
    canvas[y0:y0 + zoomed.shape[0], x0:x0 + zoomed.shape[1]] = zoomed
    return canvas


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, bool)
    b = np.asarray(b, bool)
    denom = int(a.sum() + b.sum())
    if denom <= 0:
        return 0.0
    return float(2.0 * np.logical_and(a, b).sum() / denom)


def _shift_mask(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Translate a mask without wraparound."""
    mask = np.asarray(mask, bool)
    out = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    src_y0 = max(0, -dy)
    src_y1 = min(h, h - dy)
    dst_y0 = max(0, dy)
    dst_y1 = min(h, h + dy)
    src_x0 = max(0, -dx)
    src_x1 = min(w, w - dx)
    dst_x0 = max(0, dx)
    dst_x1 = min(w, w + dx)
    if src_y1 > src_y0 and src_x1 > src_x0:
        out[dst_y0:dst_y1, dst_x0:dst_x1] = mask[src_y0:src_y1, src_x0:src_x1]
    return out


def _edge_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, bool)
    if not mask.any():
        return mask
    return mask ^ ndi.binary_erosion(mask, iterations=1)


def _one_sided_chamfer(source_edge: np.ndarray, target_edge: np.ndarray) -> float:
    """Return a 0..1 edge proximity score from source edge to target edge."""
    source_edge = np.asarray(source_edge, bool)
    target_edge = np.asarray(target_edge, bool)
    if not source_edge.any() or not target_edge.any():
        return 0.0
    dist = ndi.distance_transform_edt(~target_edge)
    # 4 px is a good-enough contour tolerance on the 128 px canonical canvas.
    return float(np.exp(-np.mean(np.clip(dist[source_edge], 0, 12)) / 4.0))


def _partial_shape_score(hist_shape: np.ndarray, atlas_shape: np.ndarray, max_shift: int = 8) -> float:
    """Score a partial histology silhouette against a complete atlas silhouette.

    Symmetric Dice is brittle when a mounted section has missing cortex or a split
    midline. This score mostly asks whether observed tissue is explained by the
    atlas outline, and only lightly penalizes atlas tissue absent from histology.
    A small translation search absorbs mounting/cropping offsets after scale
    normalization.
    """
    return acceleration.partial_shape_score(hist_shape, atlas_shape, max_shift=max_shift)


def _partial_feature_score(hist_feature: np.ndarray, atlas_feature: np.ndarray, max_shift: int = 8) -> float:
    return acceleration.partial_feature_score(hist_feature, atlas_feature, max_shift=max_shift)


def _corr_score(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    mask = np.asarray(mask, bool)
    if mask.sum() < 16:
        return 0.0
    x = np.asarray(a, dtype=float)[mask]
    y = np.asarray(b, dtype=float)[mask]
    x = x - float(x.mean())
    y = y - float(y.mean())
    denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    if denom <= 1e-12:
        return 0.0
    return float(np.clip((np.sum(x * y) / denom + 1.0) * 0.5, 0.0, 1.0))


def _shape_features(mask: np.ndarray) -> tuple[float, float]:
    box = _bbox(mask)
    if box is None:
        return 0.0, 0.0
    r0, r1, c0, c1 = box
    h = max(1, r1 - r0)
    w = max(1, c1 - c0)
    area = float(np.asarray(mask, bool).sum())
    return float(w / h), float(area / (w * h))


def _grab_coronal_match_slice(atlas: AllenCCFAtlas, ap: int, spacing: int) -> dict:
    """Fast no-tilt coronal slice sampler used by auto-match scoring.

    ``AllenCCFAtlas.grab_atlas_slice`` supports arbitrary tilted planes and builds
    coordinate grids every time. Auto-match only scores untilted coronal planes,
    so direct AP indexing avoids that overhead. Fake atlas objects in tests still
    fall back to the generic method.
    """
    tv = getattr(atlas, "tv", None)
    av = getattr(atlas, "av", None)
    if tv is None or av is None:
        return atlas.grab_atlas_slice(
            coronal_slice_point(ap, atlas),
            coronal_camera_vector(0, 0),
            spacing=spacing,
        )
    ap_n, _dv_n, _ml_n = atlas.shape
    ap_idx = int(np.clip(int(round(ap)) - 1, 0, ap_n - 1))
    step = max(1, int(spacing))
    tv_plane = np.asarray(tv[ap_idx, ::step, ::step], dtype=np.float64)
    av_plane = np.asarray(av[ap_idx, ::step, ::step], dtype=np.float64)
    brain = av_plane > 0
    tv_slice = np.full(tv_plane.shape, np.nan, dtype=np.float64)
    av_slice = np.full(av_plane.shape, np.nan, dtype=np.float64)
    tv_slice[brain] = tv_plane[brain]
    av_slice[brain] = av_plane[brain]
    return {"tv_slices": tv_slice, "av_slices": av_slice}


def automatch_coronal_ap(
    image: np.ndarray,
    atlas: AllenCCFAtlas,
    *,
    center_ap: int | None = None,
    search_radius: int | None = None,
    prior_weight: float = 0.08,
    atlas_to_histology_scale: float | None = None,
    size_weight: float = 0.50,
    ap_step: int = 10,
    refine_radius: int = 20,
    refine_step: int = 2,
    spacing: int = 8,
    canvas_size: int = 128,
    workers: int | None = None,
) -> dict:
    """Predict the best coronal AP position from slice/atlas silhouettes.

    Returns ``{"ap": int, "score": float, "confidence": float, "top": [...]}``.
    It is intentionally light and deterministic: no model files or training data.
    """
    hist_mask = histology_shape_mask(image)
    if hist_mask.sum() < 64:
        raise RuntimeError("Could not extract a usable tissue silhouette from the current slice.")
    hist_shape = _canonical_mask(hist_mask, canvas_size)
    hist_aspect, hist_fill = _shape_features(hist_mask)
    hist_box = _bbox(hist_mask)
    if hist_box is None:
        raise RuntimeError("Could not extract a usable tissue silhouette from the current slice.")
    hr0, hr1, hc0, hc1 = hist_box
    hist_width = max(1.0, float(hc1 - hc0))
    hist_height = max(1.0, float(hr1 - hr0))
    hist_signal = _histology_signal(image)
    hist_edges = _canonical_mask(_internal_edge_mask(hist_signal, hist_mask), canvas_size)
    hist_void = _canonical_mask(_histology_void_mask(hist_signal, hist_mask), canvas_size)
    hist_signal_c = _canonical_image(hist_signal, hist_mask, canvas_size)
    hist_grad = np.hypot(
        ndi.sobel(ndi.gaussian_filter(hist_signal, sigma=1.0), axis=0),
        ndi.sobel(ndi.gaussian_filter(hist_signal, sigma=1.0), axis=1),
    )
    hist_grad_c = _canonical_image(hist_grad, hist_mask, canvas_size)
    ap_n, _dv_n, _ml_n = atlas.shape
    cv = coronal_camera_vector(0, 0)
    if center_ap is not None:
        center_ap = int(np.clip(int(center_ap), 1, ap_n - 1))
    if center_ap is not None and search_radius is None:
        search_radius = 280
    if search_radius is not None and search_radius <= 0:
        search_radius = None

    cache: dict[int, tuple[float, np.ndarray]] = {}
    used_workers = 1

    def score_ap_uncached(ap: int) -> tuple[float, np.ndarray]:
        ap = int(np.clip(ap, 1, ap_n - 1))
        sl = _grab_coronal_match_slice(atlas, ap, spacing=spacing)
        atlas_mask = atlas_shape_mask(sl["tv_slices"])
        area_frac = float(atlas_mask.sum() / max(1, atlas_mask.size))
        box = _bbox(atlas_mask)
        if atlas_mask.sum() < 64 or area_frac < 0.01 or box is None:
            score = -1.0
        else:
            atlas_shape = _canonical_mask(atlas_mask, canvas_size)
            atlas_aspect, atlas_fill = _shape_features(atlas_mask)
            r0, r1, c0, c1 = box
            atlas_width = max(1.0, float(c1 - c0))
            atlas_height = max(1.0, float(r1 - r0))
            height_frac = float((r1 - r0) / max(1, atlas_mask.shape[0]))
            width_frac = float((c1 - c0) / max(1, atlas_mask.shape[1]))
            score = _partial_shape_score(hist_shape, atlas_shape)
            atlas_edges = _canonical_mask(_atlas_internal_edges(sl["tv_slices"], sl.get("av_slices"), atlas_mask), canvas_size)
            atlas_void = _canonical_mask(_atlas_void_mask(sl["tv_slices"], atlas_mask), canvas_size)
            edge_score = _partial_feature_score(hist_edges, atlas_edges)
            void_score = _partial_shape_score(hist_void, atlas_void, max_shift=10) if hist_void.any() and atlas_void.any() else 0.0
            tv = np.nan_to_num(np.asarray(sl["tv_slices"], dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
            positive = tv[atlas_mask & (tv > 0)]
            if positive.size:
                lo_tv, hi_tv = np.percentile(positive, [2.0, 99.0])
                tv_norm = np.clip((tv - lo_tv) / max(float(hi_tv - lo_tv), 1e-9), 0, 1)
            else:
                tv_norm = np.zeros_like(tv, dtype=float)
            atlas_signal_c = _canonical_image(tv_norm, atlas_mask, canvas_size)
            atlas_grad = np.hypot(
                ndi.sobel(ndi.gaussian_filter(tv_norm, sigma=1.0), axis=0),
                ndi.sobel(ndi.gaussian_filter(tv_norm, sigma=1.0), axis=1),
            )
            atlas_grad_c = _canonical_image(atlas_grad, atlas_mask, canvas_size)
            overlap = hist_shape & atlas_shape
            signal_score = max(
                _corr_score(hist_signal_c, atlas_signal_c, overlap),
                _corr_score(1.0 - hist_signal_c, atlas_signal_c, overlap),
            )
            grad_score = _corr_score(hist_grad_c, atlas_grad_c, overlap)
            # Aspect/fill still help reject absurd planes, but the penalties
            # are deliberately mild because the mounted slice may be partial.
            score -= 0.06 * min(abs(hist_aspect - atlas_aspect), 2.0)
            score -= 0.03 * min(abs(hist_fill - atlas_fill), 1.0)
            score += 0.10 * edge_score
            score += 0.08 * void_score
            score += 0.34 * signal_score
            score += 0.08 * grad_score
            if atlas_edges.any() and edge_score < 0.25:
                edge_density = float(atlas_edges.sum() / max(1, atlas_edges.size))
                score -= 0.10 * min(edge_density / 0.08, 1.0)
            # Very anterior/posterior fragments can have a similar normalized
            # outline after canonical scaling; keep their low absolute coverage
            # from beating a real full coronal section.
            score -= 0.18 * max(0.0, 0.08 - area_frac) / 0.08
            score -= 0.10 * max(0.0, 0.35 - width_frac) / 0.35
            score -= 0.06 * max(0.0, 0.35 - height_frac) / 0.35
            if center_ap is not None and search_radius is not None:
                score -= float(prior_weight) * min(((ap - center_ap) / max(1.0, float(search_radius))) ** 2, 4.0)
            if atlas_to_histology_scale is not None and atlas_to_histology_scale > 0:
                expected_w = atlas_width * float(spacing) * float(atlas_to_histology_scale)
                expected_h = atlas_height * float(spacing) * float(atlas_to_histology_scale)
                width_err = abs(np.log(hist_width / max(expected_w, 1e-9)))
                height_err = abs(np.log(hist_height / max(expected_h, 1e-9)))
                score -= float(size_weight) * min(0.65 * width_err + 0.35 * height_err, 1.5)
        return float(score), atlas_mask

    def score_ap(ap: int) -> float:
        ap = int(np.clip(ap, 1, ap_n - 1))
        if ap not in cache:
            cache[ap] = score_ap_uncached(ap)
        return cache[ap][0]

    def score_aps(aps: Sequence[int]) -> list[tuple[int, float]]:
        nonlocal used_workers
        requested: list[int] = [int(np.clip(int(ap), 1, ap_n - 1)) for ap in aps]
        missing = list(dict.fromkeys(ap for ap in requested if ap not in cache))
        if missing:
            n_workers = int(workers) if workers is not None else acceleration.auto_worker_count(len(missing))
            n_workers = max(1, min(n_workers, len(missing)))
            if n_workers > 1 and len(missing) >= max(4, n_workers):
                used_workers = max(used_workers, n_workers)
                with ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="histmatch") as ex:
                    for ap, item in zip(missing, ex.map(score_ap_uncached, missing)):
                        cache[ap] = item
            else:
                for ap in missing:
                    cache[ap] = score_ap_uncached(ap)
        return [(ap, cache[ap][0]) for ap in requested]

    ap_lo = 1
    ap_hi = ap_n - 1
    if center_ap is not None and search_radius is not None:
        ap_lo = max(1, center_ap - int(search_radius))
        ap_hi = min(ap_n - 1, center_ap + int(search_radius))
    coarse_aps = list(range(ap_lo, ap_hi + 1, max(1, int(ap_step))))
    if coarse_aps[-1] != ap_hi:
        coarse_aps.append(ap_hi)
    coarse_scores = score_aps(coarse_aps)
    coarse_best = max(coarse_scores, key=lambda item: item[1])[0]
    lo = max(1, coarse_best - int(refine_radius))
    hi = min(ap_n - 1, coarse_best + int(refine_radius))
    refine_aps = list(range(lo, hi + 1, max(1, int(refine_step))))
    scored = {ap: score for ap, score in coarse_scores}
    for ap, score in score_aps(refine_aps):
        scored[ap] = score
    ranked = sorted(scored.items(), key=lambda item: item[1], reverse=True)
    best_ap, best_score = ranked[0]
    if best_score < 0:
        raise RuntimeError("Auto match could not find a valid atlas tissue plane.")
    far = [score for ap, score in ranked if abs(ap - best_ap) >= max(20, refine_radius)]
    confidence = float(best_score - max(far)) if far else float(best_score)
    return {
        "ap": int(best_ap),
        "score": float(best_score),
        "confidence": confidence,
        "hist_mask_area": int(hist_mask.sum()),
        "top": [{"ap": int(ap), "score": float(score)} for ap, score in ranked[:5]],
        "engine": {
            "workers": int(used_workers),
            "numba": acceleration.numba_available(),
            "cuda": acceleration.cuda_available(),
        },
    }
