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

from typing import Dict, List, Sequence

import numpy as np
from scipy import ndimage as ndi

from .atlas import AllenCCFAtlas, coronal_camera_vector, coronal_slice_point


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
    hist_shape = np.asarray(hist_shape, bool)
    atlas_shape = np.asarray(atlas_shape, bool)
    hist_area = max(1, int(hist_shape.sum()))
    atlas_area = max(1, int(atlas_shape.sum()))
    hist_edge = _edge_mask(hist_shape)
    atlas_edge = _edge_mask(atlas_shape)
    best = 0.0
    for dy in range(-max_shift, max_shift + 1, 4):
        for dx in range(-max_shift, max_shift + 1, 4):
            shifted = _shift_mask(hist_shape, dy, dx)
            shifted_edge = _shift_mask(hist_edge, dy, dx)
            inter = int(np.logical_and(shifted, atlas_shape).sum())
            observed_fit = inter / hist_area
            atlas_coverage = inter / atlas_area
            dice = _dice(shifted, atlas_shape)
            edge_fit = _one_sided_chamfer(shifted_edge, atlas_edge)
            edge_back = _one_sided_chamfer(atlas_edge, shifted_edge)
            score = (
                0.46 * observed_fit
                + 0.22 * edge_fit
                + 0.16 * dice
                + 0.10 * atlas_coverage
                + 0.06 * edge_back
            )
            best = max(best, float(score))
    return best


def _shape_features(mask: np.ndarray) -> tuple[float, float]:
    box = _bbox(mask)
    if box is None:
        return 0.0, 0.0
    r0, r1, c0, c1 = box
    h = max(1, r1 - r0)
    w = max(1, c1 - c0)
    area = float(np.asarray(mask, bool).sum())
    return float(w / h), float(area / (w * h))


def automatch_coronal_ap(
    image: np.ndarray,
    atlas: AllenCCFAtlas,
    *,
    ap_step: int = 10,
    refine_radius: int = 20,
    refine_step: int = 2,
    spacing: int = 8,
    canvas_size: int = 128,
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
    ap_n, _dv_n, _ml_n = atlas.shape
    cv = coronal_camera_vector(0, 0)

    cache: dict[int, tuple[float, np.ndarray]] = {}

    def score_ap(ap: int) -> float:
        ap = int(np.clip(ap, 1, ap_n - 1))
        if ap in cache:
            return cache[ap][0]
        sl = atlas.grab_atlas_slice(coronal_slice_point(ap, atlas), cv, spacing=spacing)
        atlas_mask = np.isfinite(sl["tv_slices"])
        if atlas_mask.sum() < 64:
            score = 0.0
        else:
            atlas_shape = _canonical_mask(atlas_mask, canvas_size)
            atlas_aspect, atlas_fill = _shape_features(atlas_mask)
            score = _partial_shape_score(hist_shape, atlas_shape)
            # Aspect/fill still help reject absurd planes, but the penalties
            # are deliberately mild because the mounted slice may be partial.
            score -= 0.06 * min(abs(hist_aspect - atlas_aspect), 2.0)
            score -= 0.03 * min(abs(hist_fill - atlas_fill), 1.0)
        cache[ap] = (float(score), atlas_mask)
        return float(score)

    coarse_aps = list(range(1, ap_n, max(1, int(ap_step))))
    if coarse_aps[-1] != ap_n - 1:
        coarse_aps.append(ap_n - 1)
    coarse_scores = [(ap, score_ap(ap)) for ap in coarse_aps]
    coarse_best = max(coarse_scores, key=lambda item: item[1])[0]
    lo = max(1, coarse_best - int(refine_radius))
    hi = min(ap_n - 1, coarse_best + int(refine_radius))
    refine_aps = list(range(lo, hi + 1, max(1, int(refine_step))))
    scored = {ap: score for ap, score in coarse_scores}
    for ap in refine_aps:
        scored[ap] = score_ap(ap)
    ranked = sorted(scored.items(), key=lambda item: item[1], reverse=True)
    best_ap, best_score = ranked[0]
    far = [score for ap, score in ranked if abs(ap - best_ap) >= max(20, refine_radius)]
    confidence = float(best_score - max(far)) if far else float(best_score)
    return {
        "ap": int(best_ap),
        "score": float(best_score),
        "confidence": confidence,
        "hist_mask_area": int(hist_mask.sum()),
        "top": [{"ap": int(ap), "score": float(score)} for ap, score in ranked[:5]],
    }
