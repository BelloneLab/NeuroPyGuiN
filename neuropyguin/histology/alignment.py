"""Atlas <-> histology 2D alignment.

Native port of ``align_manual_histology_atlas.m`` (control-point affine) and
``align_auto_histology_atlas.m`` (intensity registration). The stored transform
``atlas2histology_tform`` is a per-slice 3x3 matrix ``T`` in the MATLAB
row-vector convention: ``[x' y' 1] = [x y 1] @ T`` mapping **atlas** pixel coords
to **histology** pixel coords (consumed by :mod:`tracing` and ``AP_histology2ccf.m``).

MATLAB ``fitgeotrans(..., 'affine')`` -> least squares here.
MATLAB ``imregtform(..., 'affine','multimodal')`` -> OpenCV ECC here.
MATLAB ``imwarp`` -> ``cv2.warpAffine`` (nearest for labels).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy import ndimage as ndi

from . import acceleration

try:  # OpenCV is optional; manual alignment works without it.
    import cv2  # type: ignore
    _HAS_CV2 = True
except Exception:  # pragma: no cover
    _HAS_CV2 = False


AUTO_ALIGN_MIN_SHAPE_SCORE = 0.65


def fit_affine_from_points(atlas_pts: np.ndarray, histology_pts: np.ndarray) -> np.ndarray:
    """Least-squares affine ``T`` (3x3) mapping atlas -> histology points.

    Equivalent to ``fitgeotrans(atlas_pts, histology_pts, 'affine')``. Needs at
    least 3 non-collinear point pairs.
    """
    atlas_pts = np.atleast_2d(np.asarray(atlas_pts, dtype=np.float64))
    histology_pts = np.atleast_2d(np.asarray(histology_pts, dtype=np.float64))
    if len(atlas_pts) < 3 or len(atlas_pts) != len(histology_pts):
        raise ValueError("Need >= 3 matched control-point pairs.")
    M = np.column_stack([atlas_pts, np.ones(len(atlas_pts))])  # (N,3)
    H = np.column_stack([histology_pts, np.ones(len(histology_pts))])  # (N,3)
    T, *_ = np.linalg.lstsq(M, H, rcond=None)  # (3,3), [x y 1]@T = [x' y' 1]
    T[:, 2] = [0.0, 0.0, 1.0]  # enforce affine (no projective component)
    return T


def matlab_T_to_cv2(T: np.ndarray) -> np.ndarray:
    """Convert a 3x3 MATLAB row-vector ``T`` to an OpenCV 2x3 forward matrix."""
    T = np.asarray(T, dtype=np.float64)
    return np.array([
        [T[0, 0], T[1, 0], T[2, 0]],
        [T[0, 1], T[1, 1], T[2, 1]],
    ], dtype=np.float64)


def cv2_to_matlab_T(M: np.ndarray) -> np.ndarray:
    """Convert an OpenCV 2x3 forward matrix to a 3x3 MATLAB row-vector ``T``."""
    M = np.asarray(M, dtype=np.float64)
    return np.array([
        [M[0, 0], M[1, 0], 0.0],
        [M[0, 1], M[1, 1], 0.0],
        [M[0, 2], M[1, 2], 1.0],
    ], dtype=np.float64)


def warp_atlas(image: np.ndarray, T: np.ndarray, out_shape: Tuple[int, int],
               nearest: bool = True, use_cv2: bool = True) -> np.ndarray:
    """Warp an atlas image into histology space using ``T`` (atlas -> histology).

    ``use_cv2=False`` forces the pure-scipy path. OpenCV's ``warpAffine`` can, on
    pathological transforms, fault at the C++ level (uncatchable); callers that run
    on the GUI thread (e.g. the live overlay) pass ``False`` so a bad transform can
    only raise a normal, catchable Python exception instead of crashing the app.
    """
    out_h, out_w = out_shape
    src = np.nan_to_num(np.asarray(image, dtype=np.float64))
    M = matlab_T_to_cv2(T)
    if _HAS_CV2 and use_cv2:
        flags = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
        return cv2.warpAffine(src, M, (out_w, out_h), flags=flags)
    # scipy fallback: affine_transform needs the inverse (output -> input).
    A = np.array([[M[0, 0], M[0, 1]], [M[1, 0], M[1, 1]]])
    t = np.array([M[0, 2], M[1, 2]])
    Ainv = np.linalg.inv(A)
    order = 0 if nearest else 1
    # output coord (row=y, col=x); map to input via inverse.
    mat = np.array([[Ainv[1, 1], Ainv[1, 0]], [Ainv[0, 1], Ainv[0, 0]]])
    offset_xy = -Ainv @ t
    offset = np.array([offset_xy[1], offset_xy[0]])
    return ndi.affine_transform(src, mat, offset=offset, output_shape=(out_h, out_w), order=order)


def atlas_boundaries(av_warped: np.ndarray) -> np.ndarray:
    """Boundary mask of a (warped) annotation slice (port of the conv2 trick)."""
    av = np.nan_to_num(np.asarray(av_warped, dtype=np.float64))
    smoothed = np.round(ndi.uniform_filter(av, size=3))
    return smoothed != av


def _pad_to(img: np.ndarray, h: int, w: int) -> np.ndarray:
    """Zero-pad ``img`` (top-left aligned) to at least ``(h, w)``."""
    out = np.zeros((h, w), dtype=img.dtype)
    out[: img.shape[0], : img.shape[1]] = img[:h, :w]
    return out


def _as_gray(image: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(np.asarray(image, dtype=np.float32))
    if arr.ndim == 3:
        arr = arr[..., :3].mean(axis=2)
    return arr


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    rows = np.flatnonzero(np.any(mask, axis=1))
    cols = np.flatnonzero(np.any(mask, axis=0))
    if rows.size == 0 or cols.size == 0:
        return None
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def _edge_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, bool)
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    return mask ^ ndi.binary_erosion(mask, iterations=1)


def _safe_scale(mask: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    mask = np.asarray(mask, bool)
    side = max(mask.shape) if mask.ndim == 2 and mask.size else 0
    if side <= max_side or side <= 0:
        return mask, 1.0
    scale = float(max_side) / float(side)
    small = ndi.zoom(mask.astype(float), (scale, scale), order=0) > 0.5
    return small, scale


def _histology_tissue_mask(image: np.ndarray) -> np.ndarray:
    """Extract a filled histology tissue mask while keeping split hemispheres."""
    try:
        from . import matching

        mask = matching.histology_shape_mask(image)
    except Exception:
        gray = _as_gray(image)
        if gray.ndim != 2 or gray.size == 0:
            return np.zeros(gray.shape[:2], dtype=bool)
        positive = gray[gray > 0]
        thresh = float(np.percentile(positive, 55.0)) if positive.size else float(gray.mean())
        mask = gray >= thresh
        mask = ndi.binary_opening(mask, iterations=1)
        mask = ndi.binary_closing(mask, iterations=max(2, int(min(mask.shape) * 0.012)))
        mask = ndi.binary_fill_holes(mask)
    return np.asarray(mask, bool)


def _atlas_tissue_mask(atlas_tv: np.ndarray) -> np.ndarray:
    try:
        from . import matching

        return matching.atlas_shape_mask(atlas_tv)
    except Exception:
        tv = np.nan_to_num(np.asarray(atlas_tv, dtype=float), nan=0.0)
        return tv > 0


def _midline_x(mask: np.ndarray) -> float | None:
    """Estimate the coronal midline from a split tissue gap or bbox center."""
    mask = np.asarray(mask, bool)
    box = _bbox(mask)
    if box is None:
        return None
    _r0, _r1, c0, c1 = box
    cols = mask.sum(axis=0).astype(float)
    lo = int(round(c0 + 0.32 * (c1 - c0)))
    hi = int(round(c0 + 0.68 * (c1 - c0)))
    if hi > lo + 2:
        central = cols[lo:hi]
        if central.size:
            return float(lo + int(np.argmin(central)))
    return float((c0 + c1 - 1) / 2.0)


def _centroid(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def _transform_from_centers(
    atlas_center: tuple[float, float],
    hist_center: tuple[float, float],
    sx: float,
    sy: float,
    angle_deg: float = 0.0,
) -> np.ndarray:
    ax, ay = atlas_center
    hx, hy = hist_center
    theta = np.deg2rad(float(angle_deg))
    c, s = float(np.cos(theta)), float(np.sin(theta))
    # Row-vector affine: [x y 1] @ T = [x' y' 1].
    a = sx * c
    b = sx * s
    d = -sy * s
    e = sy * c
    tx = hx - (ax * a + ay * d)
    ty = hy - (ax * b + ay * e)
    return np.array([[a, b, 0.0], [d, e, 0.0], [tx, ty, 1.0]], dtype=np.float64)


def _warp_mask(mask: np.ndarray, T: np.ndarray, out_shape: tuple[int, int]) -> np.ndarray:
    if not np.isfinite(T).all():
        return np.zeros(out_shape, dtype=bool)
    try:
        warped = warp_atlas(mask.astype(np.uint8), T, out_shape, nearest=True, use_cv2=_HAS_CV2)
    except Exception:
        warped = warp_atlas(mask.astype(np.uint8), T, out_shape, nearest=True, use_cv2=False)
    return np.asarray(warped) > 0.5


def _shape_score(
    hist_mask: np.ndarray,
    atlas_warped: np.ndarray,
    *,
    hist_edge: np.ndarray | None = None,
    dist_to_hist_edge: np.ndarray | None = None,
) -> float:
    hist_mask = np.asarray(hist_mask, bool)
    atlas_warped = np.asarray(atlas_warped, bool)
    if not hist_mask.any() or not atlas_warped.any():
        return 0.0
    inter = np.logical_and(hist_mask, atlas_warped).sum()
    hist_sum = max(1, int(hist_mask.sum()))
    atlas_sum = max(1, int(atlas_warped.sum()))
    observed_fit = float(inter / hist_sum)
    atlas_precision = float(inter / atlas_sum)
    dice = float(2.0 * inter / max(1, hist_sum + atlas_sum))

    hist_edge = _edge_mask(hist_mask) if hist_edge is None else np.asarray(hist_edge, bool)
    atlas_edge = _edge_mask(atlas_warped)
    if hist_edge.any() and atlas_edge.any():
        dist_to_hist = (
            ndi.distance_transform_edt(~hist_edge)
            if dist_to_hist_edge is None
            else np.asarray(dist_to_hist_edge, dtype=np.float32)
        )
        dist_to_atlas = ndi.distance_transform_edt(~atlas_edge)
        atlas_edge_fit = float(np.exp(-np.mean(np.clip(dist_to_hist[atlas_edge], 0, 20)) / 5.0))
        hist_edge_fit = float(np.exp(-np.mean(np.clip(dist_to_atlas[hist_edge], 0, 20)) / 5.0))
    else:
        atlas_edge_fit = hist_edge_fit = 0.0

    area_ratio = atlas_sum / hist_sum
    area_penalty = min(abs(np.log(max(area_ratio, 1e-6))), 2.0)
    return float(
        0.34 * observed_fit
        + 0.26 * atlas_edge_fit
        + 0.18 * hist_edge_fit
        + 0.14 * dice
        + 0.08 * atlas_precision
        - 0.05 * area_penalty
    )


def _shape_align_impl(histology_image: np.ndarray, atlas_tv: np.ndarray) -> tuple[np.ndarray, float]:
    hist_mask_full = _histology_tissue_mask(histology_image)
    atlas_mask_full = _atlas_tissue_mask(atlas_tv)
    if hist_mask_full.ndim != 2 or atlas_mask_full.ndim != 2:
        return resize_only_transform(hist_mask_full.shape, atlas_mask_full.shape), 0.0
    if hist_mask_full.sum() < 64 or atlas_mask_full.sum() < 64:
        return resize_only_transform(hist_mask_full.shape, atlas_mask_full.shape), 0.0

    max_side = 320
    hist_mask, hist_scale = _safe_scale(hist_mask_full, max_side)
    atlas_mask, atlas_scale = _safe_scale(atlas_mask_full, max_side)
    hbox = _bbox(hist_mask)
    abox = _bbox(atlas_mask)
    if hbox is None or abox is None:
        return resize_only_transform(hist_mask_full.shape, atlas_mask_full.shape), 0.0

    hr0, hr1, hc0, hc1 = hbox
    ar0, ar1, ac0, ac1 = abox
    hw, hh = max(1, hc1 - hc0), max(1, hr1 - hr0)
    aw, ah = max(1, ac1 - ac0), max(1, ar1 - ar0)
    hist_mid = _midline_x(hist_mask)
    atlas_mid = _midline_x(atlas_mask)
    hist_centroid = _centroid(hist_mask)
    atlas_centroid = _centroid(atlas_mask)
    hist_bbox_center = ((hc0 + hc1 - 1) / 2.0, (hr0 + hr1 - 1) / 2.0)
    atlas_bbox_center = ((ac0 + ac1 - 1) / 2.0, (ar0 + ar1 - 1) / 2.0)
    if hist_centroid is None:
        hist_centroid = hist_bbox_center
    if atlas_centroid is None:
        atlas_centroid = atlas_bbox_center

    hist_centers = [hist_bbox_center, hist_centroid]
    if hist_mid is not None:
        hist_centers.append((hist_mid, hist_bbox_center[1]))
        hist_centers.append((hist_mid, hist_centroid[1]))
    atlas_centers = [atlas_bbox_center, atlas_centroid]
    if atlas_mid is not None:
        atlas_centers.append((atlas_mid, atlas_bbox_center[1]))

    width_scale = hw / aw
    height_scale = hh / ah
    if hist_mid is not None:
        sym_half = max(abs(hist_mid - hc0), abs((hc1 - 1) - hist_mid))
        sym_width_scale = max(1.0, 2.0 * sym_half) / aw
    else:
        sym_width_scale = width_scale
    base_scales = [
        (width_scale, height_scale),
        (height_scale, height_scale),
        (sym_width_scale, height_scale),
        (np.sqrt(max(width_scale * height_scale, 1e-9)), np.sqrt(max(width_scale * height_scale, 1e-9))),
    ]
    scale_pairs: list[tuple[float, float]] = []
    for sx, sy in base_scales:
        for mult in (0.90, 1.0, 1.10):
            sxm, sym = float(sx * mult), float(sy * mult)
            if 0.05 <= sxm <= 20 and 0.05 <= sym <= 20:
                scale_pairs.append((sxm, sym))

    dy_step = max(4.0, hh * 0.06)
    dx_step = max(4.0, hw * 0.06)
    offsets = [
        (0.0, 0.0),
        (-dx_step, 0.0), (dx_step, 0.0), (0.0, -dy_step), (0.0, dy_step),
        (-0.5 * dx_step, -0.5 * dy_step), (0.5 * dx_step, -0.5 * dy_step),
        (-0.5 * dx_step, 0.5 * dy_step), (0.5 * dx_step, 0.5 * dy_step),
    ]
    angles = (-8.0, -4.0, 0.0, 4.0, 8.0)

    candidates: list[np.ndarray] = []
    seen: set[tuple[float, ...]] = set()
    for ac in atlas_centers:
        for hc in hist_centers:
            for sx, sy in scale_pairs:
                for angle in angles:
                    for dx, dy in offsets:
                        hcx = (hc[0] + dx, hc[1] + dy)
                        T = _transform_from_centers(ac, hcx, sx, sy, angle)
                        key = tuple(np.round(T.ravel(), 3))
                        if key in seen:
                            continue
                        seen.add(key)
                        candidates.append(T)

    if not candidates:
        return resize_only_transform(hist_mask_full.shape, atlas_mask_full.shape), 0.0

    keep = min(len(candidates), max(48, len(candidates) // 12))
    ranked = acceleration.rank_affine_candidates(hist_mask, atlas_mask, candidates, keep=keep)
    candidate_order = ranked if ranked is not None and len(candidates) > keep else list(range(len(candidates)))

    best_T_small: np.ndarray | None = None
    best_score = -np.inf
    hist_edge = _edge_mask(hist_mask)
    dist_to_hist_edge = ndi.distance_transform_edt(~hist_edge).astype(np.float32) if hist_edge.any() else None
    for i in candidate_order:
        T = candidates[i]
        warped = _warp_mask(atlas_mask, T, hist_mask.shape)
        score = _shape_score(hist_mask, warped, hist_edge=hist_edge, dist_to_hist_edge=dist_to_hist_edge)
        if score > best_score:
            best_score = score
            best_T_small = T

    if best_T_small is None or not np.isfinite(best_score):
        return resize_only_transform(hist_mask_full.shape, atlas_mask_full.shape), 0.0

    # Convert small-mask coordinates back to full-resolution atlas -> histology.
    scale_atlas_to_small = np.diag([atlas_scale, atlas_scale, 1.0])
    scale_hist_to_full = np.diag([1.0 / hist_scale, 1.0 / hist_scale, 1.0])
    T_full = scale_atlas_to_small @ best_T_small @ scale_hist_to_full
    if not np.isfinite(T_full).all():
        return resize_only_transform(hist_mask_full.shape, atlas_mask_full.shape), 0.0
    return T_full, float(best_score)


def auto_align_shape(histology_image: np.ndarray, atlas_tv: np.ndarray) -> np.ndarray:
    """Shape-constrained atlas->histology alignment.

    This uses the filled tissue silhouette and atlas brain mask, so it remains
    useful when sections are split, cropped, or have missing tissue. It is the
    preferred initializer for automatic alignment; manual control points remain
    the exact override.
    """
    T, _score = _shape_align_impl(histology_image, atlas_tv)
    return T


def auto_align(
    histology_gray: np.ndarray,
    atlas_tv: np.ndarray,
    downsample: int = 5,
    max_iter: int = 200,
) -> np.ndarray:
    """Intensity-based affine alignment of atlas template -> histology.

    The primary path is shape-constrained: estimate the histology tissue outline,
    align the atlas brain mask to it, and return the full-resolution 3x3 ``T``.
    The older ECC intensity registration is kept only as a fallback for slices
    where a tissue outline cannot be extracted.
    """
    hist_image = np.nan_to_num(np.asarray(histology_gray, dtype=np.float32))
    hist = _as_gray(hist_image)
    atlas = np.nan_to_num(np.asarray(atlas_tv, dtype=np.float32))
    # Degenerate inputs (wrong rank, empty) can hard-crash the native ECC solver;
    # bail out to the identity rather than letting OpenCV abort the process.
    if hist.ndim != 2 or atlas.ndim != 2 or hist.size == 0 or atlas.size == 0:
        return np.eye(3)
    if hist.max() > 0:
        hist = hist / hist.max()
    if atlas.max() > 0:
        atlas = atlas / atlas.max()

    resize_factor = float(min(np.array(hist.shape) / np.array(atlas.shape)))
    if not np.isfinite(resize_factor) or resize_factor <= 0:
        return np.eye(3)
    scale_match = np.diag([resize_factor, resize_factor, 1.0])
    shape_T, shape_score = _shape_align_impl(hist_image, atlas)
    if shape_score > 0.25 and np.isfinite(shape_T).all():
        return shape_T

    if not _HAS_CV2:
        return scale_match  # only the resize component

    try:
        atlas_resized = cv2.resize(
            atlas,
            (max(1, int(atlas.shape[1] * resize_factor)),
             max(1, int(atlas.shape[0] * resize_factor))),
            interpolation=cv2.INTER_NEAREST,
        )
        # Shrink only as far as the smaller image allows: ECC is unstable (and can
        # abort natively) on tiny images, so keep the min side >= 32 px.
        min_side = min(hist.shape[0], hist.shape[1],
                       atlas_resized.shape[0], atlas_resized.shape[1])
        ds = int(np.clip(downsample, 1, max(1, min_side // 32)))
        fixed = cv2.resize(hist, (max(1, hist.shape[1] // ds), max(1, hist.shape[0] // ds)))
        moving = cv2.resize(
            atlas_resized,
            (max(1, atlas_resized.shape[1] // ds), max(1, atlas_resized.shape[0] // ds)),
        )
        # Pad to common size.
        H = max(fixed.shape[0], moving.shape[0])
        W = max(fixed.shape[1], moving.shape[1])
        fixed = _pad_to(fixed, H, W)
        moving = _pad_to(moving, H, W)
    except Exception:
        return scale_match

    # ECC needs textured (non-constant) images: a flat slice gives a singular
    # gradient system that the C++ solver can crash on. Fall back to the resize.
    if float(fixed.std()) < 1e-6 or float(moving.std()) < 1e-6:
        return scale_match

    warp = np.eye(2, 3, dtype=np.float32)
    try:
        gauss = 5 if min(H, W) >= 16 else 1
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iter, 1e-5)
        cv2.findTransformECC(fixed, moving, warp, cv2.MOTION_AFFINE, criteria, None, gauss)
    except Exception:
        return scale_match
    if not np.isfinite(warp).all():
        return scale_match

    # Compose: scale to histology, downscale, affine, upscale (as in MATLAB).
    down = np.diag([1.0 / ds, 1.0 / ds, 1.0])
    up = np.diag([float(ds), float(ds), 1.0])
    affine3 = np.vstack([warp, [0, 0, 1]])
    cv_full = scale_match @ down @ affine3 @ up  # 3x3 in cv2 (forward) convention
    T = cv2_to_matlab_T(cv_full[:2, :])
    if not np.isfinite(T).all():
        return scale_match
    return T


def resize_only_transform(hist_shape, atlas_shape) -> np.ndarray:
    """Pure atlas->histology resize ``T`` (the fallback when ECC is unavailable)."""
    hs = np.asarray(hist_shape, np.float64)[:2]
    ats = np.asarray(atlas_shape, np.float64)[:2]
    if hs.size < 2 or ats.size < 2 or np.any(ats == 0):
        return np.eye(3)
    rf = float(min(hs / ats))
    if not np.isfinite(rf) or rf <= 0:
        return np.eye(3)
    return np.diag([rf, rf, 1.0])


def auto_align_isolated(
    histology_gray: np.ndarray,
    atlas_tv: np.ndarray,
    python_exe: Optional[str] = None,
    timeout: float = 180.0,
) -> Tuple[np.ndarray, str]:
    """Run :func:`auto_align` in a **child process** and return ``(T, status)``.

    OpenCV's ``findTransformECC`` can, on pathological slices, abort at the C++
    level (an access violation / ``abort()``), which Python cannot catch and which
    would take the whole GUI down with it. Running it in a subprocess contains any
    such crash: we detect the abnormal exit and fall back to the pure-resize
    transform, so a bad auto-align only loses that one result instead of the app.
    """
    import shutil
    import subprocess
    import sys
    import tempfile
    from ..processes import tracked_run

    hist = np.nan_to_num(np.asarray(histology_gray, dtype=np.float32))
    atlas = np.nan_to_num(np.asarray(atlas_tv, dtype=np.float32))
    if hist.ndim not in (2, 3) or atlas.ndim != 2 or hist.size == 0 or atlas.size == 0:
        return np.eye(3), "auto-align skipped (unexpected slice shape); used identity."
    fallback = resize_only_transform(hist.shape[:2], atlas.shape)

    # The preferred shape-constrained path is pure Python/NumPy/Numba around
    # small binary masks and avoids OpenCV ECC, the native solver this isolation
    # wrapper was originally built to contain. Run it in-process so repeated GUI
    # auto-aligns reuse the Numba cache instead of launching a fresh interpreter.
    try:
        shape_T, shape_score = _shape_align_impl(hist, atlas)
        if shape_score >= AUTO_ALIGN_MIN_SHAPE_SCORE and np.isfinite(shape_T).all():
            return shape_T, f"auto-aligned (accelerated shape; score {shape_score:.2f})."
        if np.isfinite(shape_score):
            return fallback, (
                f"auto-align low confidence (shape score {shape_score:.2f}; "
                f"needs >= {AUTO_ALIGN_MIN_SHAPE_SCORE:.2f}); transform not changed."
            )
    except Exception:
        pass

    tmp = Path(tempfile.mkdtemp(prefix="npx_autoalign_"))
    try:
        hp, ap, op = tmp / "hist.npy", tmp / "atlas.npy", tmp / "T.npy"
        np.save(hp, hist)
        np.save(ap, atlas)
        repo_root = Path(__file__).resolve().parents[2]
        cmd = [python_exe or sys.executable, "-m", "neuropyguin.histology.alignment",
               str(hp), str(ap), str(op)]
        try:
            proc = tracked_run(
                cmd, capture_output=True, text=True, timeout=timeout, cwd=str(repo_root),
            )
        except subprocess.TimeoutExpired:
            return fallback, "auto-align timed out; used resize-only."
        if proc.returncode != 0 or not op.exists():
            detail = (proc.stderr or proc.stdout or "").strip().splitlines()
            tail = detail[-1] if detail else f"exit code {proc.returncode}"
            return fallback, f"auto-align crashed in isolation ({tail}); used resize-only."
        T = np.asarray(np.load(op), dtype=np.float64)
        if T.shape != (3, 3) or not np.isfinite(T).all():
            return fallback, "auto-align produced an invalid transform; used resize-only."
        return T, "auto-aligned (shape-constrained registration)."
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _main(argv=None) -> int:
    """CLI used by :func:`auto_align_isolated`: ``<hist.npy> <atlas.npy> <out.npy>``."""
    import argparse

    parser = argparse.ArgumentParser(description="Isolated atlas->histology auto-align")
    parser.add_argument("hist")
    parser.add_argument("atlas")
    parser.add_argument("out")
    args = parser.parse_args(argv)
    T = auto_align(np.load(args.hist), np.load(args.atlas))
    np.save(args.out, np.asarray(T, dtype=np.float64))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
