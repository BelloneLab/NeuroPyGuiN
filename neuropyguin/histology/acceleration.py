"""Optional acceleration helpers for histology auto-match and auto-align.

The app must remain usable on machines without Numba/CUDA, so this module keeps
accelerator imports isolated and every public helper has a pure-Python fallback.
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np
from scipy import ndimage as ndi

try:  # pragma: no cover - availability depends on the local environment.
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None

try:  # pragma: no cover - availability depends on the local environment.
    import numba as nb  # type: ignore

    _HAS_NUMBA = True
except Exception:  # pragma: no cover
    nb = None
    _HAS_NUMBA = False


def numba_available() -> bool:
    return bool(_HAS_NUMBA)


def cuda_available() -> bool:
    if not _HAS_NUMBA:
        return False
    try:
        from numba import cuda  # type: ignore

        return bool(cuda.is_available())
    except Exception:
        return False


def cuda_device_name() -> str | None:
    if not cuda_available():
        return None
    try:
        from numba import cuda  # type: ignore

        dev = cuda.get_current_device()
        name = getattr(dev, "name", None)
        if isinstance(name, bytes):
            return name.decode(errors="replace")
        return str(name) if name else None
    except Exception:
        return None


def auto_worker_count(task_count: int | None = None, cap: int = 8) -> int:
    """Choose a conservative worker count for interactive histology work."""
    env = os.environ.get("NEUROPYGUIN_HISTO_WORKERS", "").strip()
    if env:
        try:
            value = int(env)
            return max(1, value)
        except ValueError:
            pass
    try:
        if psutil is not None:
            cores = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True)
        else:
            cores = os.cpu_count()
    except Exception:
        cores = os.cpu_count()
    workers = max(1, min(int(cores or 1), int(cap)))
    if task_count is not None and task_count > 0:
        workers = min(workers, int(task_count))
    return workers


def hardware_summary() -> str:
    parts = [f"CPU workers {auto_worker_count()}"]
    parts.append("Numba yes" if numba_available() else "Numba no")
    if cuda_available():
        name = cuda_device_name()
        parts.append(f"CUDA yes{f' ({name})' if name else ''}")
    else:
        parts.append("CUDA no")
    return "; ".join(parts)


def _edge_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, bool)
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    return mask ^ ndi.binary_erosion(mask, iterations=1)


def _partial_shape_score_python(hist_shape: np.ndarray, atlas_shape: np.ndarray, max_shift: int) -> float:
    hist_shape = np.asarray(hist_shape, bool)
    atlas_shape = np.asarray(atlas_shape, bool)
    hist_area = max(1, int(hist_shape.sum()))
    atlas_area = max(1, int(atlas_shape.sum()))
    hist_edge = _edge_mask(hist_shape)
    atlas_edge = _edge_mask(atlas_shape)
    dist_to_atlas = ndi.distance_transform_edt(~atlas_edge).astype(np.float32)
    dist_to_hist = ndi.distance_transform_edt(~hist_edge).astype(np.float32)
    return _partial_shape_score_numpy(
        hist_shape.astype(np.uint8),
        atlas_shape.astype(np.uint8),
        hist_edge.astype(np.uint8),
        atlas_edge.astype(np.uint8),
        dist_to_atlas,
        dist_to_hist,
        int(max_shift),
        int(hist_area),
        int(atlas_area),
        True,
    )


def _partial_feature_score_python(hist_feature: np.ndarray, atlas_feature: np.ndarray, max_shift: int) -> float:
    hist_feature = np.asarray(hist_feature, bool)
    atlas_feature = np.asarray(atlas_feature, bool)
    if not hist_feature.any() or not atlas_feature.any():
        return 0.0
    dist_to_atlas = ndi.distance_transform_edt(~atlas_feature).astype(np.float32)
    dist_to_hist = ndi.distance_transform_edt(~hist_feature).astype(np.float32)
    return _partial_shape_score_numpy(
        hist_feature.astype(np.uint8),
        atlas_feature.astype(np.uint8),
        hist_feature.astype(np.uint8),
        atlas_feature.astype(np.uint8),
        dist_to_atlas,
        dist_to_hist,
        int(max_shift),
        int(hist_feature.sum()),
        int(atlas_feature.sum()),
        False,
    )


def _partial_shape_score_numpy(
    hist_shape: np.ndarray,
    atlas_shape: np.ndarray,
    hist_edge: np.ndarray,
    atlas_edge: np.ndarray,
    dist_to_atlas: np.ndarray,
    dist_to_hist: np.ndarray,
    max_shift: int,
    hist_area: int,
    atlas_area: int,
    shape_weights: bool,
) -> float:
    best = 0.0
    h, w = hist_shape.shape
    atlas_edge_area = max(1, int(atlas_edge.sum()))
    for dy in range(-max_shift, max_shift + 1, 4):
        for dx in range(-max_shift, max_shift + 1, 4):
            src_y0 = max(0, -dy)
            src_y1 = min(h, h - dy)
            dst_y0 = max(0, dy)
            dst_y1 = min(h, h + dy)
            src_x0 = max(0, -dx)
            src_x1 = min(w, w - dx)
            dst_x0 = max(0, dx)
            dst_x1 = min(w, w + dx)
            if src_y1 <= src_y0 or src_x1 <= src_x0:
                continue
            shifted = np.zeros_like(hist_shape, dtype=bool)
            shifted[dst_y0:dst_y1, dst_x0:dst_x1] = hist_shape[src_y0:src_y1, src_x0:src_x1] > 0
            shifted_edge = np.zeros_like(hist_edge, dtype=bool)
            shifted_edge[dst_y0:dst_y1, dst_x0:dst_x1] = hist_edge[src_y0:src_y1, src_x0:src_x1] > 0
            inter = int(np.logical_and(shifted, atlas_shape > 0).sum())
            denom = max(1, hist_area + atlas_area)
            dice = float(2.0 * inter / denom)
            observed_fit = float(inter / max(1, hist_area))
            atlas_coverage = float(inter / max(1, atlas_area))
            edge_count = int(shifted_edge.sum())
            if edge_count:
                edge_fit = float(np.exp(-np.mean(np.clip(dist_to_atlas[shifted_edge], 0, 12)) / 4.0))
            else:
                edge_fit = 0.0
            inv = np.full_like(atlas_edge, 12.0, dtype=np.float32)
            inv[dst_y0:dst_y1, dst_x0:dst_x1] = dist_to_hist[src_y0:src_y1, src_x0:src_x1]
            edge_back = float(np.exp(-np.sum(np.clip(inv[atlas_edge > 0], 0, 12)) / (4.0 * atlas_edge_area)))
            if shape_weights:
                score = (
                    0.46 * observed_fit
                    + 0.22 * edge_fit
                    + 0.16 * dice
                    + 0.10 * atlas_coverage
                    + 0.06 * edge_back
                )
            else:
                score = 0.44 * edge_fit + 0.38 * edge_back + 0.18 * dice
            best = max(best, float(score))
    return float(best)


if _HAS_NUMBA:

    @nb.njit(cache=True, fastmath=True)
    def _partial_score_numba(
        hist_shape,
        atlas_shape,
        hist_edge,
        atlas_edge,
        dist_to_atlas,
        dist_to_hist,
        max_shift,
        hist_area,
        atlas_area,
        shape_weights,
    ):
        h, w = hist_shape.shape
        atlas_edge_area = 0
        for y in range(h):
            for x in range(w):
                if atlas_edge[y, x] != 0:
                    atlas_edge_area += 1
        if atlas_edge_area < 1:
            atlas_edge_area = 1
        best = 0.0
        for dy in range(-max_shift, max_shift + 1, 4):
            for dx in range(-max_shift, max_shift + 1, 4):
                inter = 0
                shifted_edge_count = 0
                edge_sum = 0.0
                for y in range(h):
                    sy = y - dy
                    if sy < 0 or sy >= h:
                        continue
                    for x in range(w):
                        sx = x - dx
                        if sx < 0 or sx >= w:
                            continue
                        if hist_shape[sy, sx] != 0 and atlas_shape[y, x] != 0:
                            inter += 1
                        if hist_edge[sy, sx] != 0:
                            shifted_edge_count += 1
                            d = dist_to_atlas[y, x]
                            if d > 12.0:
                                d = 12.0
                            edge_sum += d
                if shifted_edge_count > 0:
                    edge_fit = np.exp(-(edge_sum / shifted_edge_count) / 4.0)
                else:
                    edge_fit = 0.0

                back_sum = 0.0
                for y in range(h):
                    sy = y - dy
                    for x in range(w):
                        if atlas_edge[y, x] == 0:
                            continue
                        sx = x - dx
                        if sy < 0 or sy >= h or sx < 0 or sx >= w:
                            d = 12.0
                        else:
                            d = dist_to_hist[sy, sx]
                            if d > 12.0:
                                d = 12.0
                        back_sum += d
                edge_back = np.exp(-(back_sum / atlas_edge_area) / 4.0)
                observed_fit = inter / max(1, hist_area)
                atlas_coverage = inter / max(1, atlas_area)
                dice = 2.0 * inter / max(1, hist_area + atlas_area)
                if shape_weights:
                    score = (
                        0.46 * observed_fit
                        + 0.22 * edge_fit
                        + 0.16 * dice
                        + 0.10 * atlas_coverage
                        + 0.06 * edge_back
                    )
                else:
                    score = 0.44 * edge_fit + 0.38 * edge_back + 0.18 * dice
                if score > best:
                    best = score
        return best

    @nb.njit(cache=True, parallel=True, fastmath=True)
    def _affine_overlap_scores_numba(hist_mask, hist_dist, atlas_y, atlas_x, atlas_edge_y, atlas_edge_x, transforms):
        n = transforms.shape[0]
        h, w = hist_mask.shape
        hist_sum = 0
        for y in range(h):
            for x in range(w):
                if hist_mask[y, x] != 0:
                    hist_sum += 1
        if hist_sum < 1:
            hist_sum = 1
        scores = np.empty(n, dtype=np.float64)
        for i in nb.prange(n):
            warped = np.zeros((h, w), dtype=np.uint8)
            atlas_sum = 0
            for k in range(atlas_y.shape[0]):
                x0 = float(atlas_x[k])
                y0 = float(atlas_y[k])
                x1 = x0 * transforms[i, 0, 0] + y0 * transforms[i, 1, 0] + transforms[i, 2, 0]
                y1 = x0 * transforms[i, 0, 1] + y0 * transforms[i, 1, 1] + transforms[i, 2, 1]
                xi = int(np.floor(x1 + 0.5))
                yi = int(np.floor(y1 + 0.5))
                if 0 <= yi < h and 0 <= xi < w and warped[yi, xi] == 0:
                    warped[yi, xi] = 1
                    atlas_sum += 1
            if atlas_sum < 1:
                scores[i] = 0.0
                continue
            inter = 0
            for y in range(h):
                for x in range(w):
                    if warped[y, x] != 0 and hist_mask[y, x] != 0:
                        inter += 1
            edge_count = 0
            edge_sum = 0.0
            for k in range(atlas_edge_y.shape[0]):
                x0 = float(atlas_edge_x[k])
                y0 = float(atlas_edge_y[k])
                x1 = x0 * transforms[i, 0, 0] + y0 * transforms[i, 1, 0] + transforms[i, 2, 0]
                y1 = x0 * transforms[i, 0, 1] + y0 * transforms[i, 1, 1] + transforms[i, 2, 1]
                xi = int(np.floor(x1 + 0.5))
                yi = int(np.floor(y1 + 0.5))
                if 0 <= yi < h and 0 <= xi < w:
                    d = hist_dist[yi, xi]
                    if d > 20.0:
                        d = 20.0
                    edge_sum += d
                    edge_count += 1
            if edge_count > 0:
                edge_fit = np.exp(-(edge_sum / edge_count) / 5.0)
            else:
                edge_fit = 0.0
            observed_fit = inter / hist_sum
            atlas_precision = inter / atlas_sum
            dice = 2.0 * inter / max(1, hist_sum + atlas_sum)
            area_ratio = atlas_sum / hist_sum
            if area_ratio < 1e-6:
                area_ratio = 1e-6
            area_penalty = abs(np.log(area_ratio))
            if area_penalty > 2.0:
                area_penalty = 2.0
            scores[i] = (
                0.40 * observed_fit
                + 0.28 * edge_fit
                + 0.18 * dice
                + 0.14 * atlas_precision
                - 0.05 * area_penalty
            )
        return scores


def partial_shape_score(hist_shape: np.ndarray, atlas_shape: np.ndarray, max_shift: int = 8) -> float:
    hist_shape = np.asarray(hist_shape, bool)
    atlas_shape = np.asarray(atlas_shape, bool)
    if not hist_shape.any() or not atlas_shape.any():
        return 0.0
    hist_edge = _edge_mask(hist_shape)
    atlas_edge = _edge_mask(atlas_shape)
    dist_to_atlas = ndi.distance_transform_edt(~atlas_edge).astype(np.float32)
    dist_to_hist = ndi.distance_transform_edt(~hist_edge).astype(np.float32)
    if _HAS_NUMBA:
        try:
            return float(
                _partial_score_numba(
                    hist_shape.astype(np.uint8),
                    atlas_shape.astype(np.uint8),
                    hist_edge.astype(np.uint8),
                    atlas_edge.astype(np.uint8),
                    dist_to_atlas,
                    dist_to_hist,
                    int(max_shift),
                    int(hist_shape.sum()),
                    int(atlas_shape.sum()),
                    True,
                )
            )
        except Exception:
            pass
    return _partial_shape_score_python(hist_shape, atlas_shape, max_shift)


def partial_feature_score(hist_feature: np.ndarray, atlas_feature: np.ndarray, max_shift: int = 8) -> float:
    hist_feature = np.asarray(hist_feature, bool)
    atlas_feature = np.asarray(atlas_feature, bool)
    if not hist_feature.any() or not atlas_feature.any():
        return 0.0
    dist_to_atlas = ndi.distance_transform_edt(~atlas_feature).astype(np.float32)
    dist_to_hist = ndi.distance_transform_edt(~hist_feature).astype(np.float32)
    if _HAS_NUMBA:
        try:
            return float(
                _partial_score_numba(
                    hist_feature.astype(np.uint8),
                    atlas_feature.astype(np.uint8),
                    hist_feature.astype(np.uint8),
                    atlas_feature.astype(np.uint8),
                    dist_to_atlas,
                    dist_to_hist,
                    int(max_shift),
                    int(hist_feature.sum()),
                    int(atlas_feature.sum()),
                    False,
                )
            )
        except Exception:
            pass
    return _partial_feature_score_python(hist_feature, atlas_feature, max_shift)


def rank_affine_candidates(
    hist_mask: np.ndarray,
    atlas_mask: np.ndarray,
    transforms: Sequence[np.ndarray],
    keep: int,
) -> list[int] | None:
    """Return candidate indices ranked by a fast Numba overlap pre-score."""
    if not _HAS_NUMBA or not transforms:
        return None
    hist_mask_u8 = np.asarray(hist_mask, bool).astype(np.uint8)
    atlas_mask_bool = np.asarray(atlas_mask, bool)
    atlas_y, atlas_x = np.nonzero(atlas_mask_bool)
    if atlas_y.size == 0:
        return None
    hist_edge = _edge_mask(hist_mask_u8)
    hist_dist = ndi.distance_transform_edt(~hist_edge).astype(np.float32)
    atlas_edge = _edge_mask(atlas_mask_bool)
    atlas_edge_y, atlas_edge_x = np.nonzero(atlas_edge)
    if atlas_edge_y.size == 0:
        atlas_edge_y, atlas_edge_x = atlas_y, atlas_x
    try:
        scores = _affine_overlap_scores_numba(
            hist_mask_u8,
            hist_dist,
            atlas_y.astype(np.int64),
            atlas_x.astype(np.int64),
            atlas_edge_y.astype(np.int64),
            atlas_edge_x.astype(np.int64),
            np.asarray(transforms, dtype=np.float64),
        )
    except Exception:
        return None
    n_keep = max(1, min(int(keep), len(transforms)))
    return [int(i) for i in np.argsort(scores)[::-1][:n_keep]]
