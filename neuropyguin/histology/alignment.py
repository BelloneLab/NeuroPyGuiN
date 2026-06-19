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

try:  # OpenCV is optional; manual alignment works without it.
    import cv2  # type: ignore
    _HAS_CV2 = True
except Exception:  # pragma: no cover
    _HAS_CV2 = False


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


def auto_align(
    histology_gray: np.ndarray,
    atlas_tv: np.ndarray,
    downsample: int = 5,
    max_iter: int = 200,
) -> np.ndarray:
    """Intensity-based affine alignment of atlas template -> histology.

    Port of ``align_auto_histology_atlas.m``: resize the atlas to roughly match
    the histology, run a multi-resolution affine registration, and return the
    full-resolution 3x3 ``T`` (atlas -> histology). Best effort; returns a pure
    resize transform if registration cannot converge or OpenCV is missing.
    """
    hist = np.nan_to_num(np.asarray(histology_gray, dtype=np.float32))
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

    hist = np.nan_to_num(np.asarray(histology_gray, dtype=np.float32))
    atlas = np.nan_to_num(np.asarray(atlas_tv, dtype=np.float32))
    if hist.ndim != 2 or atlas.ndim != 2 or hist.size == 0 or atlas.size == 0:
        return np.eye(3), "auto-align skipped (unexpected slice shape); used identity."
    fallback = resize_only_transform(hist.shape, atlas.shape)

    tmp = Path(tempfile.mkdtemp(prefix="npx_autoalign_"))
    try:
        hp, ap, op = tmp / "hist.npy", tmp / "atlas.npy", tmp / "T.npy"
        np.save(hp, hist)
        np.save(ap, atlas)
        repo_root = Path(__file__).resolve().parents[2]
        cmd = [python_exe or sys.executable, "-m", "neuropyguin.histology.alignment",
               str(hp), str(ap), str(op)]
        try:
            proc = subprocess.run(
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
        return T, "auto-aligned (isolated registration)."
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
