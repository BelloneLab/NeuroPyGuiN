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

from .atlas import AllenCCFAtlas


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
