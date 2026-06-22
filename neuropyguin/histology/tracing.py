"""Probe-track tracing -> ``probe_ccf``.

Native port of ``+ap_histology/annotate_neuropixels.m`` (the ``close_gui`` save
path) and ``AP_histology2ccf.m``. Given probe lines drawn on histology slices,
the matched CCF planes (``histology_ccf``) and the atlas->histology affine
transforms, it produces, per probe:

* ``points``            (N, 3) CCF voxel coords [AP, DV, ML], sorted top->bottom
* ``trajectory_areas``  DataFrame of regions crossed by the best-fit line, with
                        ``depth_start_um`` / ``depth_end_um`` (0 at the first
                        in-brain boundary)
* ``trajectory_coords`` (2, 3) CCF coords of the trajectory's brain entry/exit

Coordinate conventions match AP_histology: plane grids and ``points`` carry
1-based CCF voxel coordinates; volumes are indexed ``[AP, DV, ML]``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .atlas import AllenCCFAtlas


def probe_colormap(n: int = 30) -> np.ndarray:
    """Probe colour map mirroring ``annotate_neuropixels.m`` (``lines(7)`` recombined).

    Returns an (M, 3) RGB array with at least ``max(n, base_rows)`` rows. The
    seven MATLAB ``lines`` colours are recombined by channel permutation so that
    many probes stay visually distinct.
    """
    base = np.array([
        [0.0000, 0.4470, 0.7410],
        [0.8500, 0.3250, 0.0980],
        [0.9290, 0.6940, 0.1250],
        [0.4940, 0.1840, 0.5560],
        [0.4660, 0.6740, 0.1880],
        [0.3010, 0.7450, 0.9330],
        [0.6350, 0.0780, 0.1840],
    ])
    cmap = np.vstack([base, base[:, [1, 2, 0]], base[:, [2, 0, 1]], base, base[:, [1, 2, 0]]])
    return cmap[:max(n, cmap.shape[0])]


def _apply_tform(T: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Apply a MATLAB row-vector affine ``T`` (3x3) to (N, 2) points: [x y 1] @ T."""
    xy = np.atleast_2d(np.asarray(xy, dtype=np.float64))
    homog = np.column_stack([xy, np.ones(len(xy))])
    out = homog @ T
    return out[:, :2]


def histology_points_to_ccf(
    points_per_slice: Sequence[Optional[np.ndarray]],
    histology_ccf: Sequence[Dict[str, np.ndarray]],
    tforms: Sequence[np.ndarray],
) -> List[Optional[np.ndarray]]:
    """Map histology-pixel points to CCF coords (port of ``AP_histology2ccf.m``).

    ``points_per_slice[i]`` is an (N, 2) array of (x, y) pixel coords on slice
    ``i`` (or None). Returns a parallel list of (N, 3) CCF coords ``[AP, DV, ML]``
    (NaN rows where a point fell outside the slice grid).
    """
    out: List[Optional[np.ndarray]] = [None] * len(points_per_slice)
    for s, pts in enumerate(points_per_slice):
        if pts is None or len(pts) == 0:
            continue
        T = np.asarray(tforms[s], dtype=np.float64)
        Tinv = np.linalg.inv(T)  # CCF->histology stored, invert for histology->CCF
        atlas_xy = _apply_tform(Tinv, pts)
        ax = np.round(atlas_xy[:, 0]).astype(np.int64)
        ay = np.round(atlas_xy[:, 1]).astype(np.int64)
        plane_ap = histology_ccf[s]["plane_ap"]
        plane_ml = histology_ccf[s]["plane_ml"]
        plane_dv = histology_ccf[s]["plane_dv"]
        rows, cols = plane_ap.shape
        ccf = np.full((len(pts), 3), np.nan, dtype=np.float64)
        for k in range(len(pts)):
            xi, yi = ax[k] - 1, ay[k] - 1  # plane coords are 1-based -> numpy 0-based
            if 0 <= yi < rows and 0 <= xi < cols:
                ccf[k] = [plane_ap[yi, xi], plane_dv[yi, xi], plane_ml[yi, xi]]
        out[s] = ccf
    return out


def _best_fit_direction(points: np.ndarray) -> np.ndarray:
    """First singular vector through the points, oriented to go down in DV."""
    r0 = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - r0, full_matrices=False)
    direction = vt[0]
    if direction[1] < 0:  # ensure DV-increasing (top -> bottom)
        direction = -direction
    return direction


def _line_box_interval(r0: np.ndarray, direction: np.ndarray, shape: Tuple[int, int, int]) -> Optional[Tuple[float, float]]:
    """Return the parameter interval where a 3D line intersects the atlas box."""
    lo = np.ones(3, dtype=np.float64)
    hi = np.asarray(shape, dtype=np.float64)
    t_min = -np.inf
    t_max = np.inf
    for dim in range(3):
        d = float(direction[dim])
        if abs(d) < 1e-12:
            if r0[dim] < lo[dim] or r0[dim] > hi[dim]:
                return None
            continue
        a = (lo[dim] - r0[dim]) / d
        b = (hi[dim] - r0[dim]) / d
        if a > b:
            a, b = b, a
        t_min = max(t_min, a)
        t_max = min(t_max, b)
        if t_min > t_max:
            return None
    return float(t_min), float(t_max)


def trajectory_areas_from_points(
    points: np.ndarray, atlas: AllenCCFAtlas, sample_um: float = 10.0
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Sample the Allen annotation along the best-fit line through ``points``.

    Returns ``(trajectory_areas_df, trajectory_coords)``. Port of the
    trajectory-area block in ``annotate_neuropixels.m:close_gui``.
    """
    av = atlas.av
    ap_n, dv_n, ml_n = av.shape

    r0 = points.mean(axis=0)
    direction = _best_fit_direction(points)
    interval = _line_box_interval(r0, direction, av.shape)
    empty_df = pd.DataFrame(columns=[
        "acronym", "name", "id", "color_hex_triplet", "depth_start_um", "depth_end_um",
    ])
    if interval is None:
        return empty_df, np.zeros((0, 3))
    line_eval = np.array(interval, dtype=np.float64)
    fit_line = r0[None, :] + line_eval[:, None] * direction[None, :]  # (2,3)

    sample_um = max(float(sample_um), 1.0)
    length_um = float(np.linalg.norm(np.diff(fit_line, axis=0)) * 10.0)
    n_coords = max(int(np.ceil(length_um / sample_um)) + 1, 2)
    traj = np.vstack([
        np.round(np.linspace(fit_line[0, d], fit_line[1, d], n_coords)) for d in range(3)
    ]).T.astype(np.int64)  # (n_coords, 3) [AP, DV, ML]

    in_bounds = (
        (traj[:, 0] >= 1) & (traj[:, 0] <= ap_n)
        & (traj[:, 1] >= 1) & (traj[:, 1] <= dv_n)
        & (traj[:, 2] >= 1) & (traj[:, 2] <= ml_n)
    )
    coords = traj[in_bounds]
    if len(coords) == 0:
        return empty_df, np.zeros((0, 3))

    area_idx_sampled = np.asarray(av[coords[:, 0] - 1, coords[:, 1] - 1, coords[:, 2] - 1])

    # Boundaries between contiguous runs of the same area.
    change = np.flatnonzero(np.diff(area_idx_sampled.astype(np.int64)) != 0) + 1
    bins = np.concatenate([[0], change, [len(area_idx_sampled)]])
    boundaries = np.column_stack([bins[:-1], bins[1:]])  # [start, end) per run
    run_area_idx = area_idx_sampled[boundaries[:, 0]]

    store = run_area_idx > 1  # only regions inside the brain (idx > 1)
    if not np.any(store):
        return empty_df, np.zeros((0, 3))

    first_in_brain_start = boundaries[np.flatnonzero(store)[0], 0]
    depth = (boundaries[store] - first_in_brain_start).astype(float) * sample_um

    rows = []
    row_cache = {}
    for v in run_area_idx[store]:
        vi = int(v)
        if vi not in row_cache:
            row_cache[vi] = atlas.region_row(vi)
        r = row_cache[vi]
        rows.append({
            "acronym": "" if r is None else str(r["acronym"]),
            "name": "" if r is None else str(r["name"]),
            "id": np.nan if r is None else int(r["id"]),
            "color_hex_triplet": "" if r is None else str(r["color_hex_triplet"]),
        })
    df = pd.DataFrame(rows)
    df["depth_start_um"] = depth[:, 0].astype(float)
    df["depth_end_um"] = depth[:, 1].astype(float)

    # Brain entry/exit CCF coords (first start, last end of in-brain runs).
    store_idx = np.flatnonzero(store)
    entry = coords[boundaries[store_idx[0], 0]]
    exit_i = min(boundaries[store_idx[-1], 1] - 1, len(coords) - 1)
    traj_coords = np.vstack([entry, coords[exit_i]]).astype(np.float64)
    return df, traj_coords


def build_probe_ccf(
    probe_points_histology: Dict[Tuple[int, int], np.ndarray],
    histology_ccf: Sequence[Dict[str, np.ndarray]],
    tforms: Sequence[np.ndarray],
    atlas: AllenCCFAtlas,
    n_probes: Optional[int] = None,
) -> List[Dict]:
    """Assemble ``probe_ccf`` for every probe.

    ``probe_points_histology`` maps ``(slice_idx, probe_idx)`` (both 0-based) to
    an (N, 2) array of histology pixel coords. Returns a list of probe dicts
    (one per probe index, 0..n_probes-1) ready for :func:`io_formats.save_probe_ccf`.
    """
    if n_probes is None:
        n_probes = (max((p for _, p in probe_points_histology), default=-1) + 1)

    probes: List[Dict] = []
    for probe in range(n_probes):
        # Gather this probe's pixel points per slice.
        per_slice: List[Optional[np.ndarray]] = [None] * len(histology_ccf)
        for (s, p), pts in probe_points_histology.items():
            if p == probe and pts is not None and len(pts):
                per_slice[s] = np.asarray(pts, dtype=np.float64)

        ccf_per_slice = histology_points_to_ccf(per_slice, histology_ccf, tforms)
        all_pts = [c[~np.isnan(c).any(axis=1)] for c in ccf_per_slice if c is not None]
        all_pts = [c for c in all_pts if len(c)]
        if not all_pts:
            probes.append({"points": np.zeros((0, 3)), "trajectory_areas": pd.DataFrame(),
                           "trajectory_coords": np.zeros((0, 3))})
            continue
        points = np.vstack(all_pts)
        points = points[np.argsort(points[:, 1])]  # sort by DV (top -> bottom)

        if len(points) >= 2:
            areas, coords = trajectory_areas_from_points(points, atlas)
        else:
            areas, coords = pd.DataFrame(), np.zeros((0, 3))
        probes.append({"points": points, "trajectory_areas": areas, "trajectory_coords": coords})
    return probes
