"""Read/write the AP_histology data products and their CSV exports.

Files handled (kept compatible with the original MATLAB toolbox and the IBL
glue scripts):

* ``histology_ccf.mat``        per-slice CCF planes (HDF5 / v7.3 layout)
* ``atlas2histology_tform.mat`` per-slice 3x3 affine matrices (atlas -> histology)
* ``probe_ccf.mat``           per-probe CCF points + trajectory areas

Plus CSV exports of the tabular content (``probe_ccf.csv``,
``probe_ccf_points.csv``, ``histology_ccf.csv``).

Orientation note: ``grab_atlas_slice`` returns arrays shaped ``(n_dv, n_ml)``
(same as MATLAB). HDF5/v7.3 stores column-major, so h5py reads such an array
transposed; we transpose on the way in and out so the in-memory arrays always
match the AP_histology / numpy orientation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import h5py
import numpy as np
import pandas as pd
import scipy.io as sio


# ---------------------------------------------------------------------------
# histology_ccf.mat
# ---------------------------------------------------------------------------

_HCCF_FIELDS = ("tv_slices", "av_slices", "plane_ap", "plane_ml", "plane_dv")


def save_histology_ccf(path: str | Path, slices: Sequence[Dict[str, np.ndarray]]) -> None:
    """Write ``histology_ccf`` (list of per-slice dicts) as a v7.3-style HDF5 .mat.

    For a single slice each field is written as a direct 2D dataset (exactly the
    MATLAB scalar-struct layout). For multiple slices each field is an array of
    object references into ``/#refs#`` (the MATLAB struct-array layout).
    """
    path = Path(path)
    n = len(slices)
    with h5py.File(path, "w") as f:
        f.attrs["MATLAB_class"] = np.bytes_(b"struct")
        grp = f.create_group("histology_ccf")
        grp.attrs["MATLAB_class"] = np.bytes_(b"struct")
        grp.attrs["MATLAB_fields"] = _matlab_fields_attr(_HCCF_FIELDS)

        if n == 1:
            sl = slices[0]
            for fld in _HCCF_FIELDS:
                d = grp.create_dataset(fld, data=np.asarray(sl[fld], np.float64).T)
                d.attrs["MATLAB_class"] = np.bytes_(b"double")
            return

        refs = f.create_group("#refs#")
        for fld in _HCCF_FIELDS:
            ref_arr = np.empty((n, 1), dtype=h5py.ref_dtype)
            for i, sl in enumerate(slices):
                ds = refs.create_dataset(
                    f"{fld}_{i}", data=np.asarray(sl[fld], np.float64).T
                )
                ds.attrs["MATLAB_class"] = np.bytes_(b"double")
                ref_arr[i, 0] = ds.ref
            d = grp.create_dataset(fld, data=ref_arr)
            d.attrs["MATLAB_class"] = np.bytes_(b"cell")


def load_histology_ccf(path: str | Path) -> List[Dict[str, np.ndarray]]:
    """Load ``histology_ccf.mat`` (our layout or the original MATLAB v7.3 layout)."""
    path = Path(path)
    with h5py.File(path, "r") as f:
        grp = f["histology_ccf"]
        first = grp[_HCCF_FIELDS[0]]
        if isinstance(first, h5py.Dataset) and first.dtype == h5py.ref_dtype:
            n = first.shape[0]
            out: List[Dict[str, np.ndarray]] = []
            for i in range(n):
                sl = {}
                for fld in _HCCF_FIELDS:
                    ref = grp[fld][i, 0]
                    sl[fld] = np.asarray(f[ref]).T.astype(np.float64)
                out.append(sl)
            return out
        # Single scalar struct: each field a direct dataset.
        sl = {fld: np.asarray(grp[fld]).T.astype(np.float64) for fld in _HCCF_FIELDS}
        return [sl]


def _matlab_fields_attr(fields: Sequence[str]) -> np.ndarray:
    """Build the MATLAB_fields attribute (array of variable-length uint8 names)."""
    dt = h5py.vlen_dtype(np.dtype("uint8"))
    arr = np.empty((len(fields),), dtype=dt)
    for i, name in enumerate(fields):
        arr[i] = np.frombuffer(name.encode("ascii"), dtype=np.uint8)
    return arr


# ---------------------------------------------------------------------------
# atlas2histology_tform.mat
# ---------------------------------------------------------------------------

def save_tforms(path: str | Path, tforms: Sequence[Optional[np.ndarray]]) -> None:
    """Save per-slice 3x3 affine ``T`` matrices (MATLAB row-vector convention).

    Stored as a 1xN MATLAB cell array named ``atlas2histology_tform`` so the
    original toolbox and ``AP_histology2ccf.m`` keep working.
    """
    path = Path(path)
    cell = np.empty((1, len(tforms)), dtype=object)
    for i, t in enumerate(tforms):
        cell[0, i] = np.eye(3) if t is None else np.asarray(t, dtype=np.float64)
    sio.savemat(str(path), {"atlas2histology_tform": cell})


def load_tforms(path: str | Path) -> List[np.ndarray]:
    """Load per-slice 3x3 ``T`` matrices from ``atlas2histology_tform.mat``."""
    path = Path(path)
    try:
        mat = sio.loadmat(str(path))
        cell = mat["atlas2histology_tform"]
        out = []
        for i in range(cell.shape[1]):
            out.append(np.asarray(cell[0, i], dtype=np.float64))
        return out
    except NotImplementedError:
        # v7.3 HDF5 fallback
        with h5py.File(path, "r") as f:
            grp = f["atlas2histology_tform"]
            out = []
            for i in range(grp.shape[0]):
                out.append(np.asarray(f[grp[i, 0]]).T.astype(np.float64))
            return out


# ---------------------------------------------------------------------------
# probe_ccf.mat
# ---------------------------------------------------------------------------

def save_probe_ccf(path: str | Path, probes: Sequence[Dict]) -> None:
    """Save the per-probe CCF result.

    ``probes[i]`` keys:
        points            (N, 3) float, CCF voxel [AP, DV, ML]
        trajectory_areas  dict/DataFrame with acronym/id/color/depth (optional)
        trajectory_coords (2, 3) float (optional)

    Written so ``mat['probe_ccf'][p][0][0]`` returns ``points`` (the convention
    ``D:\\NPX\\process_histology.py`` relies on: ``points`` is the first field).
    """
    path = Path(path)
    n = len(probes)
    dt = np.dtype([("points", "O"), ("trajectory_areas", "O"), ("trajectory_coords", "O")])
    arr = np.empty((n, 1), dtype=dt)
    for i, p in enumerate(probes):
        pts = np.asarray(p.get("points", np.zeros((0, 3))), dtype=np.float64)
        arr[i, 0]["points"] = pts
        arr[i, 0]["trajectory_areas"] = _areas_to_struct(p.get("trajectory_areas"))
        coords = p.get("trajectory_coords")
        arr[i, 0]["trajectory_coords"] = (
            np.zeros((0, 3)) if coords is None else np.asarray(coords, dtype=np.float64)
        )
    sio.savemat(str(path), {"probe_ccf": arr}, do_compression=True)


def load_probe_ccf_points(path: str | Path) -> List[np.ndarray]:
    """Load just the per-probe ``points`` arrays (same access as process_histology)."""
    path = Path(path)
    mat = sio.loadmat(str(path))
    pc = mat["probe_ccf"]
    return [np.asarray(pc[p][0][0], dtype=np.float64) for p in range(pc.shape[0])]


def _areas_depth_matrix(areas) -> np.ndarray:
    """Return an (N, 2) [start, end] depth matrix from a DataFrame or dict."""
    if isinstance(areas, pd.DataFrame):
        if "depth_start_um" in areas and "depth_end_um" in areas:
            return np.column_stack([
                areas["depth_start_um"].to_numpy(dtype=np.float64),
                areas["depth_end_um"].to_numpy(dtype=np.float64),
            ]) if len(areas) else np.zeros((0, 2))
        td = areas.get("trajectory_depth")
        if td is not None:
            return np.array([np.asarray(v, dtype=np.float64).ravel()[:2] for v in td]) \
                if len(areas) else np.zeros((0, 2))
        return np.zeros((len(areas), 2))
    depth = np.asarray((areas or {}).get("trajectory_depth", np.zeros((0, 2))), dtype=np.float64)
    return depth.reshape(-1, 2) if depth.size else np.zeros((0, 2))


def _areas_to_struct(areas) -> np.ndarray:
    """Convert a trajectory-areas table/dict to a (1,1) MATLAB-style struct."""
    depth = _areas_depth_matrix(areas)
    if areas is None:
        areas = {"acronym": [], "name": [], "id": [], "color_hex_triplet": []}
    if isinstance(areas, pd.DataFrame):
        areas = {
            "acronym": areas.get("acronym", pd.Series([], dtype=str)).tolist(),
            "name": areas.get("name", pd.Series([], dtype=str)).tolist(),
            "id": areas.get("id", pd.Series([], dtype=int)).tolist(),
            "color_hex_triplet": areas.get("color_hex_triplet", pd.Series([], dtype=str)).tolist(),
            "trajectory_depth": depth,
        }
    else:
        areas = dict(areas)
        areas["trajectory_depth"] = depth
    acronym = np.array(list(areas.get("acronym", [])), dtype=object)
    name = np.array(list(areas.get("name", [])), dtype=object)
    ids = np.array(list(areas.get("id", [])), dtype=np.float64)
    colors = np.array(list(areas.get("color_hex_triplet", [])), dtype=object)
    depth = np.asarray(areas.get("trajectory_depth", np.zeros((0, 2))), dtype=np.float64)
    dt = np.dtype([
        ("acronym", "O"), ("name", "O"), ("id", "O"),
        ("color_hex_triplet", "O"), ("trajectory_depth", "O"),
    ])
    s = np.empty((1, 1), dtype=dt)
    s[0, 0]["acronym"] = acronym.reshape(-1, 1)
    s[0, 0]["name"] = name.reshape(-1, 1)
    s[0, 0]["id"] = ids.reshape(-1, 1)
    s[0, 0]["color_hex_triplet"] = colors.reshape(-1, 1)
    s[0, 0]["trajectory_depth"] = depth
    return s


# ---------------------------------------------------------------------------
# CSV exports
# ---------------------------------------------------------------------------

def export_probe_ccf_csv(folder: str | Path, probes: Sequence[Dict]) -> Dict[str, Path]:
    """Write ``probe_ccf.csv`` (trajectory areas) and ``probe_ccf_points.csv``."""
    folder = Path(folder)
    area_rows = []
    point_rows = []
    for pi, p in enumerate(probes, start=1):
        ta = p.get("trajectory_areas")
        if isinstance(ta, pd.DataFrame) and len(ta):
            depth = _areas_depth_matrix(ta)
            for j in range(len(ta)):
                area_rows.append({
                    "probe": pi,
                    "acronym": ta.iloc[j].get("acronym", ""),
                    "name": ta.iloc[j].get("name", ""),
                    "region_id": ta.iloc[j].get("id", ""),
                    "color_hex_triplet": ta.iloc[j].get("color_hex_triplet", ""),
                    "depth_start_um": float(depth[j, 0]) if depth.size else np.nan,
                    "depth_end_um": float(depth[j, 1]) if depth.size else np.nan,
                })
        pts = np.asarray(p.get("points", np.zeros((0, 3))), dtype=np.float64)
        for k in range(pts.shape[0]):
            point_rows.append({
                "probe": pi, "ccf_ap": pts[k, 0], "ccf_dv": pts[k, 1], "ccf_ml": pts[k, 2],
            })

    areas_fn = folder / "probe_ccf.csv"
    points_fn = folder / "probe_ccf_points.csv"
    pd.DataFrame(area_rows, columns=[
        "probe", "acronym", "name", "region_id", "color_hex_triplet",
        "depth_start_um", "depth_end_um",
    ]).to_csv(areas_fn, index=False)
    pd.DataFrame(point_rows, columns=["probe", "ccf_ap", "ccf_dv", "ccf_ml"]).to_csv(
        points_fn, index=False
    )
    return {"areas": areas_fn, "points": points_fn}


def export_histology_ccf_csv(folder: str | Path, slices: Sequence[Dict[str, np.ndarray]]) -> Path:
    """Write ``histology_ccf.csv`` with a per-slice plane summary."""
    folder = Path(folder)
    rows = []
    for i, sl in enumerate(slices, start=1):
        brain = np.isfinite(sl["tv_slices"])
        n_brain = int(np.count_nonzero(brain))
        def _ctr(key):
            vals = sl[key][brain]
            return float(np.nanmean(vals)) if vals.size else np.nan
        rows.append({
            "slice": i,
            "n_brain_px": n_brain,
            "ccf_ap_center": _ctr("plane_ap"),
            "ccf_dv_center": _ctr("plane_dv"),
            "ccf_ml_center": _ctr("plane_ml"),
            "position_mm": float(np.nanmean(sl["plane_ap"][brain]) / 100.0) if n_brain else np.nan,
        })
    fn = folder / "histology_ccf.csv"
    pd.DataFrame(rows).to_csv(fn, index=False)
    return fn
