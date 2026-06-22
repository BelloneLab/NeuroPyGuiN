"""Allen Mouse Brain CCF access for the histology workflow.

This is a faithful, native-Python port of the atlas handling in AP_histology
(`+ap_histology/loadStructureTree.m`, `match_histology_atlas.m:grab_atlas_slice`,
`allenCCFbregma.m`). It loads the same three files AP_histology uses so the
results stay byte-compatible with the original toolbox:

    template_volume_10um.npy             template (grayscale) volume   (AP, DV, ML)
    annotation_volume_10um_by_index.npy  annotation volume, by st-row  (AP, DV, ML)
    structure_tree_safe_2017.csv         Allen structure ontology table

Coordinate conventions (kept identical to AP_histology so downstream MATLAB /
IBL scripts keep working):

* Volumes are indexed ``[AP, DV, ML]``.
* Plane coordinate grids (``plane_ap`` etc.) hold **1-based** voxel coordinates
  (MATLAB convention). When indexing the numpy volumes we subtract 1.
* ``annotation_volume_..._by_index`` stores, per voxel, the 1-based MATLAB row
  number of the structure-tree table. In Python that is ``structure_tree.iloc[v - 1]``.

Download the atlas (if missing) from https://osf.io/fv7ed/overview and point the
``histology/atlas_path`` setting (or ``NPG_ATLAS_PATH`` env var) at the folder.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

#: Default location of the Allen CCF files (the AP_histology install on this box).
DEFAULT_ATLAS_PATH = r"D:\AP_histology\allen_atlas_path"

#: CCF voxel size in micrometres.
CCF_VOXEL_UM = 10.0

#: Bregma in CCF 10um voxel coordinates [AP, DV, ML] (allenCCFbregma.m).
BREGMA_CCF = np.array([540.0, 0.0, 570.0])

_TEMPLATE_FN = "template_volume_10um.npy"
_ANNOTATION_FN = "annotation_volume_10um_by_index.npy"
_STRUCTURE_FN = "structure_tree_safe_2017.csv"


def resolve_atlas_path(atlas_path: Optional[str] = None) -> Path:
    """Resolve the atlas folder from an explicit arg, env var, or default."""
    candidate = atlas_path or os.environ.get("NPG_ATLAS_PATH") or DEFAULT_ATLAS_PATH
    return Path(candidate)


def atlas_files_present(atlas_path: Optional[str] = None) -> bool:
    """Return True only if all three required atlas files exist in the folder."""
    base = resolve_atlas_path(atlas_path)
    return all((base / fn).exists() for fn in (_TEMPLATE_FN, _ANNOTATION_FN, _STRUCTURE_FN))


def load_structure_tree(fn: str | Path) -> pd.DataFrame:
    """Load ``structure_tree_safe_2017.csv`` as AP_histology's loadStructureTree does.

    A leading ``index`` column (0..N-1) is prepended so that ``df.iloc[v - 1]``
    matches MATLAB ``st(v, :)`` for an annotation value ``v``.
    """
    df = pd.read_csv(fn, dtype={"color_hex_triplet": str}, keep_default_na=False)
    df.insert(0, "index", np.arange(len(df)))
    # Normalise the colour column: occasionally a leading zero is dropped (5 chars).
    def _fix_hex(value: str) -> str:
        v = str(value).strip()
        if len(v) == 5:
            return "0" + v
        if len(v) < 6:
            return v.rjust(6, "0")
        return v[:6]

    df["color_hex_triplet"] = df["color_hex_triplet"].map(_fix_hex)
    return df


def structure_tree_rgb(structure_tree: pd.DataFrame) -> np.ndarray:
    """Return an (N, 3) float RGB array (0..1) for the structure tree, row-aligned."""
    hexes = structure_tree["color_hex_triplet"].to_numpy()
    rgb = np.zeros((len(hexes), 3), dtype=np.float64)
    for i, h in enumerate(hexes):
        h = str(h)
        try:
            rgb[i] = [int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)]
        except (ValueError, IndexError):
            rgb[i] = [25, 25, 25]
    return rgb / 255.0


class AllenCCFAtlas:
    """Lazy, memory-mapped access to the Allen CCF 10um volumes."""

    def __init__(self, atlas_path: Optional[str] = None, mmap: bool = True) -> None:
        self.atlas_path = resolve_atlas_path(atlas_path)
        self._mmap = mmap
        self.tv: Optional[np.ndarray] = None
        self.av: Optional[np.ndarray] = None
        self.structure_tree: Optional[pd.DataFrame] = None
        self._st_rgb: Optional[np.ndarray] = None
        self._load()

    # -- loading -------------------------------------------------------------
    def _load(self) -> None:
        missing = [
            fn
            for fn in (_TEMPLATE_FN, _ANNOTATION_FN, _STRUCTURE_FN)
            if not (self.atlas_path / fn).exists()
        ]
        if missing:
            raise FileNotFoundError(
                "Allen CCF atlas files missing from "
                f"{self.atlas_path}: {missing}. Download them from "
                "https://osf.io/fv7ed/overview and set the atlas path."
            )
        mode = "r" if self._mmap else None
        self.tv = np.load(self.atlas_path / _TEMPLATE_FN, mmap_mode=mode)
        self.av = np.load(self.atlas_path / _ANNOTATION_FN, mmap_mode=mode)
        self.structure_tree = load_structure_tree(self.atlas_path / _STRUCTURE_FN)
        self._st_rgb = structure_tree_rgb(self.structure_tree)

    # -- basic geometry ------------------------------------------------------
    @property
    def shape(self) -> Tuple[int, int, int]:
        """Volume shape ``(AP, DV, ML)``."""
        return tuple(int(s) for s in self.tv.shape)  # type: ignore[union-attr]

    @property
    def st_rgb(self) -> np.ndarray:
        """Cached (N, 3) float RGB array (0..1) aligned with the structure tree."""
        return self._st_rgb  # type: ignore[return-value]

    # -- structure-tree lookups ---------------------------------------------
    def region_row(self, av_index: int) -> Optional[pd.Series]:
        """Structure-tree row for an annotation value (1-based MATLAB index)."""
        if av_index is None or av_index < 1 or av_index > len(self.structure_tree):
            return None
        return self.structure_tree.iloc[int(av_index) - 1]

    def acronym(self, av_index: int) -> str:
        """Acronym for an annotation value, or "" if it is out of range."""
        row = self.region_row(av_index)
        return "" if row is None else str(row["acronym"])

    # -- slice extraction (port of grab_atlas_slice) -------------------------
    def grab_atlas_slice(
        self,
        slice_point: np.ndarray,
        camera_vector: np.ndarray,
        spacing: int = 1,
    ) -> dict:
        """Sample a planar slice through the volume.

        Faithful port of ``match_histology_atlas.m:grab_atlas_slice``. The plane
        passes through ``slice_point`` (CCF 1-based voxel coords ``[AP, DV, ML]``)
        with normal ``camera_vector``. Returns a dict with ``tv_slices``,
        ``av_slices`` (NaN outside the brain) and ``plane_ap/ml/dv`` grids that
        carry 1-based CCF coordinates, plus ``position_mm`` for the title bar.
        """
        tv = self.tv
        av = self.av
        ap_n, dv_n, ml_n = tv.shape  # type: ignore[union-attr]

        cv = np.asarray(camera_vector, dtype=np.float64).reshape(3)
        sp = np.asarray(slice_point, dtype=np.float64).reshape(3)
        plane_offset = -float(cv @ sp)

        cam_plane = int(np.argmax(np.abs(cv / np.linalg.norm(cv))))  # 0:AP 1:DV 2:ML

        if cam_plane == 0:
            ml_rng = np.arange(1, ml_n + 1, spacing, dtype=np.float64)
            dv_rng = np.arange(1, dv_n + 1, spacing, dtype=np.float64)
            plane_ml, plane_dv = np.meshgrid(ml_rng, dv_rng)
            plane_ap = (cv[1] * plane_ml + cv[2] * plane_dv + plane_offset) / -cv[0]
        elif cam_plane == 1:
            ap_rng = np.arange(1, ap_n + 1, spacing, dtype=np.float64)
            dv_rng = np.arange(1, dv_n + 1, spacing, dtype=np.float64)
            plane_ap, plane_dv = np.meshgrid(ap_rng, dv_rng)
            plane_ml = (cv[0] * plane_ap + cv[2] * plane_dv + plane_offset) / -cv[1]
        else:  # cam_plane == 2
            ap_rng = np.arange(ap_n, 0, -spacing, dtype=np.float64)
            ml_rng = np.arange(1, ml_n + 1, spacing, dtype=np.float64)
            plane_ap, plane_ml = np.meshgrid(ap_rng, ml_rng)
            plane_dv = (cv[0] * plane_ap + cv[1] * plane_ml + plane_offset) / -cv[2]

        ap_idx = np.round(plane_ap).astype(np.int64)
        dv_idx = np.round(plane_dv).astype(np.int64)
        ml_idx = np.round(plane_ml).astype(np.int64)

        # In-bounds mask (strict upper bound, matching MATLAB ``< size``).
        use = (
            (ap_idx > 0) & (ap_idx < ap_n)
            & (dv_idx > 0) & (dv_idx < dv_n)
            & (ml_idx > 0) & (ml_idx < ml_n)
        )

        tv_slice = np.full(use.shape, np.nan, dtype=np.float64)
        av_slice = np.full(use.shape, np.nan, dtype=np.float64)

        ap_f = ap_idx[use] - 1
        dv_f = dv_idx[use] - 1
        ml_f = ml_idx[use] - 1
        av_vals = np.asarray(av[ap_f, dv_f, ml_f])  # type: ignore[index]
        is_brain = av_vals > 0

        brain_ap = ap_f[is_brain]
        brain_dv = dv_f[is_brain]
        brain_ml = ml_f[is_brain]
        tv_vals = np.asarray(tv[brain_ap, brain_dv, brain_ml])  # type: ignore[index]

        flat_idx = np.flatnonzero(use)
        brain_flat = flat_idx[is_brain]
        tv_slice.flat[brain_flat] = tv_vals
        av_slice.flat[brain_flat] = av_vals[is_brain]

        return {
            "tv_slices": tv_slice,
            "av_slices": av_slice,
            "plane_ap": plane_ap,
            "plane_ml": plane_ml,
            "plane_dv": plane_dv,
            "position_mm": plane_offset / 100.0,  # CCF = 10um voxels
        }


def coronal_camera_vector(tilt_lr_deg: float = 0.0, tilt_si_deg: float = 0.0) -> np.ndarray:
    """Camera/normal vector for a (possibly tilted) coronal plane.

    A pure coronal slice has normal ``[1, 0, 0]`` (the AP axis). ``tilt_lr_deg``
    tilts the plane about the DV axis (left/right yaw), ``tilt_si_deg`` about the
    ML axis (superior/inferior pitch). Returns a unit vector in ``[AP, DV, ML]``.
    """
    yaw = np.deg2rad(tilt_lr_deg)
    pitch = np.deg2rad(tilt_si_deg)
    # Start along AP, rotate towards ML (yaw) and DV (pitch).
    v = np.array([
        np.cos(yaw) * np.cos(pitch),
        np.sin(pitch),
        np.sin(yaw),
    ], dtype=np.float64)
    return v / np.linalg.norm(v)


def coronal_slice_point(ap: float, atlas: AllenCCFAtlas) -> np.ndarray:
    """Slice point at AP position ``ap`` (1-based voxels), centred in DV/ML."""
    ap_n, dv_n, ml_n = atlas.shape
    return np.array([float(ap), dv_n / 2.0, ml_n / 2.0], dtype=np.float64)
