"""Regression tests for the histology pipeline.

Pure-function tests run anywhere. Atlas-dependent tests are skipped if the Allen
CCF files are missing; IBL-dependent tests are skipped if iblatlas is absent.
Run with the neuropygui interpreter::

    python -m pytest tests/test_histology.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from neuropyguin.histology import alignment, io_formats, tracing
from neuropyguin.histology import atlas as hatlas


REF = Path(r"B:\NPX\processedData\VTA_NPX\29237\2\histology")
ATLAS_OK = hatlas.atlas_files_present()


# --------------------------------------------------------------- alignment
def test_fit_affine_recovers_known_transform():
    atlas_pts = np.array([[0, 0], [100, 0], [0, 50], [40, 80]], dtype=float)
    true_T = np.array([[1.2, 0.1, 0.0], [-0.05, 0.9, 0.0], [5.0, -3.0, 1.0]])
    hist_pts = (np.column_stack([atlas_pts, np.ones(len(atlas_pts))]) @ true_T)[:, :2]
    T = alignment.fit_affine_from_points(atlas_pts, hist_pts)
    assert np.allclose(T, true_T, atol=1e-6)


def test_tform_cv2_roundtrip():
    T = np.array([[1.1, 0.2, 0.0], [-0.1, 0.95, 0.0], [7.0, -4.0, 1.0]])
    back = alignment.cv2_to_matlab_T(alignment.matlab_T_to_cv2(T))
    assert np.allclose(T, back)


def test_warp_atlas_identity():
    img = np.arange(64, dtype=float).reshape(8, 8)
    out = alignment.warp_atlas(img, np.eye(3), (8, 8), nearest=True)
    assert np.allclose(out, img)


# ------------------------------------------------------------------- io
def test_probe_ccf_roundtrip(tmp_path):
    pts = np.array([[540.0, 400.0, 570.0], [560.0, 420.0, 575.0]])
    probes = [{"points": pts, "trajectory_areas": pd.DataFrame(), "trajectory_coords": np.zeros((2, 3))}]
    fn = tmp_path / "probe_ccf.mat"
    io_formats.save_probe_ccf(fn, probes)
    # process_histology.py access pattern: mat['probe_ccf'][p][0][0] == points
    import scipy.io as sio
    mat = sio.loadmat(str(fn))
    assert np.allclose(mat["probe_ccf"][0][0][0], pts)
    assert np.allclose(io_formats.load_probe_ccf_points(fn)[0], pts)


def test_histology_ccf_roundtrip(tmp_path):
    sl = {k: np.random.rand(20, 30) for k in
          ("tv_slices", "av_slices", "plane_ap", "plane_ml", "plane_dv")}
    for n in (1, 3):
        fn = tmp_path / f"hccf_{n}.mat"
        io_formats.save_histology_ccf(fn, [sl] * n)
        out = io_formats.load_histology_ccf(fn)
        assert len(out) == n
        assert np.allclose(out[0]["plane_ap"], sl["plane_ap"])


def test_tform_io_roundtrip(tmp_path):
    tforms = [np.eye(3), np.array([[1.0, 0, 0], [0, 1, 0], [5, 7, 1]])]
    fn = tmp_path / "t.mat"
    io_formats.save_tforms(fn, tforms)
    out = io_formats.load_tforms(fn)
    assert np.allclose(out[1][2, :2], [5, 7])


def test_probe_ccf_csv_schema(tmp_path):
    ta = pd.DataFrame({"acronym": ["VTA"], "name": ["v"], "id": [1],
                       "color_hex_triplet": ["ff0000"],
                       "depth_start_um": [0.0], "depth_end_um": [100.0]})
    probes = [{"points": np.zeros((2, 3)), "trajectory_areas": ta}]
    res = io_formats.export_probe_ccf_csv(tmp_path, probes)
    df = pd.read_csv(res["areas"])
    assert list(df.columns) == ["probe", "acronym", "name", "region_id",
                                "color_hex_triplet", "depth_start_um", "depth_end_um"]
    assert df.iloc[0]["acronym"] == "VTA"


# ---------------------------------------------------------------- atlas
@pytest.mark.skipif(not ATLAS_OK, reason="Allen CCF atlas files not present")
def test_coronal_slice_is_constant_ap():
    at = hatlas.AllenCCFAtlas()
    sl = at.grab_atlas_slice(hatlas.coronal_slice_point(700, at),
                             hatlas.coronal_camera_vector(), spacing=4)
    aps = np.unique(np.round(sl["plane_ap"]))
    assert aps.size == 1 and aps[0] == 700


@pytest.mark.skipif(not ATLAS_OK, reason="Allen CCF atlas files not present")
def test_region_lookup():
    at = hatlas.AllenCCFAtlas()
    assert at.acronym(2) == "grey"


@pytest.mark.skipif(not (ATLAS_OK and REF.exists()), reason="atlas or reference data missing")
def test_trajectory_areas_from_reference_points():
    at = hatlas.AllenCCFAtlas()
    pts = io_formats.load_probe_ccf_points(REF / "probe_ccf.mat")[1]  # shank 2
    df, coords = tracing.trajectory_areas_from_points(np.asarray(pts, float), at)
    assert len(df) > 3
    assert {"acronym", "depth_start_um", "depth_end_um"} <= set(df.columns)
    # First in-brain region starts at depth 0.
    assert float(df.iloc[0]["depth_start_um"]) == 0.0


# --------------------------------------------------------------- IBL bridge
@pytest.mark.skipif(not REF.exists(), reason="reference data missing")
def test_channel_locations_match_reference(tmp_path):
    pytest.importorskip("iblatlas")
    pytest.importorskip("ibllib")
    import shutil
    from neuropyguin.histology import ibl_bridge
    for f in ["channels.localCoordinates.npy", "xyz_picks_shank1.json",
              "xyz_picks_shank2.json", "xyz_picks_shank3.json", "xyz_picks_shank4.json",
              "prev_alignments_shank1.json", "prev_alignments_shank2.json",
              "prev_alignments_shank3.json"]:
        if (REF / f).exists():
            shutil.copy(REF / f, tmp_path / f)
    ibl_bridge.compute_channel_locations(tmp_path, alignment="latest")
    gen = json.load(open(tmp_path / "channel_locations_all_shanks.json"))
    ref = json.load(open(REF / "channel_locations_all_shanks.json"))
    gk, rk = set(gen) - {"origin"}, set(ref) - {"origin"}
    assert gk == rk
    assert gen["origin"] == ref["origin"]
    mismatches = sum(1 for k in gk if gen[k]["brain_region_id"] != ref[k]["brain_region_id"])
    assert mismatches == 0
