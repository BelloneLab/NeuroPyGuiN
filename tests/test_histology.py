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

from neuropyguin.histology import acceleration, alignment, auto_align, io_formats, matching, tracing, slice_prep
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


def test_shape_auto_align_centers_different_sized_slice():
    yy, xx = np.ogrid[:80, :100]
    atlas_mask = ((yy - 40) / 25) ** 2 + ((xx - 50) / 34) ** 2 <= 1
    atlas_tv = np.full((80, 100), np.nan, dtype=float)
    atlas_tv[atlas_mask] = 1.0

    hist = np.zeros((170, 230, 3), dtype=np.uint8)
    yyh, xxh = np.ogrid[:170, :230]
    hist_mask = ((yyh - 96) / 50) ** 2 + ((xxh - 128) / 68) ** 2 <= 1
    hist[hist_mask] = [8, 35, 190]

    T = alignment.auto_align_shape(hist, atlas_tv)
    warped = alignment.warp_atlas(np.isfinite(atlas_tv).astype(float), T, hist.shape[:2], nearest=True, use_cv2=False) > 0.5
    dice = 2 * np.logical_and(warped, hist_mask).sum() / (warped.sum() + hist_mask.sum())

    assert dice > 0.82
    assert np.allclose(T[2, :2], [28.0, 16.0], atol=18.0)


def test_shape_auto_align_tolerates_split_and_missing_histology():
    yy, xx = np.ogrid[:90, :120]
    left = ((yy - 45) / 28) ** 2 + ((xx - 42) / 32) ** 2 <= 1
    right = ((yy - 45) / 28) ** 2 + ((xx - 78) / 32) ** 2 <= 1
    atlas_mask = left | right
    atlas_tv = np.full((90, 120), np.nan, dtype=float)
    atlas_tv[atlas_mask] = 1.0

    hist = np.zeros((180, 240, 3), dtype=np.uint8)
    yyh, xxh = np.ogrid[:180, :240]
    left_h = ((yyh - 92) / 56) ** 2 + ((xxh - 84) / 64) ** 2 <= 1
    right_h = ((yyh - 92) / 56) ** 2 + ((xxh - 156) / 64) ** 2 <= 1
    gap = (xxh > 112) & (xxh < 128)
    missing_left_corner = (xxh < 70) & (yyh < 72)
    hist_mask = (left_h | right_h) & ~gap & ~missing_left_corner
    hist[hist_mask] = [8, 35, 190]

    T = alignment.auto_align_shape(hist, atlas_tv)
    warped = alignment.warp_atlas(np.isfinite(atlas_tv).astype(float), T, hist.shape[:2], nearest=True, use_cv2=False) > 0.5
    observed_fit = np.logical_and(warped, hist_mask).sum() / hist_mask.sum()

    assert observed_fit > 0.82
    assert 105 <= T[2, 0] + 60 * T[0, 0] <= 135


def test_probe_alignment_peak_metrics_rejects_broad_or_edge_peaks():
    grid = np.arange(-600, 601, 20, dtype=float)
    sharp = np.exp(-0.5 * ((grid - 420.0) / 45.0) ** 2)
    sharp_metrics = auto_align._peak_metrics(grid, sharp, [400.0, 420.0, 420.0])

    broad = np.where(np.abs(grid - 120.0) <= 340.0, 0.4, 0.2)
    broad_metrics = auto_align._peak_metrics(grid, broad, [100.0, 120.0, 140.0])
    edge = np.exp(-0.5 * ((grid - grid[0]) / 45.0) ** 2)
    edge_metrics = auto_align._peak_metrics(grid, edge, [-600.0, -600.0, -580.0])

    assert sharp_metrics["good"]
    assert sharp_metrics["confidence"] > broad_metrics["confidence"]
    assert not broad_metrics["good"]
    assert broad_metrics["plateau_width_um"] > 220
    assert not edge_metrics["good"]
    assert edge_metrics["edge_limited"]


def test_probe_alignment_consensus_chooses_supported_peak_not_average():
    grid = np.arange(-600, 601, 20, dtype=float)
    strong = np.exp(-0.5 * ((grid - 480.0) / 45.0) ** 2)
    weak = 0.45 * np.exp(-0.5 * ((grid - 100.0) / 90.0) ** 2)
    shanks = [
        {"shank": 2, "good": True, "confidence": 0.24, "offset_um": 480.0, "score_curve": strong.tolist()},
        {"shank": 3, "good": True, "confidence": 0.04, "offset_um": 100.0, "score_curve": weak.tolist()},
    ]

    shared, _metrics, sources = auto_align._consensus_offset(shanks, grid)

    assert shared in {460.0, 480.0, 500.0}
    assert abs(shared - 302.0) > 120.0
    assert sources == [2, 3]


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


def test_load_image_reports_missing_imagecodecs(monkeypatch, tmp_path):
    class BrokenTiffFile:
        @staticmethod
        def imread(_path):
            raise ValueError("<COMPRESSION.LZW: 5> requires the 'imagecodecs' package")

    def broken_pillow(_path):
        raise OSError("cannot decode")

    monkeypatch.setattr(slice_prep, "_HAS_TIFFFILE", True)
    monkeypatch.setattr(slice_prep, "tifffile", BrokenTiffFile)
    monkeypatch.setattr(slice_prep, "_read_tiff_with_pillow", broken_pillow)

    path = tmp_path / "lzw_slice.tif"
    path.write_bytes(b"")
    with pytest.raises(RuntimeError) as excinfo:
        slice_prep.load_image(path)

    msg = str(excinfo.value)
    assert "lzw_slice.tif" in msg
    assert "imagecodecs" in msg
    assert "python -m pip install imagecodecs" in msg


def test_png_raw_images_are_discovered_and_loaded(tmp_path):
    from PIL import Image

    raw = tmp_path / "raw"
    raw.mkdir()
    png = raw / "slide_2.png"
    tif = raw / "slide_10.tif"
    Image.fromarray(np.full((4, 5, 3), 7, dtype=np.uint8)).save(png)
    Image.fromarray(np.full((4, 5, 3), 11, dtype=np.uint8)).save(tif)

    paths = slice_prep.list_raw_images(raw)

    assert [p.name for p in paths] == ["slide_2.png", "slide_10.tif"]
    loaded = slice_prep.load_image(png)
    assert loaded.shape == (4, 5, 3)
    assert loaded.dtype == np.uint8
    assert int(loaded[0, 0, 0]) == 7


def test_saved_slice_discovery_accepts_png_but_ignores_other_assets(tmp_path):
    from PIL import Image

    Image.fromarray(np.zeros((3, 3), dtype=np.uint8)).save(tmp_path / "slice_1.png")
    Image.fromarray(np.zeros((3, 3), dtype=np.uint8)).save(tmp_path / "overview.png")
    Image.fromarray(np.zeros((3, 3), dtype=np.uint8)).save(tmp_path / "slice_2.tif")

    assert [p.name for p in slice_prep.list_saved_slices(tmp_path)] == ["slice_1.png", "slice_2.tif"]


def test_png_alpha_channel_is_converted_to_rgb(tmp_path):
    from PIL import Image

    path = tmp_path / "rgba.png"
    Image.fromarray(np.zeros((3, 4, 4), dtype=np.uint8), mode="RGBA").save(path)

    assert slice_prep.load_image(path).shape == (3, 4, 3)


def test_acceleration_worker_count_is_bounded():
    workers = acceleration.auto_worker_count(task_count=3, cap=8)

    assert 1 <= workers <= 3
    assert "CPU workers" in acceleration.hardware_summary()


def test_pixel_size_um_reads_zen_sidecar_metadata(tmp_path):
    path = tmp_path / "slice.png"
    path.write_bytes(b"not used")
    sidecar = tmp_path / "slice.png_metadata.xml"
    sidecar.write_text(
        """
<Metadata>
  <Scaling>
    <Items>
      <Distance Id="X"><Value>1.20E-05</Value></Distance>
      <Distance Id="Y"><Value>1.40E-05</Value></Distance>
      <Distance Id="Z"><Value>2.00E-06</Value></Distance>
    </Items>
  </Scaling>
</Metadata>
""".strip(),
        encoding="utf-8",
    )

    assert slice_prep.pixel_size_um(path) == pytest.approx(13.0)


def test_histology_shape_mask_fills_tissue_outline():
    img = np.zeros((80, 100, 3), dtype=np.uint8)
    yy, xx = np.ogrid[:80, :100]
    left = ((yy - 42) / 24) ** 2 + ((xx - 42) / 25) ** 2 <= 1
    right = ((yy - 42) / 24) ** 2 + ((xx - 58) / 25) ** 2 <= 1
    outline = left | right
    img[outline] = [5, 30, 160]
    img[40:45, 50:54] = [160, 5, 5]

    mask = matching.histology_shape_mask(img)

    assert mask.sum() > 2000
    assert mask[42, 50]


def test_atlas_shape_mask_ignores_finite_zero_background():
    tv = np.zeros((80, 100), dtype=float)
    yy, xx = np.ogrid[:80, :100]
    tissue = ((yy - 42) / 25) ** 2 + ((xx - 52) / 32) ** 2 <= 1
    tv[tissue] = 1.0

    mask = matching.atlas_shape_mask(tv)

    assert mask.sum() > 1800
    assert mask[42, 52]
    assert not mask[0, 0]


def test_automatch_coronal_ap_prefers_matching_shape(monkeypatch):
    class FakeAtlas:
        shape = (120, 80, 100)

        def grab_atlas_slice(self, slice_point, _camera_vector, spacing=8):
            ap = int(round(float(slice_point[0])))
            yy, xx = np.ogrid[:80:spacing, :100:spacing]
            width = 16 if ap < 60 else 32
            mask = ((yy - 40) / 22) ** 2 + ((xx - 50) / width) ** 2 <= 1
            tv = np.full(mask.shape, np.nan, dtype=float)
            tv[mask] = 1.0
            return {"tv_slices": tv}

    hist = np.zeros((80, 100, 3), dtype=np.uint8)
    yy, xx = np.ogrid[:80, :100]
    wide = ((yy - 40) / 22) ** 2 + ((xx - 50) / 32) ** 2 <= 1
    hist[wide] = [10, 30, 180]

    result = matching.automatch_coronal_ap(
        hist,
        FakeAtlas(),
        center_ap=70,
        search_radius=45,
        ap_step=20,
        refine_radius=10,
        refine_step=5,
        spacing=4,
        canvas_size=64,
    )

    assert result["ap"] >= 60
    assert result["score"] > 0.5
    assert result["engine"]["workers"] >= 1
    assert "numba" in result["engine"]


def test_automatch_ignores_blank_finite_atlas_planes():
    class FakeAtlas:
        shape = (120, 80, 100)

        def grab_atlas_slice(self, slice_point, _camera_vector, spacing=8):
            ap = int(round(float(slice_point[0])))
            yy, xx = np.ogrid[:80:spacing, :100:spacing]
            tv = np.zeros(yy.shape[:1] + xx.shape[1:], dtype=float)
            if ap >= 60:
                mask = ((yy - 40) / 22) ** 2 + ((xx - 50) / 32) ** 2 <= 1
                tv[mask] = 1.0
            return {"tv_slices": tv}

    hist = np.zeros((80, 100, 3), dtype=np.uint8)
    yy, xx = np.ogrid[:80, :100]
    tissue = ((yy - 40) / 22) ** 2 + ((xx - 50) / 32) ** 2 <= 1
    hist[tissue] = [10, 30, 180]

    result = matching.automatch_coronal_ap(
        hist,
        FakeAtlas(),
        ap_step=20,
        refine_radius=10,
        refine_step=5,
        spacing=4,
        canvas_size=64,
    )

    assert result["ap"] >= 60
    assert all(int(item["ap"]) >= 50 for item in result["top"][:3])


def test_automatch_center_prior_breaks_ambiguous_shape_tie():
    class FakeAtlas:
        shape = (130, 90, 120)

        def grab_atlas_slice(self, slice_point, _camera_vector, spacing=8):
            ap = int(round(float(slice_point[0])))
            yy, xx = np.ogrid[:90:spacing, :120:spacing]
            width = 26 if ap < 70 else 28
            mask = ((yy - 45) / 24) ** 2 + ((xx - 60) / width) ** 2 <= 1
            tv = np.zeros(mask.shape, dtype=float)
            tv[mask] = 1.0
            return {"tv_slices": tv}

    hist = np.zeros((90, 120, 3), dtype=np.uint8)
    yy, xx = np.ogrid[:90, :120]
    tissue = ((yy - 45) / 24) ** 2 + ((xx - 60) / 27) ** 2 <= 1
    hist[tissue] = [10, 30, 180]

    result = matching.automatch_coronal_ap(
        hist,
        FakeAtlas(),
        center_ap=55,
        search_radius=30,
        ap_step=20,
        refine_radius=10,
        refine_step=5,
        spacing=4,
        canvas_size=64,
    )

    assert result["ap"] < 70


def test_automatch_absolute_size_uses_atlas_spacing():
    class FakeAtlas:
        shape = (120, 80, 120)

        def grab_atlas_slice(self, slice_point, _camera_vector, spacing=8):
            ap = int(round(float(slice_point[0])))
            yy, xx = np.ogrid[:80:spacing, :120:spacing]
            width = 24 if ap < 60 else 38
            mask = ((yy - 40) / 22) ** 2 + ((xx - 60) / width) ** 2 <= 1
            tv = np.zeros(mask.shape, dtype=float)
            tv[mask] = 1.0
            return {"tv_slices": tv}

    hist = np.zeros((80, 120, 3), dtype=np.uint8)
    yy, xx = np.ogrid[:80, :120]
    tissue = ((yy - 40) / 22) ** 2 + ((xx - 60) / 24) ** 2 <= 1
    hist[tissue] = [10, 30, 180]

    result = matching.automatch_coronal_ap(
        hist,
        FakeAtlas(),
        center_ap=80,
        search_radius=80,
        atlas_to_histology_scale=1.0,
        ap_step=20,
        refine_radius=10,
        refine_step=5,
        spacing=4,
        canvas_size=64,
    )

    assert result["ap"] < 60


def test_automatch_tolerates_split_and_missing_tissue():
    class FakeAtlas:
        shape = (120, 90, 120)

        def grab_atlas_slice(self, slice_point, _camera_vector, spacing=8):
            ap = int(round(float(slice_point[0])))
            yy, xx = np.ogrid[:90:spacing, :120:spacing]
            if ap < 60:
                mask = ((yy - 45) / 18) ** 2 + ((xx - 60) / 26) ** 2 <= 1
            else:
                left = ((yy - 45) / 25) ** 2 + ((xx - 42) / 28) ** 2 <= 1
                right = ((yy - 45) / 25) ** 2 + ((xx - 78) / 28) ** 2 <= 1
                mask = left | right
            tv = np.full(mask.shape, np.nan, dtype=float)
            tv[mask] = 1.0
            return {"tv_slices": tv}

    hist = np.zeros((90, 120, 3), dtype=np.uint8)
    yy, xx = np.ogrid[:90, :120]
    left = ((yy - 45) / 25) ** 2 + ((xx - 42) / 28) ** 2 <= 1
    right = ((yy - 45) / 25) ** 2 + ((xx - 78) / 28) ** 2 <= 1
    gap = (xx > 54) & (xx < 66)
    missing_left = (xx < 32) & (yy < 38)
    partial = (left | right) & ~gap & ~missing_left
    hist[partial] = [6, 34, 190]
    hist[np.broadcast_to((xx > 54) & (xx < 66), hist.shape[:2])] = [30, 0, 0]

    mask = matching.histology_shape_mask(hist)
    labels, n_labels = matching.ndi.label(mask)
    result = matching.automatch_coronal_ap(
        hist,
        FakeAtlas(),
        ap_step=20,
        refine_radius=10,
        refine_step=5,
        spacing=4,
        canvas_size=64,
    )

    assert n_labels >= 2
    assert result["ap"] >= 60
    assert result["score"] > 0.45


def test_match_auto_toggle_assigns_and_auto_adjusts(monkeypatch, tmp_path):
    from PySide6 import QtCore, QtWidgets
    from neuropyguin.tabs.histology_tab import HistologyTab

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    tab = HistologyTab(QtCore.QThreadPool.globalInstance())
    tab.folder = tmp_path
    tab.slice_images = [np.ones((16, 20, 3), dtype=np.uint8) * 120]
    tab.slice_specs = [None]
    tab.ck_match_auto_adjust.setChecked(True)
    seen = []

    class FakeAtlas:
        shape = (120, 80, 100)

    monkeypatch.setattr(tab, "_ensure_atlas", lambda: FakeAtlas())
    monkeypatch.setattr(tab, "_match_update_atlas", lambda: tab._match_update_slider_labels())
    monkeypatch.setattr(
        matching,
        "automatch_coronal_ap",
        lambda _image, _atlas, **_kwargs: {"ap": 77, "score": 0.8, "confidence": 0.2, "top": []},
    )
    monkeypatch.setattr(tab, "_auto_adjust_current_match_slice", lambda _atlas, idx: seen.append(idx))
    monkeypatch.setattr(tab, "_run_bg", lambda fn, done, *args, **kwargs: done(fn()))

    tab._match_auto()

    assert tab.sl_ap.value() == 77
    assert tab.slice_specs[0] is not None
    assert seen == [0]
    assert (tmp_path / tab._MATCH_SPECS_FN).exists()


def test_match_assign_refreshes_downstream_atlas_cache(monkeypatch, tmp_path):
    from PySide6 import QtCore, QtWidgets
    from neuropyguin.tabs import histology_tab
    from neuropyguin.tabs.histology_tab import HistologyTab

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    tab = HistologyTab(QtCore.QThreadPool.globalInstance())
    tab.folder = tmp_path
    tab.slice_images = [np.zeros((40, 50, 3), dtype=np.uint8)]
    tab.histology_ccf = [{"tv_slices": np.ones((3, 4)), "av_slices": np.ones((3, 4))}]
    tab.tforms = [np.array([[1.0, 0, 0], [0, 1, 0], [12.0, 18.0, 1.0]])]
    tab._align_hist_pts[0] = [(1.0, 2.0)]
    tab._align_atlas_pts[0] = [(3.0, 4.0)]
    (tmp_path / "probe_ccf.mat").write_bytes(b"stale")
    (tmp_path / "channel_locations_all_shanks.json").write_text("{}")

    class FakeAtlas:
        shape = (120, 80, 100)

    rebuilt = {"tv_slices": np.ones((5, 6)) * 7, "av_slices": np.ones((5, 6)) * 9}

    monkeypatch.setattr(tab, "_ensure_atlas", lambda: FakeAtlas())
    monkeypatch.setattr(tab, "_match_update_atlas", lambda: None)
    monkeypatch.setattr(histology_tab.matching, "build_histology_ccf", lambda *_args, **_kwargs: [rebuilt])

    assert tab._match_assign_current_plane()

    assert np.array_equal(tab.histology_ccf[0]["tv_slices"], rebuilt["tv_slices"])
    assert np.allclose(tab.tforms[0], np.eye(3))
    assert 0 not in tab._align_hist_pts
    assert 0 not in tab._align_atlas_pts
    assert not (tmp_path / "probe_ccf.mat").exists()
    assert not (tmp_path / "channel_locations_all_shanks.json").exists()


def test_align_auto_uses_existing_landmark_pairs(tmp_path):
    from PySide6 import QtCore, QtWidgets
    from neuropyguin.tabs.histology_tab import HistologyTab

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    tab = HistologyTab(QtCore.QThreadPool.globalInstance())
    tab.folder = tmp_path
    tab.slice_images = [np.zeros((40, 50, 3), dtype=np.uint8)]
    tab.histology_ccf = [{"tv_slices": np.ones((20, 25)), "av_slices": np.ones((20, 25))}]
    tab._cur_align_slice = 0
    atlas_pts = [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0), (8.0, 8.0)]
    hist_pts = [(5.0, 6.0), (15.0, 6.0), (5.0, 16.0), (13.0, 14.0)]
    tab._align_atlas_pts[0] = atlas_pts
    tab._align_hist_pts[0] = hist_pts

    tab._align_auto()

    assert np.allclose(tab.tforms[0][2, :2], [5.0, 6.0])
    assert (tmp_path / "atlas2histology_tform.mat").exists()


def test_align_show_rebuilds_metadata_only_histology_ccf(monkeypatch, tmp_path):
    from PySide6 import QtCore, QtWidgets
    from neuropyguin.tabs import histology_tab
    from neuropyguin.tabs.histology_tab import HistologyTab

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    tab = HistologyTab(QtCore.QThreadPool.globalInstance())
    tab.folder = tmp_path
    tab.slice_images = [np.zeros((40, 50, 3), dtype=np.uint8)]
    tab.histology_ccf = [{"plane_ap": np.ones((2, 2))}]
    tab.slice_specs = [{"slice_point": np.array([10.0, 20.0, 30.0]), "camera_vector": np.array([1.0, 0.0, 0.0])}]
    tab._cur_align_slice = 0

    class FakeAtlas:
        shape = (120, 80, 100)

    rebuilt = {
        "tv_slices": np.ones((12, 14), dtype=float),
        "av_slices": np.ones((12, 14), dtype=float) * 2,
        "plane_ap": np.ones((12, 14), dtype=float),
        "plane_ml": np.ones((12, 14), dtype=float),
        "plane_dv": np.ones((12, 14), dtype=float),
    }

    monkeypatch.setattr(tab, "_ensure_atlas", lambda: FakeAtlas())
    monkeypatch.setattr(histology_tab.matching, "build_histology_ccf", lambda *_args, **_kwargs: [rebuilt])

    tab._align_show()

    assert "tv_slices" in tab.histology_ccf[0]
    assert np.array_equal(tab.histology_ccf[0]["tv_slices"], rebuilt["tv_slices"])


def test_align_auto_low_confidence_does_not_replace_transform(monkeypatch, tmp_path):
    from PySide6 import QtCore, QtWidgets
    from neuropyguin.tabs import histology_tab
    from neuropyguin.tabs.histology_tab import HistologyTab

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    tab = HistologyTab(QtCore.QThreadPool.globalInstance())
    tab.folder = tmp_path
    tab.slice_images = [np.zeros((40, 50, 3), dtype=np.uint8)]
    tab.histology_ccf = [{"tv_slices": np.ones((20, 25)), "av_slices": np.ones((20, 25))}]
    tab._cur_align_slice = 0
    original = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [7.0, 9.0, 1.0]])
    tab.tforms = [original.copy()]

    monkeypatch.setattr(
        histology_tab.alignment,
        "auto_align_isolated",
        lambda _hist, _atlas: (np.eye(3), "auto-align low confidence (shape score 0.50); transform not changed."),
    )
    monkeypatch.setattr(tab, "_run_bg", lambda fn, done, *args, **kwargs: done(fn()))

    tab._align_auto()

    assert np.allclose(tab.tforms[0], original)


def test_image_canvas_can_preserve_zoom_between_images():
    from PySide6 import QtWidgets
    from neuropyguin.tabs.histology_tab import ImageCanvas

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    canvas = ImageCanvas()
    canvas.set_image(np.zeros((40, 60), dtype=np.uint8))
    canvas.view.setRange(xRange=(10, 25), yRange=(12, 30), padding=0)
    before = canvas.view.viewRange()

    canvas.set_image(np.ones((40, 60), dtype=np.uint8), preserve_view=True)
    after = canvas.view.viewRange()

    assert np.allclose(before[0], after[0])
    assert np.allclose(before[1], after[1])


def test_histology_slice_views_all_have_contrast_controls(tmp_path):
    from PySide6 import QtCore, QtWidgets
    from neuropyguin.tabs.histology_tab import HistologyTab

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    tab = HistologyTab(QtCore.QThreadPool.globalInstance())
    tab.folder = tmp_path
    tab.slice_images = [np.arange(16 * 20 * 3, dtype=np.uint16).reshape(16, 20, 3)]

    expected = {"preproc", "match", "align", "trace"}
    assert expected <= set(tab._histogram_widgets)
    assert expected <= set(tab._histogram_canvases)

    for stage in expected:
        tab._show_histology_slice(stage, tab._histogram_canvases[stage], 0, tab.slice_images[0])
        lo, hi = tab._histogram_widgets[stage].item.getLevels()
        assert np.isfinite([lo, hi]).all()
        assert hi > lo


def test_trace_default_line_reuses_nearby_probe_geometry(tmp_path):
    from PySide6 import QtCore, QtWidgets
    from neuropyguin.tabs.histology_tab import HistologyTab

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    tab = HistologyTab(QtCore.QThreadPool.globalInstance())
    tab.folder = tmp_path
    tab.slice_images = [np.zeros((120, 160, 3), dtype=np.uint8)]
    tab._cur_trace_slice = 0
    tab._active_probe = 2
    reference = np.array([[80.0, 45.0], [86.0, 83.0]])
    tab.probe_points[(0, 1)] = reference

    pts = tab._trace_default_line()

    assert np.isclose(np.linalg.norm(pts[1] - pts[0]), np.linalg.norm(reference[1] - reference[0]))
    assert not np.allclose(pts.mean(axis=0), reference.mean(axis=0))
    assert np.all(pts[:, 0] >= 0)
    assert np.all(pts[:, 0] <= 159)
    assert np.all(pts[:, 1] >= 0)
    assert np.all(pts[:, 1] <= 119)


def test_trace_contrast_survives_shank_switch(tmp_path):
    from PySide6 import QtCore, QtWidgets
    from neuropyguin.tabs.histology_tab import HistologyTab

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    tab = HistologyTab(QtCore.QThreadPool.globalInstance())
    tab.folder = tmp_path
    tab.slice_images = [np.zeros((40, 60, 3), dtype=np.uint8)]
    tab._cur_trace_slice = 0
    tab._active_probe = 1

    tab._trace_show()
    tab.canvas_trace.image_item().setLevels((12.0, 180.0))
    tab._trace_set_probe(2)

    assert np.allclose(tab.canvas_trace.image_item().levels, (12.0, 180.0))


def test_trace_show_keeps_all_drawn_shank_lines_visible(tmp_path):
    from PySide6 import QtCore, QtWidgets
    from neuropyguin.tabs.histology_tab import HistologyTab

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    tab = HistologyTab(QtCore.QThreadPool.globalInstance())
    tab.folder = tmp_path
    tab.slice_images = [np.zeros((80, 100, 3), dtype=np.uint8)]
    tab._cur_trace_slice = 0
    tab._active_probe = 2
    tab.probe_points[(0, 1)] = np.array([[20.0, 20.0], [21.0, 60.0]])
    tab.probe_points[(0, 2)] = np.array([[40.0, 20.0], [41.0, 60.0]])
    tab.probe_points[(0, 3)] = np.array([[60.0, 20.0], [61.0, 60.0]])

    tab._trace_show()

    assert tab._trace_roi is not None
    assert len(tab.canvas_trace._overlays) >= 6
    assert tab.lbl_trace_line.text() == "shank 2 on slice 1"


def test_trace_build_rebuilds_sparse_ccf_for_later_traced_slice(monkeypatch, tmp_path):
    from PySide6 import QtCore, QtWidgets
    from neuropyguin.tabs import histology_tab
    from neuropyguin.tabs.histology_tab import HistologyTab

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    tab = HistologyTab(QtCore.QThreadPool.globalInstance())
    tab.folder = tmp_path
    tab.slice_images = [np.zeros((20, 20, 3), dtype=np.uint8) for _ in range(3)]
    tab.slice_specs = [None, None, {"slice_point": np.array([1.0, 2.0, 3.0]), "camera_vector": np.array([1.0, 0.0, 0.0])}]
    tab.histology_ccf = []
    tab.tforms = [np.eye(3)]
    tab.probe_points[(2, 1)] = np.array([[5.0, 6.0], [7.0, 8.0]])

    rebuilt = {
        "tv_slices": np.ones((10, 10)),
        "av_slices": np.ones((10, 10)),
        "plane_ap": np.ones((10, 10)),
        "plane_ml": np.ones((10, 10)) * 2,
        "plane_dv": np.ones((10, 10)) * 3,
    }
    calls = {}
    probes = [{
        "points": np.array([[1.0, 2.0, 3.0], [1.0, 4.0, 3.0]]),
        "trajectory_coords": np.array([[1.0, 2.0, 3.0], [1.0, 4.0, 3.0]]),
        "trajectory_areas": pd.DataFrame(),
    }]

    class FakeAtlas:
        pass

    def fake_build_probe(pts0, histology_ccf, tforms, atlas, n_probes):
        calls["pts0"] = pts0
        calls["histology_ccf"] = histology_ccf
        calls["tforms"] = tforms
        calls["n_probes"] = n_probes
        return probes

    tab.atlas = FakeAtlas()
    monkeypatch.setattr(tab, "_ensure_atlas", lambda: tab.atlas)
    monkeypatch.setattr(histology_tab.matching, "build_histology_ccf", lambda *_args, **_kwargs: [rebuilt])
    monkeypatch.setattr(histology_tab.tracing, "build_probe_ccf", fake_build_probe)
    monkeypatch.setattr(histology_tab.io_formats, "save_probe_ccf", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(histology_tab.io_formats, "export_probe_ccf_csv", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(tab, "_draw_trajectory_areas", lambda _probes: None)
    monkeypatch.setattr(tab, "_save_trajectory_3d_gif", lambda _probes: None)
    monkeypatch.setattr(tab, "_run_bg", lambda fn, done, *args, **kwargs: done(fn()))

    tab._trace_build()

    assert set(calls["pts0"]) == {(2, 0)}
    assert calls["n_probes"] == 1
    assert len(calls["histology_ccf"]) == 3
    assert len(calls["tforms"]) == 3
    assert np.array_equal(calls["histology_ccf"][2]["plane_dv"], rebuilt["plane_dv"])


def test_image_selection_limits_preprocess_inputs(tmp_path):
    from PIL import Image
    from PySide6 import QtCore, QtWidgets
    from neuropyguin.tabs.histology_tab import HistologyTab

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    raw = tmp_path / "raw"
    raw.mkdir()
    for i, value in enumerate((20, 80, 140), start=1):
        Image.fromarray(np.full((8, 10, 3), value, dtype=np.uint8)).save(raw / f"img_{i}.png")

    tab = HistologyTab(QtCore.QThreadPool.globalInstance())
    tab.ed_raw.setText(str(raw))
    tab._selection_refresh()
    assert "select" in tab._histogram_widgets
    lo, hi = tab._histogram_widgets["select"].item.getLevels()
    assert np.isfinite([lo, hi]).all()
    assert hi > lo
    assert tab._selection_preview_image is not None
    middle = tab.lst_raw_images.item(1)
    tab._hist_levels_by_stage["select"][0] = (10.0, 200.0)
    tab._selection_show_preview(middle)
    assert np.allclose(tab.canvas_select_raw.image_item().levels, (10.0, 200.0))
    assert not bool(middle.flags() & QtCore.Qt.ItemIsUserCheckable)
    assert bool(middle.data(tab._SELECTION_SELECTED_ROLE))
    assert not middle.icon().isNull()
    pix = middle.icon().pixmap(tab.lst_raw_images.iconSize())
    qimg = pix.toImage()
    green_samples = 0
    for y in range(0, qimg.height(), 4):
        for x in range(0, qimg.width(), 4):
            c = qimg.pixelColor(x, y)
            if c.green() > 120 and c.red() < 80 and c.blue() < 120:
                green_samples += 1
    assert green_samples > 0
    tab._selection_toggle_item(middle)
    assert not bool(middle.data(tab._SELECTION_SELECTED_ROLE))
    assert "ignored" in tab.lbl_select_preview.text()
    monkeypatch_run = lambda fn, done, *args, **kwargs: done(fn())
    tab._run_bg = monkeypatch_run  # type: ignore[method-assign]

    tab._selection_validate()

    assert [p.name for p in tab.validated_raw_paths] == ["img_1.png", "img_3.png"]
    assert len(tab.slice_images) == 2
    assert [p.name for p in tab.selected_raw_paths] == ["img_1.png", "img_3.png"]
    assert int(tab.slice_images[0][0, 0, 0]) == 20
    assert int(tab.slice_images[1][0, 0, 0]) == 140


def test_image_selection_repairs_stale_raw_path_from_session_folder(tmp_path):
    from PIL import Image
    from PySide6 import QtCore, QtWidgets
    from neuropyguin.tabs.histology_tab import HistologyTab

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    hist = tmp_path / "object_healthy_sick" / "histology"
    raw = hist / "raw"
    raw.mkdir(parents=True)
    Image.fromarray(np.full((8, 10, 3), 120, dtype=np.uint8)).save(raw / "raw_1.png")

    tab = HistologyTab(QtCore.QThreadPool.globalInstance())
    tab.folder = hist
    tab.ed_raw.setText(str(tmp_path / "histology" / "raw"))

    tab._selection_refresh()

    assert tab.ed_raw.text() == str(raw)
    assert tab.lst_raw_images.count() == 1
    assert tab.lbl_image_selection.text() == "1 / 1 selected, not validated"


def test_preproc_save_removes_stale_app_slice_tiffs_but_keeps_png(tmp_path):
    from PIL import Image
    from PySide6 import QtCore, QtWidgets
    from neuropyguin.tabs.histology_tab import HistologyTab

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    stale_tif = tmp_path / "slice_2.tif"
    stale_tif.write_bytes(b"stale")
    Image.fromarray(np.zeros((3, 3), dtype=np.uint8)).save(tmp_path / "slice_2.png")
    tab = HistologyTab(QtCore.QThreadPool.globalInstance())
    tab.folder = tmp_path
    tab.slice_images = [np.zeros((5, 6, 3), dtype=np.uint8)]

    tab._preproc_save()

    assert (tmp_path / "slice_1.tif").exists()
    assert not stale_tif.exists()
    assert (tmp_path / "slice_2.png").exists()


def test_prepare_for_ibl_runs_fast_required_pipeline_without_rms(monkeypatch, tmp_path):
    from PySide6 import QtCore, QtWidgets
    from neuropyguin.tabs import histology_tab
    from neuropyguin.tabs.histology_tab import HistologyTab

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    tab = HistologyTab(QtCore.QThreadPool.globalInstance())
    tab.folder = tmp_path
    (tmp_path / "probe_ccf.mat").write_bytes(b"placeholder")
    tab.ed_ks.setText(str(tmp_path / "ks"))
    tab.ed_ephys.setText(str(tmp_path / "ephys"))
    seen = []

    monkeypatch.setattr(histology_tab.ibl_launch, "run_bridge", lambda args, **kwargs: seen.append(args) or (0, ""))
    monkeypatch.setattr(tab, "_run_bg", lambda fn, done, *args, **kwargs: done(fn()))
    monkeypatch.setattr(tab, "_on_channels_done", lambda rc: None)

    tab._prepare_for_ibl()

    assert seen == [[
        "all",
        str(tmp_path),
        "--alignment", "original",
        "--ks", str(tmp_path / "ks"),
        "--ephys", str(tmp_path / "ephys"),
    ]]


def test_optional_rms_button_runs_rms_extraction(monkeypatch, tmp_path):
    from PySide6 import QtCore, QtWidgets
    from neuropyguin.tabs import histology_tab
    from neuropyguin.tabs.histology_tab import HistologyTab

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    tab = HistologyTab(QtCore.QThreadPool.globalInstance())
    tab.folder = tmp_path
    tab.ed_ks.setText(str(tmp_path / "ks"))
    tab.ed_ephys.setText(str(tmp_path / "ephys"))
    seen = []

    monkeypatch.setattr(histology_tab.ibl_launch, "run_bridge", lambda args, **kwargs: seen.append(args) or (0, ""))
    monkeypatch.setattr(tab, "_run_bg", lambda fn, done, *args, **kwargs: done(fn()))

    tab._compute_rms_qc_maps()

    assert seen == [["extract_alf", str(tmp_path / "ks"), str(tmp_path / "ephys"), str(tmp_path), "--rms"]]


def test_fast_extract_alf_exports_minimal_arrays(tmp_path):
    from neuropyguin.histology import ibl_bridge

    ks = tmp_path / "ks"
    out = tmp_path / "histology"
    ephys = tmp_path / "ephys"
    ks.mkdir()
    ephys.mkdir()
    np.save(ks / "channel_positions.npy", np.array([[0, 0], [0, 20], [0, 40], [0, 60]], dtype=np.float32))
    np.save(ks / "channel_map.npy", np.arange(4, dtype=np.int32))
    np.save(ks / "spike_times.npy", np.array([0, 30_000, 60_000, 90_000], dtype=np.int64))
    np.save(ks / "spike_clusters.npy", np.array([0, 1, 1, 3], dtype=np.uint32))
    np.save(ks / "amplitudes.npy", np.array([2, 4, 6, 8], dtype=np.float32))
    np.save(ks / "spike_templates.npy", np.array([0, 1, 1, 2], dtype=np.int32))
    np.save(ks / "templates.npy", np.zeros((3, 5, 4), dtype=np.float32))
    (ks / "params.py").write_text("sample_rate = 30000.0\n", encoding="utf-8")
    (ks / "metrics.csv").write_text(
        "cluster_id,peak_channel,amplitude\n0,2,10\n1,1,20\n3,3,30\n",
        encoding="utf-8",
    )

    ibl_bridge.extract_alf(ks, ephys, out, mode="fast")

    assert np.allclose(np.load(out / "spikes.times.npy"), [0, 1, 2, 3])
    assert np.array_equal(np.load(out / "spikes.clusters.npy"), [0, 1, 1, 3])
    assert np.array_equal(np.load(out / "clusters.channels.npy")[[0, 1, 3]], [2, 1, 3])
    assert np.allclose(np.load(out / "clusters.depths.npy")[[0, 1, 3]], [40, 20, 60])
    assert np.allclose(np.load(out / "spikes.depths.npy"), [40, 20, 20, 60])
    assert np.array_equal(np.load(out / "channels.localCoordinates.npy"), np.load(ks / "channel_positions.npy"))
    assert np.array_equal(np.load(out / "channels.rawInd.npy"), np.arange(4))
    assert np.array_equal(np.load(out / "spikes.amps.npy"), [2, 4, 6, 8])
    assert np.load(out / "clusters.waveforms.npy").shape == (4, 5, 1)
    assert np.load(out / "clusters.peakToTrough.npy").shape == (4,)


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


def test_trajectory_3d_region_segments_follow_depth_table():
    from neuropyguin.tabs.histology_tab import Trajectory3DCanvas

    coords = np.array([[540.0, 100.0, 570.0], [540.0, 300.0, 570.0]])
    ta = pd.DataFrame({
        "color_hex_triplet": ["ff0000", "00ff00"],
        "depth_start_um": [0.0, 1000.0],
        "depth_end_um": [1000.0, 2000.0],
    })
    segments = Trajectory3DCanvas._trajectory_region_segments(
        {"trajectory_areas": ta},
        coords,
    )
    assert len(segments) == 2
    assert np.allclose(segments[0][0], coords[0])
    assert np.allclose(segments[0][1], [540.0, 200.0, 570.0])
    assert np.allclose(segments[1][1], coords[1])
    assert segments[0][2] == (1.0, 0.0, 0.0)


def test_trajectory_3d_tube_mesh_shape():
    from neuropyguin.tabs.histology_tab import Trajectory3DCanvas

    mesh = Trajectory3DCanvas._tube_mesh(
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        radius_mm=0.05,
    )
    assert mesh is not None
    assert mesh[0].shape == (2, 11)
    assert all(part.shape == mesh[0].shape for part in mesh)


def test_trajectory_3d_probe_core_is_red():
    from neuropyguin.tabs.histology_tab import Trajectory3DCanvas

    assert Trajectory3DCanvas._probe_rgb(0) == pytest.approx((0.86, 0.05, 0.04))
    assert Trajectory3DCanvas._probe_rgb(4) == pytest.approx((0.86, 0.05, 0.04))


def test_trace_build_updates_embedded_3d_without_popup(monkeypatch, tmp_path):
    from PySide6 import QtCore, QtWidgets
    from neuropyguin.tabs import histology_tab
    from neuropyguin.tabs.histology_tab import HistologyTab

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    tab = HistologyTab(QtCore.QThreadPool.globalInstance())
    tab.folder = tmp_path
    tab.slice_images = [np.zeros((10, 10, 3), dtype=np.uint8)]
    tab.histology_ccf = [{
        "tv_slices": np.ones((10, 10)),
        "av_slices": np.ones((10, 10)),
        "plane_ap": np.ones((10, 10)),
        "plane_ml": np.ones((10, 10)) * 2,
        "plane_dv": np.ones((10, 10)) * 3,
    }]
    tab.tforms = [np.eye(3)]
    tab.probe_points[(0, 1)] = np.array([[1.0, 2.0], [3.0, 4.0]])

    class FakeAtlas:
        pass

    probes = [{
        "points": np.array([[540.0, 100.0, 570.0], [540.0, 200.0, 570.0]]),
        "trajectory_coords": np.array([[540.0, 100.0, 570.0], [540.0, 200.0, 570.0]]),
        "trajectory_areas": pd.DataFrame(),
    }]
    calls = {"render": 0, "popup": 0, "gif": 0}

    tab.atlas = FakeAtlas()
    monkeypatch.setattr(tab, "_ensure_atlas", lambda: tab.atlas)
    monkeypatch.setattr(histology_tab.tracing, "build_probe_ccf", lambda *_args, **_kwargs: probes)
    monkeypatch.setattr(tab, "_draw_trajectory_areas", lambda _probes: tab.trace_3d.render(tab.atlas, _probes))
    monkeypatch.setattr(tab.trace_3d, "render", lambda _atlas, _probes: calls.__setitem__("render", calls["render"] + 1))
    monkeypatch.setattr(tab, "_show_trajectory_3d_popup", lambda _probes: calls.__setitem__("popup", calls["popup"] + 1))
    monkeypatch.setattr(tab, "_save_trajectory_3d_gif", lambda _probes: calls.__setitem__("gif", calls["gif"] + 1))
    monkeypatch.setattr(tab, "_run_bg", lambda fn, done, *args, **kwargs: done(fn()))

    tab._trace_build()

    assert calls["render"] == 1
    assert calls["popup"] == 0
    assert calls["gif"] == 1


def test_trajectory_3d_shell_draws_visible_contour_lines():
    from neuropyguin.tabs import histology_tab
    from neuropyguin.tabs.histology_tab import Trajectory3DCanvas

    if histology_tab.Figure is None:
        pytest.skip("Matplotlib is unavailable")

    fig = histology_tab.Figure(figsize=(2.0, 2.0), dpi=80)
    ax = fig.add_subplot(111, projection="3d")
    polygon = np.array([
        [540.0, 400.0, 570.0],
        [540.0, 420.0, 590.0],
        [540.0, 440.0, 570.0],
        [540.0, 400.0, 570.0],
    ])

    ok = Trajectory3DCanvas._draw_brain_shell(ax, [polygon], "#d3dbe5", "Light")

    assert ok
    assert len(ax.lines) == 1
    assert ax.lines[0].get_alpha() >= 0.20
    assert ax.lines[0].get_linewidth() >= 0.3


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
def test_fast_xyz_picks_from_probe_ccf_without_ibl_stack(tmp_path):
    from neuropyguin.histology import ibl_bridge

    points = np.array([
        [100.0, 10.0, 600.0],
        [100.0, 20.0, 600.0],
        [100.0, 30.0, 600.0],
    ])
    io_formats.save_probe_ccf(tmp_path / "probe_ccf.mat", [{
        "points": points,
        "trajectory_coords": np.array([[99.0, 9.0, 600.0], [100.0, 30.0, 600.0]]),
    }])

    written = ibl_bridge.compute_xyz_picks(tmp_path / "probe_ccf.mat", tmp_path, mode="fast")
    data = json.loads(written[0].read_text())
    xyz = np.asarray(data["xyz_picks"], dtype=float)

    assert xyz.shape == (2, 3)
    assert np.allclose(xyz[0], [261.0, 4420.0, 252.0])
    assert np.allclose(xyz[1], [261.0, 4400.0, 32.0])


def test_fast_xyz_picks_skips_empty_sparse_probe_slots(tmp_path):
    from neuropyguin.histology import ibl_bridge

    points = np.array([
        [120.0, 10.0, 620.0],
        [120.0, 20.0, 620.0],
        [120.0, 30.0, 620.0],
    ])
    io_formats.save_probe_ccf(tmp_path / "probe_ccf.mat", [
        {},
        {},
        {
            "points": points,
            "trajectory_coords": np.array([[119.0, 9.0, 620.0], [120.0, 30.0, 620.0]]),
        },
    ])
    (tmp_path / "xyz_picks.json").write_text("{}")
    (tmp_path / "xyz_picks_shank1.json").write_text("{}")

    written = ibl_bridge.compute_xyz_picks(tmp_path / "probe_ccf.mat", tmp_path, mode="fast")

    assert [p.name for p in written] == ["xyz_picks_shank3.json"]
    assert not (tmp_path / "xyz_picks_shank1.json").exists()
    assert not (tmp_path / "xyz_picks_shank2.json").exists()
    assert (tmp_path / "xyz_picks.json").exists()
    assert json.loads((tmp_path / "xyz_picks.json").read_text()) == json.loads(written[0].read_text())
    assert ibl_bridge._xyz_pick_files_for_recorded_shank(tmp_path, 0, 1) == [tmp_path / "xyz_picks.json"]
    (tmp_path / "xyz_picks.json").unlink()
    (tmp_path / "xyz_picks_shank1.json").write_text("{}")
    assert ibl_bridge._xyz_pick_files_for_recorded_shank(tmp_path, 0, 1) == [tmp_path / "xyz_picks_shank3.json"]


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
