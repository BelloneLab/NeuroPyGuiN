"""Histology tab: AP_histology-style probe localization, unified and renewed.

Stages (left rail):

    Setup      pick the histology folder + tool paths, see what products exist
    Preprocess load raw images, colourise, segment/extract slices, reorient, save
    Match      choose the Allen CCF plane for each slice            -> histology_ccf
    Align      warp atlas <-> histology (control points / auto)     -> atlas2histology_tform
    Trace      draw probe tracks, sample regions                    -> probe_ccf (+ CSV)
    Channels   ALF extraction + xyz_picks + per-channel region map  -> channel_locations_all_shanks
    IBL refine launch the unmodified IBL ephys-alignment GUI (optional)

All heavy lifting is delegated to :mod:`neuropyguin.histology`. The IBL-dependent
steps run through :mod:`neuropyguin.histology.ibl_launch` (subprocess), so this tab
works whether or not the IBL stack is importable in-process.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from ..side_nav import SideNavStack
from ..workers import FunctionWorker
from ..histology import atlas as hatlas
from ..histology import acceleration, io_formats, matching, tracing, alignment, slice_prep, ibl_launch

try:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
except Exception:  # pragma: no cover
    FigureCanvasAgg = None  # type: ignore[assignment]
    FigureCanvasQTAgg = None  # type: ignore[assignment]
    Figure = None  # type: ignore[assignment]

try:  # crash breadcrumbs: identify the last render step before a native paint crash
    from .._diagnostics import breadcrumb as _crumb
except Exception:  # pragma: no cover
    def _crumb(msg: str) -> None:  # type: ignore
        pass


PROBE_QCOLORS = [
    QtGui.QColor(*[int(c * 255) for c in rgb]) for rgb in tracing.probe_colormap(20)
]


class ImageCanvas(pg.GraphicsLayoutWidget):
    """A simple image view with pixel-coordinate click reporting and overlays."""

    clicked = QtCore.Signal(float, float)  # image x (col), y (row)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.view = self.addViewBox()
        self.view.setAspectLocked(True)
        self.view.invertY(True)
        self.img = pg.ImageItem()
        self.img.setOpts(axisOrder="row-major")
        self.view.addItem(self.img)
        self._overlays: List[pg.GraphicsObject] = []
        self._has_image = False
        self.scene().sigMouseClicked.connect(self._on_click)

    def set_image(self, arr: Optional[np.ndarray], levels=None, *, preserve_view: bool = False) -> None:
        if arr is None:
            self.img.clear()
            self._has_image = False
            return
        old_range = None
        if preserve_view and self._has_image:
            try:
                old_range = [list(axis) for axis in self.view.viewRange()]
            except Exception:
                old_range = None
        a = np.asarray(arr)
        if a.dtype.kind == "f":
            # pyqtgraph autoLevels on NaN/Inf data yields NaN levels, which can
            # corrupt the LUT and crash the GPU/Qt path. Sanitize for display.
            a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
            if levels is None and a.size:
                lo, hi = float(a.min()), float(a.max())
                levels = (lo, hi if hi > lo else lo + 1.0)
        a = np.ascontiguousarray(a)
        _crumb(f"ImageCanvas.set_image setImage {a.shape} {a.dtype}")
        self.img.setImage(a, autoLevels=(levels is None), levels=levels)
        self._has_image = True
        if old_range is not None:
            _crumb("ImageCanvas.set_image restoreRange")
            self.view.setRange(xRange=old_range[0], yRange=old_range[1], padding=0)
        else:
            _crumb("ImageCanvas.set_image autoRange")
            self.view.autoRange()
        _crumb("ImageCanvas.set_image done")

    def image_item(self) -> pg.ImageItem:
        """Return the underlying image item for linked controls such as histograms."""
        return self.img

    def _on_click(self, ev) -> None:
        if ev.button() != QtCore.Qt.LeftButton:
            return
        p = self.view.mapSceneToView(ev.scenePos())
        self.clicked.emit(float(p.x()), float(p.y()))

    def clear_overlays(self) -> None:
        for item in self._overlays:
            self.view.removeItem(item)
        self._overlays = []

    def add_scatter(self, xs, ys, color="w", size=12) -> None:
        sp = pg.ScatterPlotItem(x=list(xs), y=list(ys), size=size,
                                brush=pg.mkBrush(color), pen=pg.mkPen("k", width=1))
        self.view.addItem(sp)
        self._overlays.append(sp)

    def add_line(self, xs, ys, color="y", width=3) -> None:
        ln = pg.PlotDataItem(x=list(xs), y=list(ys), pen=pg.mkPen(color, width=width))
        self.view.addItem(ln)
        self._overlays.append(ln)

    def add_text(
        self,
        text: str,
        x: float,
        y: float,
        color="w",
        *,
        fill=None,
        border=None,
        anchor=(0, 0),
    ) -> None:
        item = pg.TextItem(
            text=text,
            color=color,
            anchor=anchor,
            fill=fill if fill is not None else pg.mkBrush(21, 128, 61, 210),
            border=border if border is not None else pg.mkPen(255, 255, 255, 180),
        )
        item.setPos(float(x), float(y))
        item.setZValue(30)
        self.view.addItem(item)
        self._overlays.append(item)

    def add_line_roi(self, pts: np.ndarray, color, width: int = 3) -> pg.LineSegmentROI:
        pen = pg.mkPen(color, width=width)
        hover_pen = pg.mkPen(color, width=width + 2)
        roi = pg.LineSegmentROI([tuple(pts[0]), tuple(pts[1])], pen=pen, hoverPen=hover_pen)
        roi.setZValue(20)
        self.view.addItem(roi)
        self._overlays.append(roi)
        return roi

    def add_mask_overlay(self, mask: np.ndarray, color=(255, 40, 40)) -> None:
        mask = np.ascontiguousarray(np.asarray(mask).astype(bool))
        rgba = np.zeros((*mask.shape, 4), dtype=np.ubyte)
        rgba[mask] = [*color, 160]
        _crumb(f"add_mask_overlay ImageItem {rgba.shape}")
        item = pg.ImageItem(np.ascontiguousarray(rgba))
        item.setOpts(axisOrder="row-major")
        self.view.addItem(item)
        self._overlays.append(item)
        _crumb("add_mask_overlay done")


class Trajectory3DCanvas(QtWidgets.QWidget):
    """Matplotlib diagnostic view for CCF shell and probe trajectories."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._plot_theme = "Light"
        self._shell_cache_key: Optional[Tuple[str, Tuple[int, int, int]]] = None
        self._shell_cache: List[np.ndarray] = []
        self._last_atlas: Optional[hatlas.AllenCCFAtlas] = None
        self._last_probes: Optional[List[dict]] = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.figure = None
        self.canvas = None
        if Figure is None or FigureCanvasQTAgg is None:
            lbl = QtWidgets.QLabel("Matplotlib is unavailable.")
            lbl.setAlignment(QtCore.Qt.AlignCenter)
            lbl.setObjectName("SectionHint")
            layout.addWidget(lbl, 1)
            return

        self.figure = Figure(figsize=(4.2, 4.0), dpi=100)
        self.figure.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        layout.addWidget(self.canvas, 1)
        self.clear()

    def set_theme(self, theme: str) -> None:
        self._plot_theme = "Dark" if str(theme).lower().startswith("dark") else "Light"
        if self._last_probes is not None:
            self.render(self._last_atlas, self._last_probes)
        else:
            self.clear()

    def clear(self, message: str = "Build probe_ccf to render trajectories.") -> None:
        self._last_atlas = None
        self._last_probes = None
        if self.figure is None or self.canvas is None:
            return
        bg, fg, _wire = self._theme_colors()
        self.figure.clear()
        self.figure.patch.set_facecolor(bg)
        ax = self.figure.add_subplot(111)
        ax.set_facecolor(bg)
        ax.text(0.5, 0.5, message, ha="center", va="center", color=fg, fontsize=9)
        ax.set_axis_off()
        self.canvas.draw_idle()

    def render(self, atlas: Optional[hatlas.AllenCCFAtlas], probes) -> None:
        self._last_atlas = atlas
        self._last_probes = list(probes or [])
        if self.figure is None or self.canvas is None:
            return
        if not self._last_probes:
            self.clear("No probe trajectories to render.")
            return

        self.figure.clear()
        shell_polygons = self._brain_shell_polygons(atlas) if atlas is not None else []
        region_polygons = (
            self._build_probe_region_polygons(atlas, self._last_probes)
            if atlas is not None else []
        )
        self._draw_scene(
            self.figure,
            atlas,
            self._last_probes,
            self._plot_theme,
            azim=-58.0,
            shell_polygons=shell_polygons,
            region_polygons=region_polygons,
        )
        self.canvas.draw_idle()

    @classmethod
    def save_gif(
        cls,
        path: str | Path,
        atlas: Optional[hatlas.AllenCCFAtlas],
        probes,
        theme: str = "Light",
        frames: int = 36,
        fps: int = 12,
    ) -> Path:
        if Figure is None or FigureCanvasAgg is None:
            raise RuntimeError("Matplotlib is unavailable.")
        try:
            from PIL import Image
        except Exception as exc:
            raise RuntimeError("Pillow is required to save trajectory GIFs.") from exc

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        probes = list(probes or [])
        frames = max(int(frames), 2)
        fps = max(int(fps), 1)
        shell_polygons = cls._build_brain_shell_polygons(atlas) if atlas is not None else []
        region_polygons = cls._build_probe_region_polygons(atlas, probes) if atlas is not None else []

        fig = Figure(figsize=(5.6, 5.2), dpi=100)
        fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        canvas = FigureCanvasAgg(fig)
        images = []
        for azim in np.linspace(-68.0, 292.0, frames, endpoint=False):
            fig.clear()
            cls._draw_scene(
                fig,
                atlas,
                probes,
                theme,
                azim=float(azim),
                shell_polygons=shell_polygons,
                region_polygons=region_polygons,
            )
            canvas.draw()
            rgba = np.asarray(canvas.buffer_rgba()).copy()
            images.append(Image.fromarray(rgba[:, :, :3]))
        if not images:
            raise RuntimeError("No GIF frames were generated.")
        images[0].save(
            path,
            save_all=True,
            append_images=images[1:],
            duration=int(1000 / fps),
            loop=0,
            optimize=True,
        )
        return path

    @classmethod
    def _draw_scene(
        cls,
        figure,
        atlas: Optional[hatlas.AllenCCFAtlas],
        probes,
        theme: str,
        azim: float,
        shell_polygons: Optional[List[np.ndarray]] = None,
        region_polygons: Optional[List[Tuple[List[np.ndarray], Tuple[float, float, float]]]] = None,
    ) -> None:
        bg, fg, shell = cls._theme_colors_for(theme)
        figure.patch.set_facecolor(bg)
        ax = figure.add_subplot(111, projection="3d")
        ax.set_facecolor(bg)

        plotted_anything = False
        if atlas is not None:
            polygons = (
                shell_polygons
                if shell_polygons is not None
                else cls._build_brain_shell_polygons(atlas)
            )
            plotted_anything = cls._draw_brain_shell(ax, polygons, shell, theme)
            cls._draw_region_highlights(ax, region_polygons or [])
            cls._set_atlas_limits(ax, atlas)

        plotted_probe = False
        for i, probe in enumerate(probes):
            coords = cls._probe_coords(probe)
            if coords.shape[0] < 2:
                continue
            xyz = cls._ccf_to_plot_mm(coords)
            color = cls._probe_rgb(i)
            region_segments = cls._trajectory_region_segments(probe, coords)
            if region_segments:
                for seg_start, seg_end, seg_color in region_segments:
                    seg_xyz = cls._ccf_to_plot_mm(np.vstack([seg_start, seg_end]))
                    mesh = cls._tube_mesh(seg_xyz[0], seg_xyz[1], radius_mm=0.22, sides=20)
                    if mesh is None:
                        continue
                    ax.plot_surface(
                        mesh[0], mesh[1], mesh[2],
                        color=seg_color,
                        alpha=0.56,
                        linewidth=0,
                        antialiased=True,
                        shade=True,
                    )
            core_mesh = cls._tube_mesh(xyz[0], xyz[-1], radius_mm=0.065, sides=16)
            if core_mesh is not None:
                ax.plot_surface(
                    core_mesh[0], core_mesh[1], core_mesh[2],
                    color=color,
                    alpha=0.96,
                    linewidth=0,
                    antialiased=True,
                    shade=True,
                )
            ax.plot(
                xyz[:, 0], xyz[:, 1], xyz[:, 2],
                color="#7f0000", linewidth=1.1, alpha=0.98,
            )
            ax.scatter(
                xyz[:, 0], xyz[:, 1], xyz[:, 2],
                color=[color], s=24, depthshade=False,
                edgecolors="#fff0f0", linewidths=0.45,
            )
            ax.text(
                xyz[0, 0], xyz[0, 1], xyz[0, 2],
                f"P{i + 1}", color=color, fontsize=8,
            )
            plotted_anything = True
            plotted_probe = True

        if not plotted_probe:
            ax.text2D(
                0.5, 0.5, "No valid CCF trajectory points.",
                transform=ax.transAxes, ha="center", va="center", color=fg, fontsize=9,
            )
        elif not plotted_anything:
            ax.text2D(
                0.5, 0.5, "No atlas shell available.",
                transform=ax.transAxes, ha="center", va="center", color=fg, fontsize=9,
            )

        ax.view_init(elev=22, azim=azim)
        ax.set_axis_off()
        try:
            ax.set_box_aspect((1.0, 1.15, 0.82))
        except Exception:
            pass

    def _theme_colors(self) -> Tuple[str, str, str]:
        return self._theme_colors_for(self._plot_theme)

    @staticmethod
    def _theme_colors_for(theme: str) -> Tuple[str, str, str]:
        if str(theme).lower().startswith("dark"):
            return "#0b0f14", "#d7dde5", "#314253"
        return "#ffffff", "#27313a", "#d3dbe5"

    @staticmethod
    def _brain_shell_style_for(theme: str) -> Tuple[str, str, float, float, float]:
        if str(theme).lower().startswith("dark"):
            return "#aab7c2", "#dce7f0", 0.045, 0.26, 0.36
        return "#d9dee3", "#66717c", 0.055, 0.22, 0.32

    @staticmethod
    def _probe_rgb(index: int) -> Tuple[float, float, float]:
        return 0.86, 0.05, 0.04

    @staticmethod
    def _probe_coords(probe: dict) -> np.ndarray:
        coords = np.asarray(probe.get("trajectory_coords", np.zeros((0, 3))), dtype=float)
        if coords.ndim != 2 or coords.shape[0] < 2:
            coords = np.asarray(probe.get("points", np.zeros((0, 3))), dtype=float)
        if coords.ndim != 2 or coords.shape[1] < 3:
            return np.zeros((0, 3), dtype=float)
        coords = coords[:, :3]
        coords = coords[np.isfinite(coords).all(axis=1)]
        if coords.shape[0] < 2:
            return np.zeros((0, 3), dtype=float)
        return coords[np.argsort(coords[:, 1])]

    @staticmethod
    def _ccf_to_plot_mm(points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        rel = (pts - hatlas.BREGMA_CCF[None, :]) * (hatlas.CCF_VOXEL_UM / 1000.0)
        return np.column_stack([rel[:, 2], rel[:, 0], rel[:, 1]])

    @classmethod
    def _draw_brain_shell(
        cls,
        ax,
        polygons: List[np.ndarray],
        color: str,
        theme: str,
    ) -> bool:
        if not polygons:
            return False
        try:
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        except Exception:
            return False
        xyz_polygons = [
            cls._ccf_to_plot_mm(poly)
            for poly in polygons
            if np.asarray(poly).ndim == 2 and np.asarray(poly).shape[0] >= 3
        ]
        if not xyz_polygons:
            return False
        face_color, edge_color, face_alpha, line_alpha, line_width = cls._brain_shell_style_for(theme)
        shell = Poly3DCollection(
            xyz_polygons,
            facecolors=face_color,
            edgecolors="none",
            linewidths=0.0,
            alpha=face_alpha,
            zsort="average",
        )
        shell.set_antialiased(False)
        ax.add_collection3d(shell)
        for xyz in xyz_polygons:
            ax.plot(
                xyz[:, 0], xyz[:, 1], xyz[:, 2],
                color=edge_color,
                linewidth=line_width,
                alpha=line_alpha,
                solid_capstyle="round",
            )
        return True

    @classmethod
    def _draw_region_highlights(
        cls,
        ax,
        region_polygons: List[Tuple[List[np.ndarray], Tuple[float, float, float]]],
    ) -> bool:
        if not region_polygons:
            return False
        try:
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        except Exception:
            return False
        drew = False
        for polygons, color in region_polygons:
            xyz_polygons = [
                cls._ccf_to_plot_mm(poly)
                for poly in polygons
                if np.asarray(poly).ndim == 2 and np.asarray(poly).shape[0] >= 3
            ]
            if not xyz_polygons:
                continue
            coll = Poly3DCollection(
                xyz_polygons,
                facecolors=color,
                edgecolors=color,
                linewidths=0.25,
                alpha=0.22,
                zsort="average",
            )
            coll.set_antialiased(True)
            ax.add_collection3d(coll)
            drew = True
        return drew

    @classmethod
    def _atlas_row_values_for_region_id(cls, atlas: hatlas.AllenCCFAtlas, region_id) -> np.ndarray:
        try:
            rid = int(float(region_id))
        except (TypeError, ValueError):
            return np.zeros(0, dtype=int)
        st = getattr(atlas, "structure_tree", None)
        if st is None or "id" not in st:
            return np.zeros(0, dtype=int)
        ids = st["id"].astype(int).to_numpy()
        mask = ids == rid
        if "structure_id_path" in st:
            token = f"/{rid}/"
            paths = st["structure_id_path"].astype(str)
            mask = mask | paths.str.contains(token, regex=False).to_numpy()
        rows = np.flatnonzero(mask)
        return (rows + 1).astype(int)

    @classmethod
    def _build_probe_region_polygons(
        cls,
        atlas: Optional[hatlas.AllenCCFAtlas],
        probes,
        *,
        max_regions: int = 7,
        step: int = 18,
    ) -> List[Tuple[List[np.ndarray], Tuple[float, float, float]]]:
        if atlas is None or getattr(atlas, "av", None) is None:
            return []
        weighted: dict[int, dict] = {}
        for probe in probes or []:
            for row in cls._trajectory_area_rows(probe.get("trajectory_areas")):
                try:
                    rid = int(float(row.get("region_id")))
                    d0 = float(row.get("depth_start_um"))
                    d1 = float(row.get("depth_end_um"))
                except (TypeError, ValueError):
                    continue
                if rid <= 0 or not np.isfinite(d0) or not np.isfinite(d1) or d1 <= d0:
                    continue
                item = weighted.setdefault(rid, {
                    "span": 0.0,
                    "color": cls._hex_to_rgb(row.get("color_hex_triplet"), (0.4, 0.7, 0.55)),
                })
                item["span"] += float(d1 - d0)
        if not weighted:
            return []

        ranked = sorted(weighted.items(), key=lambda item: item[1]["span"], reverse=True)[:max_regions]
        step = max(8, int(step))
        av_small = np.asarray(atlas.av[::step, ::step, ::step])
        out: List[Tuple[List[np.ndarray], Tuple[float, float, float]]] = []

        for rid, meta in ranked:
            row_values = cls._atlas_row_values_for_region_id(atlas, rid)
            if row_values.size == 0:
                continue
            mask = np.isin(av_small, row_values)
            if int(mask.sum()) < 8:
                continue
            polygons: List[np.ndarray] = []
            present_ap = np.flatnonzero(mask.any(axis=(1, 2)))
            present_ml = np.flatnonzero(mask.any(axis=(0, 1)))
            present_dv = np.flatnonzero(mask.any(axis=(0, 2)))

            def selected(values: np.ndarray, count: int) -> np.ndarray:
                if values.size == 0:
                    return values
                if values.size <= count:
                    return values
                return values[np.unique(np.linspace(0, values.size - 1, count, dtype=int))]

            for ap_i in selected(present_ap, 7):
                for y, x in cls._contour_lines(mask[int(ap_i), :, :], max_points=86):
                    polygons.append(np.column_stack([
                        np.full(y.shape, int(ap_i) * step + 1.0),
                        y * step + 1.0,
                        x * step + 1.0,
                    ]))
            for ml_i in selected(present_ml, 4):
                for y, x in cls._contour_lines(mask[:, :, int(ml_i)], max_points=72):
                    polygons.append(np.column_stack([
                        y * step + 1.0,
                        x * step + 1.0,
                        np.full(y.shape, int(ml_i) * step + 1.0),
                    ]))
            for dv_i in selected(present_dv, 3):
                for y, x in cls._contour_lines(mask[:, int(dv_i), :], max_points=72):
                    polygons.append(np.column_stack([
                        y * step + 1.0,
                        np.full(y.shape, int(dv_i) * step + 1.0),
                        x * step + 1.0,
                    ]))
            if polygons:
                out.append((polygons, meta["color"]))
        return out

    @staticmethod
    def _tube_mesh(
        start_xyz: np.ndarray,
        end_xyz: np.ndarray,
        radius_mm: float,
        sides: int = 10,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        p0 = np.asarray(start_xyz, dtype=float).reshape(3)
        p1 = np.asarray(end_xyz, dtype=float).reshape(3)
        axis = p1 - p0
        length = float(np.linalg.norm(axis))
        if not np.isfinite(length) or length < 1e-9:
            return None
        direction = axis / length
        reference = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(direction, reference))) > 0.92:
            reference = np.array([0.0, 1.0, 0.0])
        u = np.cross(direction, reference)
        u_norm = float(np.linalg.norm(u))
        if not np.isfinite(u_norm) or u_norm < 1e-9:
            return None
        u /= u_norm
        v = np.cross(direction, u)
        theta = np.linspace(0.0, 2.0 * np.pi, max(int(sides), 6) + 1)
        circle = radius_mm * (
            np.cos(theta)[:, None] * u[None, :]
            + np.sin(theta)[:, None] * v[None, :]
        )
        tube = np.stack([p0[None, :] + circle, p1[None, :] + circle], axis=0)
        return tube[:, :, 0], tube[:, :, 1], tube[:, :, 2]

    @staticmethod
    def _hex_to_rgb(hex_triplet, fallback: Tuple[float, float, float]) -> Tuple[float, float, float]:
        h = str(hex_triplet or "").strip().lstrip("#")
        if len(h) == 5:
            h = "0" + h
        try:
            if len(h) >= 6:
                return (
                    int(h[0:2], 16) / 255.0,
                    int(h[2:4], 16) / 255.0,
                    int(h[4:6], 16) / 255.0,
                )
        except (TypeError, ValueError):
            pass
        return fallback

    @staticmethod
    def _flatten_field(value) -> List:
        if value is None:
            return []
        arr = np.asarray(value, dtype=object)
        return arr.reshape(-1).tolist()

    @classmethod
    def _trajectory_area_rows(cls, areas) -> List[dict]:
        if areas is None:
            return []
        rows: List[dict] = []
        if hasattr(areas, "iloc"):
            for j in range(len(areas)):
                row = areas.iloc[j]
                d0 = row.get("depth_start_um", None)
                d1 = row.get("depth_end_um", None)
                if d0 is None or d1 is None:
                    depth = np.asarray(row.get("trajectory_depth", []), dtype=float).reshape(-1)
                    if depth.size >= 2:
                        d0, d1 = depth[:2]
                rows.append({
                    "depth_start_um": d0,
                    "depth_end_um": d1,
                    "acronym": row.get("acronym", ""),
                    "region_id": row.get("id", row.get("region_id", "")),
                    "color_hex_triplet": row.get("color_hex_triplet", ""),
                })
            return rows
        if isinstance(areas, dict):
            colors = cls._flatten_field(areas.get("color_hex_triplet"))
            acronyms = cls._flatten_field(areas.get("acronym"))
            region_ids = cls._flatten_field(areas.get("id"))
            if not region_ids:
                region_ids = cls._flatten_field(areas.get("region_id"))
            if "depth_start_um" in areas and "depth_end_um" in areas:
                starts = np.asarray(areas.get("depth_start_um"), dtype=float).reshape(-1)
                ends = np.asarray(areas.get("depth_end_um"), dtype=float).reshape(-1)
                n = min(len(starts), len(ends))
                for j in range(n):
                    rows.append({
                        "depth_start_um": starts[j],
                        "depth_end_um": ends[j],
                        "acronym": acronyms[j] if j < len(acronyms) else "",
                        "region_id": region_ids[j] if j < len(region_ids) else "",
                        "color_hex_triplet": colors[j] if j < len(colors) else "",
                    })
                return rows
            depth = np.asarray(areas.get("trajectory_depth", np.zeros((0, 2))), dtype=float)
            if depth.size:
                depth = depth.reshape(-1, 2)
                for j, (d0, d1) in enumerate(depth):
                    rows.append({
                        "depth_start_um": d0,
                        "depth_end_um": d1,
                        "acronym": acronyms[j] if j < len(acronyms) else "",
                        "region_id": region_ids[j] if j < len(region_ids) else "",
                        "color_hex_triplet": colors[j] if j < len(colors) else "",
                    })
        return rows

    @classmethod
    def _trajectory_region_segments(
        cls,
        probe: dict,
        coords: np.ndarray,
    ) -> List[Tuple[np.ndarray, np.ndarray, Tuple[float, float, float]]]:
        rows = cls._trajectory_area_rows(probe.get("trajectory_areas"))
        if not rows or coords.shape[0] < 2:
            return []
        entry = np.asarray(coords[0], dtype=float)
        exit_ = np.asarray(coords[-1], dtype=float)
        axis = exit_ - entry
        length_um = float(np.linalg.norm(axis) * hatlas.CCF_VOXEL_UM)
        if not np.isfinite(length_um) or length_um <= 0.0:
            return []

        parsed = []
        for row in rows:
            try:
                d0 = float(row.get("depth_start_um"))
                d1 = float(row.get("depth_end_um"))
            except (TypeError, ValueError):
                continue
            if not (np.isfinite(d0) and np.isfinite(d1)) or d1 <= d0:
                continue
            if d1 - d0 < 12.0:
                continue
            parsed.append((d0, d1, row.get("color_hex_triplet", "")))
        if not parsed:
            return []

        depth_max = max(float(max(d1 for _, d1, _ in parsed)), 1.0)
        fallback = (0.55, 0.60, 0.66)
        segments: List[Tuple[np.ndarray, np.ndarray, Tuple[float, float, float]]] = []
        for d0, d1, hex_color in parsed:
            f0 = float(np.clip(d0 / depth_max, 0.0, 1.0))
            f1 = float(np.clip(d1 / depth_max, 0.0, 1.0))
            if f1 <= f0:
                continue
            p0 = entry + axis * f0
            p1 = entry + axis * f1
            segments.append((p0, p1, cls._hex_to_rgb(hex_color, fallback)))
        return segments

    @staticmethod
    def _boundary_mask(mask: np.ndarray) -> np.ndarray:
        m = np.asarray(mask, dtype=bool)
        if m.size == 0:
            return m
        p = np.pad(m, 1, mode="constant", constant_values=False)
        interior = (
            m
            & p[1:-1, :-2]
            & p[1:-1, 2:]
            & p[:-2, 1:-1]
            & p[2:, 1:-1]
        )
        return m & ~interior

    @classmethod
    def _contour_lines(cls, mask: np.ndarray, max_points: int = 180) -> List[Tuple[np.ndarray, np.ndarray]]:
        m = np.asarray(mask, dtype=bool)
        if not np.any(m):
            return []

        components: List[np.ndarray] = []
        try:
            from scipy import ndimage

            labels, n_labels = ndimage.label(m)
            if n_labels:
                idx = np.arange(1, n_labels + 1)
                sizes = ndimage.sum(m, labels, index=idx)
                keep = idx[np.argsort(sizes)[-4:]]
                components = [labels == int(label) for label in keep if sizes[int(label) - 1] > 8]
        except Exception:
            components = [m]

        lines: List[Tuple[np.ndarray, np.ndarray]] = []
        for comp in components or [m]:
            yy, xx = np.nonzero(cls._boundary_mask(comp))
            if yy.size < 5:
                continue
            cy = float(np.mean(yy))
            cx = float(np.mean(xx))
            order = np.argsort(np.arctan2(yy - cy, xx - cx))
            if order.size > max_points:
                take = np.linspace(0, order.size - 1, max_points, dtype=int)
                order = order[take]
            y = yy[order].astype(float)
            x = xx[order].astype(float)
            if y.size:
                y = np.r_[y, y[0]]
                x = np.r_[x, x[0]]
            lines.append((y, x))
        return lines

    def _brain_shell_polygons(self, atlas: hatlas.AllenCCFAtlas) -> List[np.ndarray]:
        key = (str(atlas.atlas_path), tuple(atlas.shape))
        if self._shell_cache_key == key:
            return self._shell_cache
        self._shell_cache_key = key
        self._shell_cache = self._build_brain_shell_polygons(atlas)
        return self._shell_cache

    @classmethod
    def _build_brain_shell_polygons(cls, atlas: hatlas.AllenCCFAtlas) -> List[np.ndarray]:
        av = atlas.av
        ap_n, dv_n, ml_n = atlas.shape
        polygons: List[np.ndarray] = []

        for ap_i in np.unique(np.linspace(0, ap_n - 1, 15, dtype=int)):
            mask = np.asarray(av[ap_i, :, :]) > 1
            for y, x in cls._contour_lines(mask, max_points=128):
                polygons.append(np.column_stack([
                    np.full(y.shape, ap_i + 1.0), y + 1.0, x + 1.0,
                ]))

        for ml_i in np.unique(np.linspace(0, ml_n - 1, 9, dtype=int)):
            mask = np.asarray(av[:, :, ml_i]) > 1
            for y, x in cls._contour_lines(mask, max_points=128):
                polygons.append(np.column_stack([
                    y + 1.0, x + 1.0, np.full(y.shape, ml_i + 1.0),
                ]))

        for dv_i in np.unique(np.linspace(0, dv_n - 1, 5, dtype=int)):
            mask = np.asarray(av[:, dv_i, :]) > 1
            for y, x in cls._contour_lines(mask, max_points=112):
                polygons.append(np.column_stack([
                    y + 1.0, np.full(y.shape, dv_i + 1.0), x + 1.0,
                ]))

        return [np.asarray(poly, dtype=float) for poly in polygons if len(poly) >= 3]

    @classmethod
    def _set_atlas_limits(cls, ax, atlas: hatlas.AllenCCFAtlas) -> None:
        ap_n, dv_n, ml_n = atlas.shape
        corners = np.array([
            [1.0, 1.0, 1.0],
            [float(ap_n), float(dv_n), float(ml_n)],
        ])
        xyz = cls._ccf_to_plot_mm(corners)
        ax.set_xlim(float(np.min(xyz[:, 0])), float(np.max(xyz[:, 0])))
        ax.set_ylim(float(np.min(xyz[:, 1])), float(np.max(xyz[:, 1])))
        ax.set_zlim(float(np.max(xyz[:, 2])), float(np.min(xyz[:, 2])))


class HistologyTab(QtWidgets.QWidget):
    """Probe-localization tab driving the AP_histology pipeline plus the IBL bridge.

    Walks the user through Setup, Preprocess, Match, Align, Trace, Channel map and
    IBL refine (see the module docstring). Heavy work runs on a dedicated thread
    pool so histology stays responsive while other tabs run long sorting jobs.
    """

    #: Emitted from worker threads so log lines reach the GUI thread safely.
    log_requested = QtCore.Signal(str)
    _SELECTION_PATH_ROLE = QtCore.Qt.UserRole
    _SELECTION_SELECTED_ROLE = QtCore.Qt.UserRole + 1
    _SELECTION_PIXMAP_ROLE = QtCore.Qt.UserRole + 2

    def __init__(self, thread_pool: QtCore.QThreadPool) -> None:
        super().__init__()
        # Histology work (atlas sampling, probe_ccf, IBL bridge) is short and
        # interactive. The shared global pool can be fully occupied for hours by
        # Kilosort/CatGT jobs queued in the Preprocessing tab, which would leave an
        # instant probe_ccf build "Building..." forever. A dedicated pool keeps
        # histology responsive regardless of what other tabs are running.
        self.pool = QtCore.QThreadPool(self)
        self.pool.setMaxThreadCount(max(2, min(4, acceleration.auto_worker_count())))
        self._shared_pool = thread_pool
        self.settings = QtCore.QSettings("NeuroPyGuiN", "NeuroPyGuiN")
        self._busy_count = 0
        self._plot_theme = "Light"

        # Pipeline state
        self.folder: Optional[Path] = None
        self.atlas: Optional[hatlas.AllenCCFAtlas] = None
        self.raw_image_paths: List[Path] = []
        self.selected_raw_paths: List[Path] = []
        self.validated_raw_paths: List[Path] = []
        self._selection_grid_updating = False
        self._selection_preview_image: Optional[np.ndarray] = None
        self.slice_images: List[np.ndarray] = []
        self.slice_pixel_um: List[Optional[float]] = []
        self.histology_ccf: List[Dict[str, np.ndarray]] = []
        self.tforms: List[np.ndarray] = []
        self.slice_specs: List[Optional[Dict[str, np.ndarray]]] = []
        self.probe_points: Dict[Tuple[int, int], np.ndarray] = {}
        self._cur_match_slice = 0
        self._hist_levels_by_stage: Dict[str, Dict[int, Tuple[float, float]]] = {
            "select": {},
            "preproc": {},
            "match": {},
            "align": {},
            "trace": {},
        }
        self._histogram_widgets: Dict[str, pg.HistogramLUTWidget] = {}
        self._histogram_canvases: Dict[str, ImageCanvas] = {}
        self._updating_histogram_stage: Optional[str] = None
        self._match_hist_levels_by_slice = self._hist_levels_by_stage["match"]
        self._updating_match_histogram = False
        self._cur_align_slice = 0
        self._cur_trace_slice = 0
        self._align_hist_pts: Dict[int, List[Tuple[float, float]]] = {}
        self._align_atlas_pts: Dict[int, List[Tuple[float, float]]] = {}
        self._active_probe = 1
        self._pending_click: List[Tuple[float, float]] = []
        self._trace_roi: Optional[pg.LineSegmentROI] = None
        self._trace_updating_roi = False
        self._trace_updating_controls = False
        self._trace_coord_spins: Dict[str, QtWidgets.QDoubleSpinBox] = {}
        self._trajectory_3d_dialog: Optional[QtWidgets.QDialog] = None
        self._trajectory_3d_popup: Optional[Trajectory3DCanvas] = None
        self._trajectory_3d_status: Optional[QtWidgets.QLabel] = None

        self._build_ui()
        self._restore_settings()
        self.log_requested.connect(self._log)  # queued: safe to emit from workers
        self._log(f"Histology acceleration: {acceleration.hardware_summary()}")

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        main = QtWidgets.QVBoxLayout(self)
        main.setContentsMargins(0, 0, 0, 0)
        main.setSpacing(0)

        self.nav = SideNavStack(
            "Histology", "Localize Neuropixels probes on the Allen CCF.",
        )
        main.addWidget(self.nav, 1)

        self._page_setup = self.nav.add_page("Setup", self._build_setup_page())
        self._page_select_images = self.nav.add_page("Select images", self._build_image_selection_page())
        self._page_preprocess = self.nav.add_page("Preprocess", self._build_preprocess_page())
        self._page_match = self.nav.add_page("Match atlas", self._build_match_page())
        self._page_align = self.nav.add_page("Align", self._build_align_page())
        self._page_trace = self.nav.add_page("Trace probes", self._build_trace_page())
        self._page_channels = self.nav.add_page("Channel map", self._build_channels_page())
        self._page_ibl = self.nav.add_page("IBL refine", self._build_ibl_page())
        self.nav.currentChanged.connect(self._on_page_changed)
        self.nav.setCurrentIndex(0)

        # Shared log dock at the bottom.
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        self.log.setFixedHeight(120)
        self.log.setObjectName("HistologyLog")
        main.addWidget(self.log, 0)

    def _on_page_changed(self, index: int) -> None:
        """Refresh lazily-rendered pages whenever the user navigates to them."""
        if index == getattr(self, "_page_preprocess", -1):
            self._preproc_show()
        elif index == getattr(self, "_page_select_images", -1):
            self._selection_refresh()
        elif index == getattr(self, "_page_match", -1):
            self._match_show()
        elif index == getattr(self, "_page_align", -1):
            self._align_show()
        elif index == getattr(self, "_page_trace", -1):
            self._trace_show()

    def _section(self, title: str, hint: str = "") -> Tuple[QtWidgets.QWidget, QtWidgets.QVBoxLayout]:
        page = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(page)
        v.setContentsMargins(18, 16, 18, 16)
        v.setSpacing(12)
        t = QtWidgets.QLabel(title)
        t.setObjectName("FieldTitle")
        v.addWidget(t)
        if hint:
            h = QtWidgets.QLabel(hint)
            h.setObjectName("SectionHint")
            h.setWordWrap(True)
            v.addWidget(h)
        return page, v

    @staticmethod
    def _workflow_steps_text() -> str:
        return (
            "Order: 1 Setup paths -> 2 Select images -> 3 Preprocess/save slices -> "
            "4 Match atlas -> 5 Align -> 6 Trace probes -> 7 Channel map -> "
            "8 optional IBL refine/finalize"
        )

    def _histology_contrast_pane(self, stage: str, canvas: ImageCanvas) -> QtWidgets.QWidget:
        """Wrap a histology image canvas with a reusable intensity histogram control."""
        histogram = pg.HistogramLUTWidget()
        histogram.setMinimumWidth(96)
        histogram.setMaximumWidth(150)
        histogram.item.gradient.hide()
        histogram.setImageItem(canvas.image_item())
        histogram.item.sigLevelsChanged.connect(lambda: self._histology_levels_changed(stage))
        self._histogram_widgets[stage] = histogram
        self._histogram_canvases[stage] = canvas
        if stage == "match":
            self.hist_match_histogram = histogram

        b_reset = QtWidgets.QPushButton("Auto")
        b_reset.setProperty("role", "secondary")
        b_reset.setToolTip("Reset histology image contrast to automatic levels.")
        b_reset.clicked.connect(lambda: self._reset_histology_levels(stage))

        pane = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(pane)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(canvas, 1)
        controls = QtWidgets.QVBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(6)
        controls.addWidget(histogram, 1)
        controls.addWidget(b_reset, 0)
        layout.addLayout(controls, 0)
        return pane

    def _current_histology_slice_index(self, stage: str) -> int:
        if stage == "select":
            row = self.lst_raw_images.currentRow() if hasattr(self, "lst_raw_images") else 0
            return max(0, int(row))
        if stage == "preproc":
            return int(getattr(self, "_cur_preproc", 0))
        if stage == "align":
            return int(self._cur_align_slice)
        if stage == "trace":
            return int(self._cur_trace_slice)
        return int(self._cur_match_slice)

    def _show_histology_slice(
        self,
        stage: str,
        canvas: ImageCanvas,
        idx: int,
        image: np.ndarray,
        *,
        preserve_view: bool = False,
    ) -> None:
        """Display a histology slice and apply the stage-specific contrast memory."""
        canvas.set_image(image, preserve_view=preserve_view)
        level_key = 0 if stage == "select" else idx
        levels = self._hist_levels_by_stage.setdefault(stage, {}).get(level_key)
        if levels is None:
            self._reset_histology_levels(stage, idx=level_key, remember=True)
        else:
            self._apply_histology_levels(stage, levels[0], levels[1])

    def _histology_levels_changed(self, stage: str) -> None:
        """Remember manually adjusted contrast levels for the active slice."""
        if self._updating_histogram_stage == stage or self._updating_match_histogram:
            return
        n_images = len(self.raw_image_paths) if stage == "select" else len(self.slice_images)
        if n_images <= 0:
            return
        idx = 0 if stage == "select" else min(self._current_histology_slice_index(stage), n_images - 1)
        widget = self._histogram_widgets.get(stage)
        if widget is None:
            return
        try:
            lo, hi = widget.item.getLevels()
        except Exception:
            return
        self._hist_levels_by_stage.setdefault(stage, {})[idx] = (float(lo), float(hi))

    def _remember_visible_histology_levels(self, stage: str) -> None:
        """Persist the levels currently applied to a visible image item."""
        canvas = self._histogram_canvases.get(stage)
        if canvas is None:
            return
        levels = getattr(canvas.image_item(), "levels", None)
        if levels is None:
            widget = self._histogram_widgets.get(stage)
            if widget is None:
                return
            try:
                levels = widget.item.getLevels()
            except Exception:
                return
        arr = np.asarray(levels, dtype=float).reshape(-1)
        if arr.size < 2 or not np.isfinite(arr[:2]).all() or arr[1] <= arr[0]:
            return
        key = 0 if stage == "select" else self._current_histology_slice_index(stage)
        self._hist_levels_by_stage.setdefault(stage, {})[key] = (float(arr[0]), float(arr[1]))

    def _apply_histology_levels(self, stage: str, lo: float, hi: float) -> None:
        """Apply contrast levels to the image and its linked histogram widget."""
        widget = self._histogram_widgets.get(stage)
        canvas = self._histogram_canvases.get(stage)
        if widget is None or canvas is None:
            return
        self._updating_histogram_stage = stage
        if stage == "match":
            self._updating_match_histogram = True
        try:
            canvas.image_item().setLevels((float(lo), float(hi)))
            widget.setLevels(float(lo), float(hi))
        finally:
            if stage == "match":
                self._updating_match_histogram = False
            self._updating_histogram_stage = None

    def _reset_histology_levels(
        self,
        stage: str,
        *,
        idx: Optional[int] = None,
        remember: bool = True,
    ) -> None:
        """Reset the selected histology contrast control to robust image percentiles."""
        if stage == "select":
            if self._selection_preview_image is None or not self.raw_image_paths:
                return
            n_images = len(self.raw_image_paths)
            idx = 0
            arr = np.asarray(self._selection_preview_image)
        else:
            if not self.slice_images:
                return
            n_images = len(self.slice_images)
            idx = min(self._current_histology_slice_index(stage) if idx is None else idx, n_images - 1)
            arr = np.asarray(self.slice_images[idx])
        if arr.size == 0:
            return
        if arr.ndim == 3:
            values = arr[..., :3].astype(float).mean(axis=2)
        else:
            values = arr.astype(float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return
        lo, hi = np.percentile(values, [0.5, 99.5])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(values.min()), float(values.max())
        if hi <= lo:
            hi = lo + 1.0
        try:
            self._apply_histology_levels(stage, float(lo), float(hi))
            if remember:
                self._hist_levels_by_stage.setdefault(stage, {})[idx] = (float(lo), float(hi))
        except Exception as exc:
            self._log(f"Could not reset histology contrast: {exc}")

    # ---- Setup page ----
    def _build_setup_page(self) -> QtWidgets.QWidget:
        page, v = self._section(
            "Session and tools",
            "Point at the session histology folder (where products are written) and "
            "the Allen CCF atlas. Optional paths enable the IBL refinement step.",
        )
        form = QtWidgets.QFormLayout()
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)

        self.ed_folder = QtWidgets.QLineEdit()
        b_folder = QtWidgets.QPushButton("Browse...")
        b_folder.clicked.connect(self._pick_folder)
        form.addRow("Histology folder", self._with_button(self.ed_folder, b_folder))

        self.ed_raw = QtWidgets.QLineEdit()
        self.ed_raw.editingFinished.connect(self._selection_refresh)
        b_raw = QtWidgets.QPushButton("Browse...")
        b_raw.clicked.connect(lambda: self._pick_into(self.ed_raw, "Raw image folder"))
        form.addRow("Raw images", self._with_button(self.ed_raw, b_raw))

        self.ed_atlas = QtWidgets.QLineEdit(hatlas.DEFAULT_ATLAS_PATH)
        b_atlas = QtWidgets.QPushButton("Browse...")
        b_atlas.clicked.connect(lambda: self._pick_into(self.ed_atlas, "Allen CCF atlas folder"))
        form.addRow("Atlas folder", self._with_button(self.ed_atlas, b_atlas))

        self.ed_ks = QtWidgets.QLineEdit()
        b_ks = QtWidgets.QPushButton("Browse...")
        b_ks.clicked.connect(lambda: self._pick_into(self.ed_ks, "Kilosort output folder"))
        form.addRow("Kilosort folder", self._with_button(self.ed_ks, b_ks))

        self.ed_ephys = QtWidgets.QLineEdit()
        b_ephys = QtWidgets.QPushButton("Browse...")
        b_ephys.clicked.connect(lambda: self._pick_into(self.ed_ephys, "Raw ephys folder (.ap.bin)"))
        form.addRow("Ephys folder", self._with_button(self.ed_ephys, b_ephys))

        self.ed_iblapps = QtWidgets.QLineEdit(ibl_launch.DEFAULT_IBLAPPS_PATH)
        form.addRow("iblapps path", self.ed_iblapps)
        self.ed_pyexe = QtWidgets.QLineEdit()
        self.ed_pyexe.setPlaceholderText("auto-detect (interpreter with iblatlas)")
        form.addRow("IBL python", self.ed_pyexe)

        v.addLayout(form)

        order = QtWidgets.QLabel(self._workflow_steps_text())
        order.setObjectName("SectionHint")
        order.setWordWrap(True)
        v.addWidget(order)

        row = QtWidgets.QHBoxLayout()
        b_load = QtWidgets.QPushButton("Load session")
        b_load.clicked.connect(self._load_session)
        b_save = QtWidgets.QPushButton("Save paths")
        b_save.clicked.connect(self._persist_settings)
        row.addWidget(b_load)
        row.addWidget(b_save)
        row.addStretch(1)
        v.addLayout(row)

        self.lbl_status = QtWidgets.QLabel("No session loaded.")
        self.lbl_status.setObjectName("SectionHint")
        self.lbl_status.setTextFormat(QtCore.Qt.RichText)
        v.addWidget(self.lbl_status)
        v.addStretch(1)
        return page

    # ---- Image selection page ----
    def _build_image_selection_page(self) -> QtWidgets.QWidget:
        page, v = self._section(
            "Select raw images",
            "Choose which raw TIFF/PNG images should enter preprocessing. Stamped images "
            "are loaded and carried forward; unmarked images are ignored.",
        )
        order = QtWidgets.QLabel(self._workflow_steps_text())
        order.setObjectName("SectionHint")
        order.setWordWrap(True)
        v.addWidget(order)

        row = QtWidgets.QHBoxLayout()
        b_refresh = QtWidgets.QPushButton("Refresh raw folder")
        b_refresh.clicked.connect(self._selection_refresh)
        b_all = QtWidgets.QPushButton("Select all")
        b_all.clicked.connect(lambda: self._selection_set_all(True))
        b_none = QtWidgets.QPushButton("Select none")
        b_none.clicked.connect(lambda: self._selection_set_all(False))
        b_invert = QtWidgets.QPushButton("Invert")
        b_invert.clicked.connect(self._selection_invert)
        b_validate = QtWidgets.QPushButton("Validate selection")
        b_validate.setProperty("role", "primary")
        b_validate.setToolTip("Commit the selected raw images and load exactly that list into Preprocess.")
        b_validate.clicked.connect(self._selection_validate)
        for b in (b_refresh, b_all, b_none, b_invert, b_validate):
            row.addWidget(b)
        row.addStretch(1)
        self.lbl_image_selection = QtWidgets.QLabel("No raw folder loaded.")
        self.lbl_image_selection.setObjectName("SectionHint")
        row.addWidget(self.lbl_image_selection)
        v.addLayout(row)

        self.lst_raw_images = QtWidgets.QListWidget()
        self.lst_raw_images.setViewMode(QtWidgets.QListView.IconMode)
        self.lst_raw_images.setResizeMode(QtWidgets.QListView.Adjust)
        self.lst_raw_images.setMovement(QtWidgets.QListView.Static)
        self.lst_raw_images.setIconSize(QtCore.QSize(150, 110))
        self.lst_raw_images.setGridSize(QtCore.QSize(178, 150))
        self.lst_raw_images.setUniformItemSizes(True)
        self.lst_raw_images.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.lst_raw_images.itemClicked.connect(self._selection_toggle_item)

        self.canvas_select_raw = ImageCanvas()
        self.lbl_select_preview = QtWidgets.QLabel("No raw image preview.")
        self.lbl_select_preview.setObjectName("SectionHint")
        self.lbl_select_preview.setAlignment(QtCore.Qt.AlignCenter)
        preview = QtWidgets.QWidget()
        preview_layout = QtWidgets.QVBoxLayout(preview)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(8)
        preview_layout.addWidget(self._histology_contrast_pane("select", self.canvas_select_raw), 1)
        preview_layout.addWidget(self.lbl_select_preview, 0)

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        split.addWidget(self.lst_raw_images)
        split.addWidget(self._titled("Raw image preview", preview))
        split.setSizes([720, 520])
        v.addWidget(split, 1)
        return page

    # ---- Preprocess page ----
    def _build_preprocess_page(self) -> QtWidgets.QWidget:
        page, v = self._section(
            "Slice preprocessing",
            "Load the selected raw TIFF/PNG images, optionally downsample, then save "
            "individual slice images. Use the reorient buttons to fix rotation/flip/order.",
        )
        ctl = QtWidgets.QHBoxLayout()
        self.sp_downsample = QtWidgets.QDoubleSpinBox()
        self.sp_downsample.setRange(1, 50)
        self.sp_downsample.setValue(1)
        self.sp_downsample.setPrefix("1/")
        ctl.addWidget(QtWidgets.QLabel("Downsample"))
        ctl.addWidget(self.sp_downsample)
        b_loadimg = QtWidgets.QPushButton("Load raw images")
        b_loadimg.clicked.connect(lambda: self._preproc_load())
        ctl.addWidget(b_loadimg)
        b_save_slices = QtWidgets.QPushButton("Save slices")
        b_save_slices.clicked.connect(self._preproc_save)
        ctl.addWidget(b_save_slices)
        ctl.addStretch(1)
        v.addLayout(ctl)

        nav = QtWidgets.QHBoxLayout()
        b_prev = QtWidgets.QPushButton("< Prev")
        b_prev.clicked.connect(lambda: self._preproc_step(-1))
        b_next = QtWidgets.QPushButton("Next >")
        b_next.clicked.connect(lambda: self._preproc_step(1))
        b_rotl = QtWidgets.QPushButton("Rotate -90")
        b_rotl.clicked.connect(lambda: self._preproc_rotate(-90))
        b_rotr = QtWidgets.QPushButton("Rotate +90")
        b_rotr.clicked.connect(lambda: self._preproc_rotate(90))
        b_fliph = QtWidgets.QPushButton("Flip H")
        b_fliph.clicked.connect(lambda: self._preproc_flip(True))
        b_flipv = QtWidgets.QPushButton("Flip V")
        b_flipv.clicked.connect(lambda: self._preproc_flip(False))
        for b in [b_prev, b_next, b_rotl, b_rotr, b_fliph, b_flipv]:
            nav.addWidget(b)
        nav.addStretch(1)
        self.lbl_preproc = QtWidgets.QLabel("0 / 0")
        nav.addWidget(self.lbl_preproc)
        v.addLayout(nav)

        self.canvas_preproc = ImageCanvas()
        v.addWidget(self._histology_contrast_pane("preproc", self.canvas_preproc), 1)
        return page

    # ---- Match page ----
    def _build_match_page(self) -> QtWidgets.QWidget:
        page, v = self._section(
            "Match atlas slices",
            "For each histology slice, dial in the Allen CCF plane (AP position and "
            "small tilts), then Assign. Save writes histology_ccf.mat.",
        )
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.canvas_match_hist = ImageCanvas()
        self.canvas_match_atlas = ImageCanvas()
        split.addWidget(self._titled("Histology", self._histology_contrast_pane("match", self.canvas_match_hist)))
        split.addWidget(self._titled("Atlas plane", self.canvas_match_atlas))
        v.addWidget(split, 1)

        ctl = QtWidgets.QHBoxLayout()
        b_prev = QtWidgets.QPushButton("< Slice")
        b_prev.clicked.connect(lambda: self._match_step(-1))
        b_next = QtWidgets.QPushButton("Slice >")
        b_next.clicked.connect(lambda: self._match_step(1))
        b_auto_match = QtWidgets.QPushButton("Auto match")
        b_auto_match.setProperty("role", "secondary")
        b_auto_match.setToolTip("Estimate the best AP plane near the current AP slider from tissue shape and signal.")
        b_auto_match.clicked.connect(self._match_auto)
        self.ck_match_auto_adjust = QtWidgets.QCheckBox("Auto-adjust histology")
        self.ck_match_auto_adjust.setToolTip(
            "After Auto match, assign the plane and run accelerated shape auto-align "
            "for this slice, then autosave progress."
        )
        ctl.addWidget(b_prev)
        ctl.addWidget(b_next)
        ctl.addWidget(b_auto_match)
        ctl.addWidget(self.ck_match_auto_adjust)
        ctl.addWidget(QtWidgets.QLabel("AP"))
        self.sl_ap = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sl_ap.setRange(1, 1320)
        self.sl_ap.setValue(540)
        self.sl_ap.valueChanged.connect(self._match_update_atlas)
        self.lbl_ap_value = QtWidgets.QLabel("")
        self.lbl_ap_value.setMinimumWidth(58)
        self.lbl_ap_value.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        ctl.addWidget(self.sl_ap, 2)
        ctl.addWidget(self.lbl_ap_value)
        ctl.addWidget(QtWidgets.QLabel("LR tilt"))
        self.sl_lr = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sl_lr.setRange(-15, 15)
        self.sl_lr.valueChanged.connect(self._match_update_atlas)
        self.lbl_lr_value = QtWidgets.QLabel("")
        self.lbl_lr_value.setMinimumWidth(38)
        self.lbl_lr_value.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        ctl.addWidget(self.sl_lr, 1)
        ctl.addWidget(self.lbl_lr_value)
        ctl.addWidget(QtWidgets.QLabel("SI tilt"))
        self.sl_si = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sl_si.setRange(-15, 15)
        self.sl_si.valueChanged.connect(self._match_update_atlas)
        self.lbl_si_value = QtWidgets.QLabel("")
        self.lbl_si_value.setMinimumWidth(38)
        self.lbl_si_value.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        ctl.addWidget(self.sl_si, 1)
        ctl.addWidget(self.lbl_si_value)
        b_reset_tilt = QtWidgets.QPushButton("Reset tilt")
        b_reset_tilt.setProperty("role", "secondary")
        b_reset_tilt.setToolTip("Set LR and SI tilt back to 0 degrees.")
        b_reset_tilt.clicked.connect(self._match_reset_tilt)
        ctl.addWidget(b_reset_tilt)
        self.cb_mode = QtWidgets.QComboBox()
        self.cb_mode.addItems(["TV", "AV", "TV-AV"])
        self.cb_mode.currentTextChanged.connect(self._match_update_atlas)
        ctl.addWidget(self.cb_mode)
        v.addLayout(ctl)

        row = QtWidgets.QHBoxLayout()
        self.lbl_match = QtWidgets.QLabel("slice 0 / 0")
        row.addWidget(self.lbl_match)
        self.lbl_match_assigned = QtWidgets.QLabel("Unassigned")
        self.lbl_match_assigned.setObjectName("SectionHint")
        row.addWidget(self.lbl_match_assigned)
        self.lbl_match_autosave = QtWidgets.QLabel("Progress autosaves on Assign")
        self.lbl_match_autosave.setObjectName("SectionHint")
        row.addWidget(self.lbl_match_autosave)
        row.addStretch(1)
        b_assign = QtWidgets.QPushButton("Assign plane to slice")
        b_assign.clicked.connect(self._match_assign)
        b_save = QtWidgets.QPushButton("Save histology_ccf")
        b_save.clicked.connect(self._match_save)
        row.addWidget(b_assign)
        row.addWidget(b_save)
        v.addLayout(row)
        return page

    # ---- Align page ----
    def _build_align_page(self) -> QtWidgets.QWidget:
        page, v = self._section(
            "Align atlas to histology",
            "Click matching landmarks on histology then atlas (>= 3 pairs) for a manual "
            "affine, or use Auto-align. Save writes atlas2histology_tform.mat.",
        )
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.canvas_align_hist = ImageCanvas()
        self.canvas_align_atlas = ImageCanvas()
        self.canvas_align_hist.clicked.connect(self._align_click_hist)
        self.canvas_align_atlas.clicked.connect(self._align_click_atlas)
        split.addWidget(self._titled("Histology (click landmarks)", self._histology_contrast_pane("align", self.canvas_align_hist)))
        split.addWidget(self._titled("Atlas (click landmarks)", self.canvas_align_atlas))
        v.addWidget(split, 1)

        ctl = QtWidgets.QHBoxLayout()
        b_prev = QtWidgets.QPushButton("< Slice")
        b_prev.clicked.connect(lambda: self._align_step(-1))
        b_next = QtWidgets.QPushButton("Slice >")
        b_next.clicked.connect(lambda: self._align_step(1))
        b_clear = QtWidgets.QPushButton("Clear points")
        b_clear.clicked.connect(self._align_clear)
        b_auto = QtWidgets.QPushButton("Auto-align")
        b_auto.clicked.connect(self._align_auto)
        b_apply = QtWidgets.QPushButton("Apply points")
        b_apply.clicked.connect(self._align_apply_points)
        b_save = QtWidgets.QPushButton("Save tform")
        b_save.clicked.connect(self._align_save)
        for b in [b_prev, b_next, b_clear, b_auto, b_apply, b_save]:
            ctl.addWidget(b)
        ctl.addStretch(1)
        self.lbl_align = QtWidgets.QLabel("slice 0 / 0")
        ctl.addWidget(self.lbl_align)
        v.addLayout(ctl)
        return page

    # ---- Trace page ----
    def _build_trace_page(self) -> QtWidgets.QWidget:
        page, v = self._section(
            "Trace shank tracks",
            "Pick a shank number, create one editable line on each slice the shank "
            "crosses, then save probe_ccf.mat, CSV files, and the trajectory-area chart.",
        )
        body = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.canvas_trace = ImageCanvas()
        self.canvas_trace.clicked.connect(self._trace_click)
        body.addWidget(self._titled("Histology shank line", self._histology_contrast_pane("trace", self.canvas_trace)))

        trajectory_panel = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.trace_areas = pg.GraphicsLayoutWidget()
        self.trace_3d = Trajectory3DCanvas()
        trajectory_panel.addWidget(self._titled("Trajectory areas", self.trace_areas))
        trajectory_panel.addWidget(self._titled("3D trajectories", self.trace_3d))
        trajectory_panel.setSizes([360, 440])

        body.addWidget(trajectory_panel)
        body.setSizes([620, 800])
        v.addWidget(body, 1)

        ctl = QtWidgets.QHBoxLayout()
        b_prev = QtWidgets.QPushButton("< Slice")
        b_prev.clicked.connect(lambda: self._trace_step(-1))
        b_next = QtWidgets.QPushButton("Slice >")
        b_next.clicked.connect(lambda: self._trace_step(1))
        ctl.addWidget(b_prev)
        ctl.addWidget(b_next)
        ctl.addWidget(QtWidgets.QLabel("Shank"))
        self.sp_probe = QtWidgets.QSpinBox()
        self.sp_probe.setRange(1, 20)
        self.sp_probe.valueChanged.connect(self._trace_set_probe)
        ctl.addWidget(self.sp_probe)
        b_new = QtWidgets.QPushButton("New line")
        b_new.clicked.connect(self._trace_new_line)
        ctl.addWidget(b_new)
        b_clear = QtWidgets.QPushButton("Clear shank on slice")
        b_clear.clicked.connect(self._trace_clear)
        ctl.addWidget(b_clear)
        self.btn_trace_build = QtWidgets.QPushButton("Build + Save probe_ccf")
        self.btn_trace_build.clicked.connect(self._trace_build)
        ctl.addWidget(self.btn_trace_build)
        ctl.addStretch(1)
        self.lbl_trace = QtWidgets.QLabel("slice 0 / 0")
        ctl.addWidget(self.lbl_trace)
        v.addLayout(ctl)

        coords = QtWidgets.QHBoxLayout()
        coords.addWidget(QtWidgets.QLabel("Line endpoints"))
        for key, label in (("x1", "x1"), ("y1", "y1"), ("x2", "x2"), ("y2", "y2")):
            coords.addWidget(QtWidgets.QLabel(label))
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(-1_000_000.0, 1_000_000.0)
            spin.setDecimals(1)
            spin.setSingleStep(1.0)
            spin.setKeyboardTracking(False)
            spin.setFixedWidth(92)
            spin.valueChanged.connect(self._trace_controls_changed)
            self._trace_coord_spins[key] = spin
            coords.addWidget(spin)
        coords.addStretch(1)
        self.lbl_trace_line = QtWidgets.QLabel("")
        self.lbl_trace_line.setObjectName("SectionHint")
        coords.addWidget(self.lbl_trace_line)
        v.addLayout(coords)
        return page

    # ---- Channel map page ----
    def _build_channels_page(self) -> QtWidgets.QWidget:
        page, v = self._section(
            "Per-channel region map",
            "Prepare the ALF, xyz_picks, and per-channel regions for the IBL "
            "Electrophysiology Atlas. RMS/QC maps are optional because they stream "
            "the whole raw binary and are much slower.",
        )
        order = QtWidgets.QLabel(
            "1 Prepare for IBL -> optional RMS/QC maps -> 2 Review unit distribution -> "
            "3 Propose alignment -> 4 Refine in IBL GUI -> 5 Finalize regions."
        )
        order.setObjectName("SectionHint")
        order.setWordWrap(True)
        v.addWidget(order)
        self.cb_alignment = QtWidgets.QComboBox()
        self.cb_alignment.addItems(["original", "latest"])
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("5 Finalize alignment source"))
        row.addWidget(self.cb_alignment)
        row.addStretch(1)
        v.addLayout(row)

        btns = QtWidgets.QHBoxLayout()
        b_prepare = QtWidgets.QPushButton("1 Prepare for IBL")
        b_prepare.setToolTip(
            "Run fast ALF extraction, xyz_picks generation, and the initial channel map. "
            "Skips slow RMS/QC maps."
        )
        b_prepare.clicked.connect(self._prepare_for_ibl)
        b_rms = QtWidgets.QPushButton("Optional RMS/QC maps")
        b_rms.setProperty("role", "secondary")
        b_rms.setToolTip(
            "Slow: streams the complete raw AP/LF binary. Only needed for RMS/QC panels "
            "inside the IBL GUI."
        )
        b_rms.clicked.connect(self._compute_rms_qc_maps)
        b_units = QtWidgets.QPushButton("2 Plot unit distribution")
        b_units.clicked.connect(self._plot_unit_distribution)
        b_prop = QtWidgets.QPushButton("3 Propose alignment (auto)")
        b_prop.clicked.connect(self._propose_alignment)
        b_final = QtWidgets.QPushButton("5 Finalize regions")
        b_final.setToolTip("Regenerate channel_locations_shankN.json from the latest "
                           "alignment you saved in the IBL GUI (per shank).")
        b_final.clicked.connect(self._finalize_channels)
        for b in [b_prepare, b_rms, b_units, b_prop, b_final]:
            btns.addWidget(b)
        btns.addStretch(1)
        v.addLayout(btns)

        split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.units_plot = pg.GraphicsLayoutWidget()
        self.units_plot.setBackground("w")
        split.addWidget(self._titled("Unit distribution across regions", self.units_plot))
        self.tbl_channels = QtWidgets.QTableWidget(0, 4)
        self.tbl_channels.setHorizontalHeaderLabels(["channel", "axial", "lateral", "region"])
        self.tbl_channels.horizontalHeader().setStretchLastSection(True)
        split.addWidget(self._titled("Per-channel region table", self.tbl_channels))
        split.setSizes([520, 220])
        v.addWidget(split, 1)
        return page

    # ---- IBL page ----
    def _build_ibl_page(self) -> QtWidgets.QWidget:
        page, v = self._section(
            "IBL ephys alignment (optional)",
            "Launch the original IBL alignment GUI to refine the probe-to-brain mapping "
            "against electrophysiology features. After saving in that GUI, regenerate "
            "the channel map with the 'latest' alignment.",
        )
        btns = QtWidgets.QHBoxLayout()
        b_launch = QtWidgets.QPushButton("4 Launch IBL alignment GUI")
        b_launch.clicked.connect(self._launch_ibl)
        b_refresh = QtWidgets.QPushButton("5 Regenerate channel map (latest)")
        b_refresh.clicked.connect(lambda: self._gen_channels(alignment_override="latest"))
        btns.addWidget(b_launch)
        btns.addWidget(b_refresh)
        btns.addStretch(1)
        v.addLayout(btns)
        note = QtWidgets.QLabel(
            "The IBL GUI opens its own window (offline mode). Select the histology "
            "folder there, refine, and save. This step is entirely optional."
        )
        note.setObjectName("SectionHint")
        note.setWordWrap(True)
        v.addWidget(note)
        v.addStretch(1)
        return page

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _with_button(edit: QtWidgets.QLineEdit, button: QtWidgets.QPushButton) -> QtWidgets.QWidget:
        host = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(host)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(edit, 1)
        h.addWidget(button, 0)
        return host

    @staticmethod
    def _titled(title: str, widget: QtWidgets.QWidget) -> QtWidgets.QWidget:
        host = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(host)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(4)
        lbl = QtWidgets.QLabel(title)
        lbl.setObjectName("SectionHint")
        v.addWidget(lbl)
        v.addWidget(widget, 1)
        return host

    @staticmethod
    def _thumbnail_pixmap(image: np.ndarray, size: QtCore.QSize) -> QtGui.QPixmap:
        arr = np.asarray(image)
        if arr.ndim == 2:
            v = np.nan_to_num(arr.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
            finite = v[np.isfinite(v)]
            lo, hi = np.percentile(finite, [1.0, 99.0]) if finite.size else (0.0, 1.0)
            g = np.clip((v - lo) / max(float(hi - lo), 1e-9), 0, 1)
            rgb = np.dstack([g, g, g])
        else:
            rgb = arr[..., :3].astype(float)
            if rgb.max(initial=0) > 1.5:
                rgb /= 255.0
            rgb = np.clip(rgb, 0, 1)
        rgb8 = np.ascontiguousarray((rgb * 255).astype(np.uint8))
        h, w = rgb8.shape[:2]
        qimg = QtGui.QImage(rgb8.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy()
        pix = QtGui.QPixmap.fromImage(qimg).scaled(
            size,
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        return pix

    @staticmethod
    def _selection_icon_from_pixmap(base: QtGui.QPixmap, selected: bool) -> QtGui.QIcon:
        if base.isNull():
            return QtGui.QIcon()
        pix = QtGui.QPixmap(base.size())
        pix.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(pix)
        if not selected:
            painter.setOpacity(0.58)
        painter.drawPixmap(0, 0, base)
        painter.setOpacity(1.0)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)

        if selected:
            width, height = pix.width(), pix.height()
            short_side = max(1, min(width, height))
            margin = max(5, int(short_side * 0.06))
            badge = max(28, int(short_side * 0.30))
            rect = QtCore.QRect(width - badge - margin, margin, badge, badge)

            painter.setPen(QtGui.QPen(QtGui.QColor("#ffffff"), 2))
            painter.setBrush(QtGui.QColor("#0f9d58"))
            painter.drawEllipse(rect)

            check = QtGui.QPainterPath()
            check.moveTo(rect.left() + badge * 0.26, rect.top() + badge * 0.54)
            check.lineTo(rect.left() + badge * 0.43, rect.top() + badge * 0.70)
            check.lineTo(rect.left() + badge * 0.76, rect.top() + badge * 0.34)
            painter.setPen(QtGui.QPen(
                QtGui.QColor("#ffffff"),
                max(3, int(badge * 0.12)),
                QtCore.Qt.SolidLine,
                QtCore.Qt.RoundCap,
                QtCore.Qt.RoundJoin,
            ))
            painter.drawPath(check)

            painter.setBrush(QtCore.Qt.NoBrush)
            painter.setPen(QtGui.QPen(QtGui.QColor("#0f9d58"), 4))
            painter.drawRoundedRect(pix.rect().adjusted(2, 2, -2, -2), 8, 8)

        painter.end()
        return QtGui.QIcon(pix)

    @staticmethod
    def _thumbnail_icon(image: np.ndarray, size: QtCore.QSize) -> QtGui.QIcon:
        return HistologyTab._selection_icon_from_pixmap(
            HistologyTab._thumbnail_pixmap(image, size),
            selected=False,
        )

    def _log(self, msg: str) -> None:
        self.log.appendPlainText(str(msg))

    def is_busy(self) -> bool:
        """True while any background histology worker is still running."""
        return self._busy_count > 0

    def set_plot_preferences(self, theme: str, show_grid: bool) -> None:
        """Apply the app-wide light/dark theme to every canvas in this tab."""
        self._plot_theme = "Dark" if str(theme).lower().startswith("dark") else "Light"
        bg = "#0b0f14" if self._plot_theme == "Dark" else "#ffffff"
        for c in [getattr(self, n, None) for n in (
            "canvas_select_raw", "canvas_preproc", "canvas_match_hist", "canvas_match_atlas",
            "canvas_align_hist", "canvas_align_atlas", "canvas_trace", "trace_areas",
        )]:
            if c is not None:
                c.setBackground(bg)
        trace_3d = getattr(self, "trace_3d", None)
        if trace_3d is not None:
            trace_3d.set_theme(self._plot_theme)

    def _run_bg(self, fn, on_done, *args, busy_msg: str = "", on_finished=None) -> None:
        if busy_msg:
            self._log(busy_msg)
        self._busy_count += 1
        worker = FunctionWorker(fn, *args)

        def _finished(payload: dict) -> None:
            self._busy_count = max(0, self._busy_count - 1)
            try:
                if payload.get("ok"):
                    on_done(payload.get("result"))
                else:
                    self._log("Task failed.")
            finally:
                if on_finished is not None:
                    on_finished(payload)

        worker.signals.finished.connect(_finished)
        worker.signals.error.connect(lambda m: self._log(f"Error: {m}"))
        worker.signals.log.connect(self._log)
        self.pool.start(worker)

    def _ensure_atlas(self) -> Optional[hatlas.AllenCCFAtlas]:
        if self.atlas is not None:
            return self.atlas
        path = self.ed_atlas.text().strip() or None
        if not hatlas.atlas_files_present(path):
            self._log("Atlas files not found. Download from https://osf.io/fv7ed/overview "
                      "and set the atlas folder.")
            return None
        self._log("Loading Allen CCF atlas (first use)...")
        self.atlas = hatlas.AllenCCFAtlas(path)
        self._log(f"Atlas loaded: shape {self.atlas.shape}.")
        return self.atlas

    # ------------------------------------------------ Image selection actions
    def _resolve_raw_folder_for_selection(self) -> Tuple[Optional[Path], List[Path], Optional[Path]]:
        """Return the raw folder and images to show, repairing stale saved paths.

        The raw-image textbox is persisted globally. When the user switches from
        one run to another, that saved path can still point at a different
        session. Prefer it only when it actually contains images; otherwise fall
        back to the loaded session's ``raw`` folder.
        """
        typed = Path(self.ed_raw.text().strip()) if self.ed_raw.text().strip() else None
        candidates: list[Path] = []
        if typed is not None:
            candidates.append(typed)
        if self.folder is not None:
            session_raw = self.folder / "raw"
            if all(session_raw != p for p in candidates):
                candidates.append(session_raw)

        for folder in candidates:
            paths = slice_prep.list_raw_images(folder)
            if paths:
                repaired_from = typed if typed is not None and folder != typed else None
                return folder, paths, repaired_from
        return typed, [], None

    def _selection_refresh(self) -> None:
        if not hasattr(self, "lst_raw_images"):
            return
        raw_folder, self.raw_image_paths, repaired_from = self._resolve_raw_folder_for_selection()
        if raw_folder is not None and self.ed_raw.text().strip() != str(raw_folder):
            self.ed_raw.setText(str(raw_folder))
        if repaired_from is not None:
            self._log(f"Raw image path had no images; using session raw folder: {raw_folder}")
        previous = {str(p) for p in self.selected_raw_paths}
        if not previous:
            previous = {str(p) for p in self.raw_image_paths}
        self._selection_grid_updating = True
        try:
            self.lst_raw_images.clear()
            icon_size = self.lst_raw_images.iconSize()
            for path in self.raw_image_paths:
                item = QtWidgets.QListWidgetItem(path.name)
                item.setData(self._SELECTION_PATH_ROLE, str(path))
                item.setFlags(item.flags() & ~QtCore.Qt.ItemIsUserCheckable)
                selected = str(path) in previous
                try:
                    pix = self._thumbnail_pixmap(slice_prep.load_image(path), icon_size)
                    item.setData(self._SELECTION_PIXMAP_ROLE, pix)
                    self._selection_set_item_selected(item, selected)
                except Exception:
                    item.setIcon(QtGui.QIcon())
                    item.setToolTip(f"Could not preview {path.name}")
                    item.setData(self._SELECTION_SELECTED_ROLE, selected)
                self.lst_raw_images.addItem(item)
            if self.raw_image_paths:
                self.lst_raw_images.setCurrentRow(0)
        finally:
            self._selection_grid_updating = False
        self._selection_item_changed()
        if not self.raw_image_paths and hasattr(self, "lbl_image_selection"):
            where = str(raw_folder) if raw_folder is not None else "no folder set"
            self.lbl_image_selection.setText(f"No raw TIFF/PNG images found ({where})")
            self._selection_clear_preview(where)
        else:
            self._selection_show_preview(self.lst_raw_images.currentItem())

    def _selection_set_item_selected(self, item: QtWidgets.QListWidgetItem, selected: bool) -> None:
        item.setData(self._SELECTION_SELECTED_ROLE, bool(selected))
        pix = item.data(self._SELECTION_PIXMAP_ROLE)
        if isinstance(pix, QtGui.QPixmap):
            item.setIcon(self._selection_icon_from_pixmap(pix, selected))
        item.setToolTip(
            "Selected for preprocessing. Click to remove."
            if selected else "Ignored during preprocessing. Click to include."
        )
        font = item.font()
        font.setBold(bool(selected))
        item.setFont(font)
        item.setForeground(QtGui.QBrush(
            QtGui.QColor("#064e3b") if selected else QtGui.QColor("#6b7280")
        ))

    def _selection_toggle_item(self, item: QtWidgets.QListWidgetItem) -> None:
        if self._selection_grid_updating:
            return
        self.lst_raw_images.setCurrentItem(item)
        self._selection_set_item_selected(item, not bool(item.data(self._SELECTION_SELECTED_ROLE)))
        self._selection_item_changed()
        self._selection_show_preview(item)

    def _selection_clear_preview(self, message: str = "") -> None:
        self._selection_preview_image = None
        if hasattr(self, "canvas_select_raw"):
            self.canvas_select_raw.set_image(None)
        if hasattr(self, "lbl_select_preview"):
            self.lbl_select_preview.setText(message or "No raw image preview.")

    def _selection_show_preview(self, item: Optional[QtWidgets.QListWidgetItem]) -> None:
        if item is None or not hasattr(self, "canvas_select_raw"):
            self._selection_clear_preview()
            return
        path = Path(str(item.data(self._SELECTION_PATH_ROLE)))
        try:
            image = slice_prep.load_image(path)
        except Exception as exc:
            self._selection_clear_preview(f"Could not preview {path.name}: {exc}")
            return
        self._selection_preview_image = image
        idx = max(0, self.lst_raw_images.row(item))
        self._show_histology_slice("select", self.canvas_select_raw, idx, image)
        if hasattr(self, "lbl_select_preview"):
            state = "selected" if bool(item.data(self._SELECTION_SELECTED_ROLE)) else "ignored"
            self.lbl_select_preview.setText(f"{path.name} ({state})")

    def _selection_item_changed(self, *_args) -> None:
        if self._selection_grid_updating or not hasattr(self, "lst_raw_images"):
            return
        selected: list[Path] = []
        for i in range(self.lst_raw_images.count()):
            item = self.lst_raw_images.item(i)
            if bool(item.data(self._SELECTION_SELECTED_ROLE)):
                selected.append(Path(str(item.data(self._SELECTION_PATH_ROLE))))
        self.selected_raw_paths = selected
        if self.validated_raw_paths and [str(p) for p in self.validated_raw_paths] != [str(p) for p in selected]:
            self.validated_raw_paths = []
        total = len(self.raw_image_paths)
        if hasattr(self, "lbl_image_selection"):
            suffix = "validated" if self.validated_raw_paths else "not validated"
            self.lbl_image_selection.setText(f"{len(selected)} / {total} selected, {suffix}")

    def _selection_validate(self) -> None:
        """Commit selected raw images and load exactly those images into Preprocess."""
        self._selection_item_changed()
        if not self.selected_raw_paths:
            self._log("No raw images selected. Select at least one image before validating.")
            return
        self.validated_raw_paths = list(self.selected_raw_paths)
        if hasattr(self, "lbl_image_selection"):
            self.lbl_image_selection.setText(
                f"{len(self.validated_raw_paths)} / {len(self.raw_image_paths)} selected, validated"
            )
        self._log(f"Validated {len(self.validated_raw_paths)} raw image(s) for preprocessing.")
        self.nav.setCurrentIndex(getattr(self, "_page_preprocess", self.nav.currentIndex()))
        self._preproc_load(paths=list(self.validated_raw_paths), require_validated=False)

    def _selection_set_all(self, checked: bool) -> None:
        if not hasattr(self, "lst_raw_images"):
            return
        self._selection_grid_updating = True
        try:
            for i in range(self.lst_raw_images.count()):
                self._selection_set_item_selected(self.lst_raw_images.item(i), checked)
        finally:
            self._selection_grid_updating = False
        self._selection_item_changed()
        self._selection_show_preview(self.lst_raw_images.currentItem())

    def _selection_invert(self) -> None:
        if not hasattr(self, "lst_raw_images"):
            return
        self._selection_grid_updating = True
        try:
            for i in range(self.lst_raw_images.count()):
                item = self.lst_raw_images.item(i)
                self._selection_set_item_selected(item, not bool(item.data(self._SELECTION_SELECTED_ROLE)))
        finally:
            self._selection_grid_updating = False
        self._selection_item_changed()
        self._selection_show_preview(self.lst_raw_images.currentItem())

    def _selected_raw_image_paths(self) -> List[Path]:
        if self.validated_raw_paths:
            return list(self.validated_raw_paths)
        if not self.raw_image_paths and self.ed_raw.text().strip():
            self._selection_refresh()
        if hasattr(self, "lst_raw_images") and self.lst_raw_images.count() > 0:
            return list(self.selected_raw_paths)
        raw = self.ed_raw.text().strip()
        return slice_prep.list_raw_images(raw) if raw else []

    # ----------------------------------------------------- Setup actions
    def _pick_into(self, edit: QtWidgets.QLineEdit, title: str) -> None:
        start = edit.text().strip() or str(self.settings.value("paths/last_folder", str(Path.cwd())))
        d = QtWidgets.QFileDialog.getExistingDirectory(self, title, start)
        if d:
            edit.setText(d)
            if edit is getattr(self, "ed_raw", None):
                self.raw_image_paths = []
                self.selected_raw_paths = []
                self.validated_raw_paths = []
                self._selection_refresh()

    def _pick_folder(self) -> None:
        self._pick_into(self.ed_folder, "Histology folder")
        if self.ed_folder.text().strip():
            self._load_session()

    def open_histology_folder(self, folder: str) -> None:
        """Point the tab at ``folder`` and load any existing session products."""
        self.ed_folder.setText(str(folder))
        self._load_session()

    @staticmethod
    def _derive_histology_folder(ks_folder: Path) -> Path:
        """Session histology folder for a Kilosort output path.

        Mirrors the lab layout ``<session>/spike_sorting/.../imecN_ks4``: the
        histology folder lives at ``<session>/histology``. Falls back to a
        ``histology`` folder next to the ephys/imec folder.
        """
        for parent in ks_folder.parents:
            if parent.name.lower() == "spike_sorting":
                return parent.parent / "histology"
        return ks_folder.parent / "histology"

    @staticmethod
    def _find_ephys_folder(ks_folder: Path) -> Path:
        """Folder holding the raw ephys (``*.ap.bin``/``*.ap.meta``) for this run."""
        for cand in (ks_folder, ks_folder.parent, ks_folder.parent.parent):
            if any(cand.glob("*.ap.bin")) or any(cand.glob("*.ap.meta")):
                return cand
        return ks_folder.parent

    def setup_from_ks_folder(self, ks_folder: str) -> None:
        """Auto-configure a histology session from a completed spike-sorting run.

        Fills the Kilosort, ephys and histology folders (creating the histology
        folder if needed) and loads the session, so the user only needs to point
        at their raw histology images and proceed.
        """
        ks = Path(str(ks_folder))
        if not ks.exists():
            self._log(f"Kilosort folder not found: {ks}")
            return
        hist = self._derive_histology_folder(ks)
        ephys = self._find_ephys_folder(ks)
        hist.mkdir(parents=True, exist_ok=True)

        self.ed_ks.setText(str(ks))
        self.ed_ephys.setText(str(ephys))
        self.ed_folder.setText(str(hist))
        if not self.ed_atlas.text().strip():
            self.ed_atlas.setText(hatlas.DEFAULT_ATLAS_PATH)
        if not self.ed_iblapps.text().strip():
            self.ed_iblapps.setText(ibl_launch.DEFAULT_IBLAPPS_PATH)
        self._persist_settings()

        self._log("Histology session set up from sorted run:")
        self._log(f"  Kilosort: {ks}")
        self._log(f"  Ephys:    {ephys}")
        self._log(f"  Histology: {hist}")
        self._load_session()
        self.nav.setCurrentIndex(0)
        if not self.ed_raw.text().strip():
            self._log("Next: point 'Raw images' at your histology scans, then Select images.")

    def _load_session(self) -> None:
        text = self.ed_folder.text().strip()
        if not text:
            return
        self.folder = Path(text)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.settings.setValue("histology/last_folder", text)
        self.validated_raw_paths = []
        self.slice_images = []
        self.slice_pixel_um = []
        self.histology_ccf = []
        self.tforms = []
        # Load any slice images already saved.
        for p in slice_prep.list_saved_slices(self.folder):
            try:
                self.slice_images.append(slice_prep.load_image(p))
            except Exception as exc:
                self._log(f"Could not load slice image {p.name}: {exc}")
        self.slice_pixel_um = self._infer_saved_slice_pixel_sizes()
        # Existing products.
        hccf = self.folder / "histology_ccf.mat"
        if hccf.exists():
            try:
                self.histology_ccf = io_formats.load_histology_ccf(hccf)
            except Exception as exc:
                self._log(f"Could not load histology_ccf.mat: {exc}")
        tfn = self.folder / "atlas2histology_tform.mat"
        if tfn.exists():
            try:
                self.tforms = io_formats.load_tforms(tfn)
            except Exception as exc:
                self._log(f"Could not load tforms: {exc}")
        self.slice_specs = [None] * max(len(self.slice_images), len(self.histology_ccf))
        self._load_match_specs()  # restore saved atlas matching for this run
        self.probe_points = {}
        self._load_probe_lines()  # restore drawn probe tracks for this run
        try:
            self.trace_areas.clear()
            self.trace_3d.clear()
        except Exception:
            pass
        self._refresh_status()
        self._selection_refresh()
        self._preproc_show()
        self._match_show()
        self._align_show()
        self._trace_show()
        self._log(f"Loaded session: {self.folder}")

    def _raw_image_paths_for_session(self) -> List[Path]:
        candidates: list[Path] = []
        if self.folder is not None:
            candidates.append(self.folder / "raw")
        text = self.ed_raw.text().strip() if hasattr(self, "ed_raw") else ""
        if text:
            candidates.append(Path(text))
        for folder in candidates:
            try:
                paths = slice_prep.list_raw_images(folder)
            except Exception:
                paths = []
            if paths:
                return paths
        return []

    def _infer_saved_slice_pixel_sizes(self) -> List[Optional[float]]:
        """Infer saved ``slice_N`` pixel sizes from raw-image sidecar metadata."""
        out: list[Optional[float]] = [None] * len(self.slice_images)
        raw_paths = self.selected_raw_paths or self._raw_image_paths_for_session()
        if not raw_paths:
            return out
        for i, raw_path in enumerate(raw_paths[: len(out)]):
            raw_px = slice_prep.pixel_size_um(raw_path)
            if raw_px is None:
                continue
            try:
                raw_img = slice_prep.load_image(raw_path)
                saved_img = self.slice_images[i]
                raw_area = max(1.0, float(raw_img.shape[0] * raw_img.shape[1]))
                saved_area = max(1.0, float(saved_img.shape[0] * saved_img.shape[1]))
                downsample = float(np.sqrt(raw_area / saved_area))
            except Exception:
                downsample = 1.0
            if np.isfinite(downsample) and downsample > 0:
                out[i] = float(raw_px * downsample)
            else:
                out[i] = float(raw_px)
        return out

    def _atlas_to_histology_scale_for_slice(self, idx: int) -> float:
        """Atlas pixel/voxel size converted into current histology image pixels."""
        if 0 <= idx < len(self.slice_pixel_um):
            px = self.slice_pixel_um[idx]
            if px is not None and np.isfinite(px) and px > 0:
                return float(hatlas.CCF_VOXEL_UM / px)
        try:
            return 1.0 / max(1.0, float(self.sp_downsample.value()))
        except Exception:
            return 1.0

    def _refresh_status(self) -> None:
        if self.folder is None:
            self.lbl_status.setText("No session loaded.")
            return
        def mark(name: str) -> str:
            ok = (self.folder / name).exists()
            color = "#2e7d32" if ok else "#9aa3af"
            return f"<span style='color:{color}'>{'YES' if ok else 'no '}</span> {name}"
        items = [
            f"Slice images: {len(self.slice_images)}",
            mark("histology_ccf.mat"),
            mark("atlas2histology_tform.mat"),
            mark("probe_ccf.mat"),
            mark("channel_locations_all_shanks.json"),
        ]
        self.lbl_status.setText("<br>".join(items))

    # ------------------------------------------------ Preprocess actions
    def _preproc_load(
        self,
        paths: Optional[List[Path]] = None,
        *,
        require_validated: bool = False,
    ) -> None:
        raw = self.ed_raw.text().strip()
        if not raw:
            self._log("Set a raw image folder first.")
            return
        factor = float(self.sp_downsample.value())
        paths = list(paths) if paths is not None else self._selected_raw_image_paths()
        if not paths:
            self._log(f"No raw images selected in {raw}. Select images first, or use Select all.")
            return
        if require_validated and not self.validated_raw_paths:
            self._log("Validate the raw image selection first.")
            return

        def job():
            imgs = []
            pixels = []
            for p in paths:
                im = slice_prep.load_image(p)
                px = slice_prep.pixel_size_um(p)
                if factor != 1:
                    im = slice_prep.downsample(im, factor)
                    if px is not None:
                        px *= factor
                imgs.append(im)
                pixels.append(px)
            return imgs, pixels

        def done(result):
            imgs, pixels = result
            self.slice_images = imgs
            self.slice_pixel_um = list(pixels)
            self._cur_preproc = 0
            self._sync_slice_state_after_preprocess()
            n_meta = sum(px is not None for px in self.slice_pixel_um)
            suffix = f" ({n_meta} with pixel-size metadata)" if n_meta else ""
            self._log(f"Loaded {len(imgs)} selected raw image(s){suffix}.")
            self._preproc_show()

        self._run_bg(job, done, busy_msg="Loading raw images...")

    def _preproc_save(self) -> None:
        if not self.slice_images or self.folder is None:
            self._log("Nothing to save (load images and set a folder).")
            return
        out = slice_prep.save_slices(self.slice_images, self.folder)
        out_set = {p.resolve() for p in out}
        removed = 0
        for stale in slice_prep.list_saved_slices(self.folder):
            if stale.suffix.lower() not in {".tif", ".tiff"}:
                continue
            try:
                stale_resolved = stale.resolve()
            except OSError:
                stale_resolved = stale
            if stale_resolved not in out_set:
                try:
                    stale.unlink()
                    removed += 1
                except OSError as exc:
                    self._log(f"Could not remove stale slice image {stale.name}: {exc}")
        self._sync_slice_state_after_preprocess()
        suffix = f" Removed {removed} stale slice image(s)." if removed else ""
        self._log(f"Saved {len(out)} slice image(s) to {self.folder}.{suffix}")
        self._refresh_status()

    def _sync_slice_state_after_preprocess(self) -> None:
        """Keep match/align/trace state valid after slice images are loaded or replaced."""
        n = len(self.slice_images)
        if n <= 0:
            self._cur_match_slice = 0
            self._cur_align_slice = 0
            self._cur_trace_slice = 0
            self.slice_specs = []
            self.slice_pixel_um = []
        else:
            self._cur_match_slice = int(np.clip(self._cur_match_slice, 0, n - 1))
            self._cur_align_slice = int(np.clip(self._cur_align_slice, 0, n - 1))
            self._cur_trace_slice = int(np.clip(self._cur_trace_slice, 0, n - 1))
            if len(self.slice_specs) < n:
                self.slice_specs.extend([None] * (n - len(self.slice_specs)))
            elif len(self.slice_specs) > n:
                self.slice_specs = self.slice_specs[:n]
            if len(self.slice_pixel_um) < n:
                self.slice_pixel_um.extend([None] * (n - len(self.slice_pixel_um)))
            elif len(self.slice_pixel_um) > n:
                self.slice_pixel_um = self.slice_pixel_um[:n]
        current = self.nav.currentIndex() if hasattr(self, "nav") else -1
        if current == getattr(self, "_page_match", -1):
            self._match_show()
        elif current == getattr(self, "_page_align", -1):
            self._align_show()
        elif current == getattr(self, "_page_trace", -1):
            self._trace_show()

    def _preproc_step(self, d: int) -> None:
        if not self.slice_images:
            return
        self._cur_preproc = int(np.clip(getattr(self, "_cur_preproc", 0) + d, 0, len(self.slice_images) - 1))
        self._preproc_show()

    def _preproc_show(self) -> None:
        idx = getattr(self, "_cur_preproc", 0)
        if not self.slice_images:
            self.canvas_preproc.set_image(None)
            self.lbl_preproc.setText("0 / 0")
            return
        idx = min(idx, len(self.slice_images) - 1)
        self._show_histology_slice("preproc", self.canvas_preproc, idx, self.slice_images[idx])
        self.lbl_preproc.setText(f"{idx + 1} / {len(self.slice_images)}")

    def _preproc_rotate(self, angle: float) -> None:
        idx = getattr(self, "_cur_preproc", 0)
        if not self.slice_images:
            return
        self.slice_images[idx] = slice_prep.rotate_center(self.slice_images[idx], angle)
        self._preproc_show()

    def _preproc_flip(self, horizontal: bool) -> None:
        idx = getattr(self, "_cur_preproc", 0)
        if not self.slice_images:
            return
        self.slice_images[idx] = slice_prep.flip(self.slice_images[idx], horizontal)
        self._preproc_show()

    # ------------------------------------------------------ Match actions
    def _match_step(self, d: int) -> None:
        if not self.slice_images:
            return
        self._cur_match_slice = int(np.clip(self._cur_match_slice + d, 0, len(self.slice_images) - 1))
        self._match_show()

    def _match_show(self) -> None:
        if not self.slice_images:
            self.canvas_match_hist.set_image(None)
            self.canvas_match_hist.clear_overlays()
            self.lbl_match.setText("slice 0 / 0")
            self.lbl_match_assigned.setText("Unassigned")
            self.lbl_match_assigned.setStyleSheet("")
            return
        idx = min(self._cur_match_slice, len(self.slice_images) - 1)
        self._show_histology_slice("match", self.canvas_match_hist, idx, self.slice_images[idx])
        self.canvas_match_hist.clear_overlays()
        self.lbl_match.setText(f"slice {idx + 1} / {len(self.slice_images)}")
        self._match_update_assignment_status(idx)
        self._restore_match_sliders(idx)
        self._match_update_atlas()

    def _match_is_assigned(self, idx: int) -> bool:
        return idx < len(self.slice_specs) and self.slice_specs[idx] is not None

    def _match_update_assignment_status(self, idx: int) -> None:
        assigned = self._match_is_assigned(idx)
        total = len(self.slice_images)
        n_assigned = sum(s is not None for s in self.slice_specs[:total])
        if assigned:
            self.lbl_match_assigned.setText(f"Assigned ({n_assigned}/{total})")
            self.lbl_match_assigned.setStyleSheet("color: #15803d; font-weight: 700;")
            self._match_draw_assigned_badge(idx)
        else:
            self.lbl_match_assigned.setText(f"Unassigned ({n_assigned}/{total})")
            self.lbl_match_assigned.setStyleSheet("color: #b45309; font-weight: 700;")

    def _match_draw_assigned_badge(self, idx: int) -> None:
        if idx >= len(self.slice_images):
            return
        arr = np.asarray(self.slice_images[idx])
        if arr.ndim < 2:
            return
        pad = max(4.0, min(float(arr.shape[0]), float(arr.shape[1])) * 0.025)
        self.canvas_match_hist.add_text("ASSIGNED", pad, pad, color="w")

    def _match_histogram_levels_changed(self) -> None:
        """Remember manually-adjusted histology contrast levels per slice."""
        self._histology_levels_changed("match")

    def _match_apply_histogram_levels(self, lo: float, hi: float) -> None:
        """Apply contrast levels to both the image and linked histogram widget."""
        self._apply_histology_levels("match", lo, hi)

    def _match_reset_histogram(self) -> None:
        """Reset the histology image contrast control to the current slice range."""
        self._reset_histology_levels("match")

    def _restore_match_sliders(self, idx: int) -> None:
        """Reflect a previously-saved plane for slice ``idx`` in the sliders."""
        spec = self.slice_specs[idx] if idx < len(self.slice_specs) else None
        if not spec or "ap" not in spec:
            return
        widgets = (self.sl_ap, self.sl_lr, self.sl_si, self.cb_mode)
        for w in widgets:
            w.blockSignals(True)
        try:
            self.sl_ap.setValue(int(spec.get("ap", self.sl_ap.value())))
            self.sl_lr.setValue(int(spec.get("lr", self.sl_lr.value())))
            self.sl_si.setValue(int(spec.get("si", self.sl_si.value())))
            mi = self.cb_mode.findText(str(spec.get("mode", "")))
            if mi >= 0:
                self.cb_mode.setCurrentIndex(mi)
        finally:
            for w in widgets:
                w.blockSignals(False)

    def _match_update_atlas(self) -> None:
        self._match_update_slider_labels()
        at = self._ensure_atlas()
        if at is None:
            return
        cv = hatlas.coronal_camera_vector(self.sl_lr.value(), self.sl_si.value())
        sp = hatlas.coronal_slice_point(self.sl_ap.value(), at)
        sl = at.grab_atlas_slice(sp, cv, spacing=3)
        rgb = matching.render_atlas_slice(sl, at, self.cb_mode.currentText())
        self.canvas_match_atlas.set_image(rgb)

    def _match_update_slider_labels(self) -> None:
        if hasattr(self, "lbl_ap_value"):
            self.lbl_ap_value.setText(f"{int(self.sl_ap.value())} um")
        if hasattr(self, "lbl_lr_value"):
            self.lbl_lr_value.setText(f"{int(self.sl_lr.value()):+d} deg")
        if hasattr(self, "lbl_si_value"):
            self.lbl_si_value.setText(f"{int(self.sl_si.value()):+d} deg")

    def _match_reset_tilt(self) -> None:
        widgets = (self.sl_lr, self.sl_si)
        for w in widgets:
            w.blockSignals(True)
        try:
            self.sl_lr.setValue(0)
            self.sl_si.setValue(0)
        finally:
            for w in widgets:
                w.blockSignals(False)
        self._match_update_atlas()

    def _match_auto(self) -> None:
        at = self._ensure_atlas()
        if at is None or not self.slice_images:
            return
        idx = min(self._cur_match_slice, len(self.slice_images) - 1)
        image = np.asarray(self.slice_images[idx])
        center_ap = int(self.sl_ap.value())
        if center_ap <= self.sl_ap.minimum() + 5 or center_ap >= self.sl_ap.maximum() - 5:
            center_ap = 540
        atlas_to_histology_scale = self._atlas_to_histology_scale_for_slice(idx)

        def job():
            return matching.automatch_coronal_ap(
                image,
                at,
                center_ap=center_ap,
                search_radius=280,
                atlas_to_histology_scale=atlas_to_histology_scale,
            )

        def done(result):
            if not isinstance(result, dict) or "ap" not in result:
                self._log("Auto match did not return an AP estimate.")
                return
            ap = int(np.clip(int(result["ap"]), self.sl_ap.minimum(), self.sl_ap.maximum()))
            self.sl_ap.setValue(ap)
            self.sl_lr.setValue(0)
            self.sl_si.setValue(0)
            self._match_update_atlas()
            if self.ck_match_auto_adjust.isChecked():
                self._match_assign_current_plane(log=False)
            score = float(result.get("score", 0.0))
            confidence = float(result.get("confidence", 0.0))
            self.lbl_match_autosave.setText(
                f"Auto matched AP {ap} um (score {score:.2f}, conf {confidence:.2f})"
            )
            top = result.get("top") or []
            engine = result.get("engine") or {}
            engine_txt = (
                f"; workers {int(engine.get('workers', 1))}"
                f"; numba {'yes' if engine.get('numba') else 'no'}"
                f"; cuda {'yes' if engine.get('cuda') else 'no'}"
            )
            if top:
                bests = ", ".join(f"{int(t['ap'])}:{float(t['score']):.2f}" for t in top[:3])
                self._log(
                    f"Auto match slice {idx + 1}: AP {ap} "
                    f"(scale {atlas_to_histology_scale:.3f}{engine_txt}; top {bests})."
                )
            else:
                self._log(
                    f"Auto match slice {idx + 1}: AP {ap} "
                    f"(scale {atlas_to_histology_scale:.3f}{engine_txt}; score {score:.2f})."
                )
            if self.ck_match_auto_adjust.isChecked():
                self._auto_adjust_current_match_slice(at, idx)

        self._run_bg(
            job,
            done,
            busy_msg=f"Auto matching slice {idx + 1} from tissue shape...",
        )

    def _match_assign(self) -> None:
        self._match_assign_current_plane(log=True)

    def _match_assign_current_plane(self, *, log: bool = True) -> bool:
        at = self._ensure_atlas()
        if at is None or not self.slice_images:
            return False
        idx = min(self._cur_match_slice, len(self.slice_images) - 1)
        cv = hatlas.coronal_camera_vector(self.sl_lr.value(), self.sl_si.value())
        sp = hatlas.coronal_slice_point(self.sl_ap.value(), at)
        while len(self.slice_specs) < len(self.slice_images):
            self.slice_specs.append(None)
        self.slice_specs[idx] = {
            "slice_point": sp,
            "camera_vector": cv,
            "ap": int(self.sl_ap.value()),
            "lr": int(self.sl_lr.value()),
            "si": int(self.sl_si.value()),
            "mode": self.cb_mode.currentText(),
        }
        self._invalidate_downstream_for_match_change(idx)
        self._ensure_histology_ccf_slice(at=at, idx=idx, spacing=1, force=True)
        saved = self._save_match_specs()  # persist immediately so matching survives a reopen
        n_assigned = sum(s is not None for s in self.slice_specs)
        self._match_show()
        if saved:
            self.lbl_match_autosave.setText(f"Autosaved match progress ({n_assigned}/{len(self.slice_images)})")
        if log:
            self._log(f"Assigned plane to slice {idx + 1} "
                      f"({n_assigned}/{len(self.slice_images)} assigned). "
                      "Downstream alignment for this slice was reset.")
        return saved

    def _invalidate_downstream_for_match_change(self, idx: int) -> None:
        """Clear atlas-derived caches that become stale when a match plane changes."""
        if idx < 0:
            return
        while len(self.histology_ccf) <= idx:
            self.histology_ccf.append({})
        self.histology_ccf[idx] = {}
        while len(self.tforms) <= idx:
            self.tforms.append(np.eye(3))
        self.tforms[idx] = np.eye(3)
        self._align_hist_pts.pop(idx, None)
        self._align_atlas_pts.pop(idx, None)
        if self.folder is not None:
            stale = [
                "probe_ccf.mat",
                "probe_ccf.csv",
                "probe_ccf_points.csv",
                "channel_locations_all_shanks.json",
            ]
            stale.extend(f"channel_locations_shank{i}.json" for i in range(1, 5))
            stale.extend(f"xyz_picks_shank{i}.json" for i in range(1, 5))
            stale.append("xyz_picks.json")
            for name in stale:
                try:
                    (self.folder / name).unlink(missing_ok=True)
                except OSError:
                    pass
        self._autosave_tforms()
        current = self.nav.currentIndex() if hasattr(self, "nav") else -1
        if current == getattr(self, "_page_align", -1):
            self._cur_align_slice = idx
            self._align_show()
        elif current == getattr(self, "_page_trace", -1):
            self._cur_trace_slice = idx
            self._trace_show()

    _MATCH_SPECS_FN = "histology_match_specs.json"

    def _save_match_specs(self) -> bool:
        """Persist the per-slice match planes (sidecar to histology_ccf.mat)."""
        if self.folder is None:
            return False
        payload = []
        for s in self.slice_specs:
            if s is None:
                payload.append(None)
                continue
            payload.append({
                "slice_point": np.asarray(s["slice_point"], float).ravel().tolist(),
                "camera_vector": np.asarray(s["camera_vector"], float).ravel().tolist(),
                "ap": int(s.get("ap", 0)),
                "lr": int(s.get("lr", 0)),
                "si": int(s.get("si", 0)),
                "mode": str(s.get("mode", "TV")),
            })
        target = self.folder / self._MATCH_SPECS_FN
        tmp = target.with_suffix(target.suffix + ".tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"specs": payload}, f, indent=2)
                f.write("\n")
            tmp.replace(target)
            return True
        except OSError as exc:
            self._log(f"Could not save match specs: {exc}")
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            return False

    def _read_match_specs_file(self) -> List[Optional[Dict[str, np.ndarray]]]:
        fp = self.folder / self._MATCH_SPECS_FN
        if not fp.exists():
            return []
        try:
            with open(fp) as f:
                data = json.load(f)
        except (OSError, ValueError) as exc:
            self._log(f"Could not load match specs: {exc}")
            return []
        specs: List[Optional[Dict[str, np.ndarray]]] = []
        for s in data.get("specs", []):
            if s is None:
                specs.append(None)
                continue
            try:
                specs.append({
                    "slice_point": np.asarray(s["slice_point"], float),
                    "camera_vector": np.asarray(s["camera_vector"], float),
                    "ap": int(s.get("ap", 0)),
                    "lr": int(s.get("lr", 0)),
                    "si": int(s.get("si", 0)),
                    "mode": str(s.get("mode", "TV")),
                })
            except (KeyError, ValueError, TypeError):
                specs.append(None)
        return specs

    @staticmethod
    def _recover_plane_spec(sl: Dict[str, np.ndarray], dv_n: int, ml_n: int):
        """Recover (slice_point, camera_vector, sliders) from a saved CCF plane."""
        try:
            ap, dv, ml = sl["plane_ap"], sl["plane_dv"], sl["plane_ml"]
        except (KeyError, TypeError):
            return None
        h, w = ap.shape
        if h < 2 or w < 2:
            return None
        y0, x0 = h // 2, w // 2
        P = np.array([ap[y0, x0], dv[y0, x0], ml[y0, x0]], float)
        dx = np.array([ap[y0, x0 + 1] - ap[y0, x0], dv[y0, x0 + 1] - dv[y0, x0], ml[y0, x0 + 1] - ml[y0, x0]])
        dy = np.array([ap[y0 + 1, x0] - ap[y0, x0], dv[y0 + 1, x0] - dv[y0, x0], ml[y0 + 1, x0] - ml[y0, x0]])
        n = np.cross(dx, dy)
        nn = float(np.linalg.norm(n))
        if not np.isfinite(nn) or nn < 1e-9 or not np.isfinite(P).all():
            return None
        cv = n / nn
        if cv[0] < 0:
            cv = -cv  # coronal convention: normal points along +AP
        si = float(np.degrees(np.arcsin(np.clip(cv[1], -1, 1))))
        cp = np.cos(np.radians(si))
        lr = float(np.degrees(np.arcsin(np.clip(cv[2] / cp, -1, 1)))) if abs(cp) > 1e-6 else 0.0
        if abs(cv[0]) > 1e-6:
            ap_s = float((cv @ P - cv[1] * (dv_n / 2.0) - cv[2] * (ml_n / 2.0)) / cv[0])
        else:
            ap_s = float(P[0])
        return {"slice_point": P, "camera_vector": cv,
                "ap": int(round(ap_s)), "lr": int(round(lr)), "si": int(round(si)), "mode": "TV"}

    def _specs_from_histology_ccf(self) -> List[Optional[Dict[str, np.ndarray]]]:
        at = self._ensure_atlas()
        n = max(len(self.slice_images), len(self.histology_ccf))
        specs: List[Optional[Dict[str, np.ndarray]]] = [None] * n
        if at is None:
            return specs
        _, dv_n, ml_n = at.shape
        for i, sl in enumerate(self.histology_ccf):
            if i < n:
                specs[i] = self._recover_plane_spec(sl, dv_n, ml_n)
        return specs

    def _load_match_specs(self) -> None:
        """Restore the per-slice match planes for this run.

        Prefer the saved sidecar; otherwise reconstruct them from histology_ccf.mat
        so a run matched before the sidecar existed still shows its planes (and the
        Match sliders) on reopen instead of starting blank.
        """
        if self.folder is None:
            return
        specs = self._read_match_specs_file()
        from_ccf = False
        if not any(s is not None for s in specs) and self.histology_ccf:
            specs = self._specs_from_histology_ccf()
            from_ccf = True
        if not any(s is not None for s in specs):
            return
        n = max(len(self.slice_images), len(self.histology_ccf), len(specs))
        specs += [None] * (n - len(specs))
        self.slice_specs = specs
        if from_ccf:
            self._save_match_specs()  # cache reconstructed specs so next load is exact
        src = "histology_ccf.mat" if from_ccf else self._MATCH_SPECS_FN
        self._log(f"Restored {sum(s is not None for s in specs)} matched plane(s) from {src}.")

    def _match_save(self) -> None:
        at = self._ensure_atlas()
        if at is None or self.folder is None:
            return
        specs = [s for s in self.slice_specs if s is not None]
        if len(specs) != len(self.slice_images):
            self._log("Assign a plane to every slice before saving histology_ccf.")
            return
        self._save_match_specs()  # keep the editable match state alongside the result

        def job():
            hccf = matching.build_histology_ccf(at, specs, spacing=1)
            io_formats.save_histology_ccf(self.folder / "histology_ccf.mat", hccf)
            io_formats.export_histology_ccf_csv(self.folder, hccf)
            return hccf

        def done(result):
            self.histology_ccf = result
            self._log(f"Saved histology_ccf.mat ({len(result)} slices) + CSV.")
            self._refresh_status()

        self._run_bg(job, done, busy_msg="Building full-resolution histology_ccf...")

    # ------------------------------------------------------ Align actions
    def _align_step(self, d: int) -> None:
        if not self.slice_images:
            return
        self._cur_align_slice = int(np.clip(self._cur_align_slice + d, 0, len(self.slice_images) - 1))
        self._align_show()

    def _align_show(self) -> None:
        if not self.slice_images:
            self.canvas_align_hist.set_image(None)
            self.canvas_align_hist.clear_overlays()
            self.canvas_align_atlas.set_image(None)
            self.canvas_align_atlas.clear_overlays()
            return
        idx = min(self._cur_align_slice, len(self.slice_images) - 1)
        atlas_ready = self._ensure_histology_ccf_slice(idx=idx, spacing=1)
        if atlas_ready:
            atlas_tv = self.histology_ccf[idx].get("tv_slices")
            if atlas_tv is not None and np.asarray(atlas_tv).size:
                self.canvas_align_atlas.set_image(atlas_tv)
            else:
                atlas_ready = False
        if not atlas_ready:
            self.canvas_align_atlas.set_image(None)
            self.canvas_align_atlas.clear_overlays()
            self._log(f"Slice {idx + 1}: atlas image is unavailable; re-run Match atlas for this slice.")
        self.lbl_align.setText(f"slice {idx + 1} / {len(self.slice_images)}")
        # A reopened run with a saved transform shows its overlay immediately.
        if atlas_ready and idx < len(self.tforms) and not np.allclose(np.asarray(self.tforms[idx]), np.eye(3)):
            self._align_overlay(idx)
        else:
            self._show_histology_slice("align", self.canvas_align_hist, idx, self.slice_images[idx])
            self._align_redraw_points()

    def _align_redraw_points(self) -> None:
        idx = self._cur_align_slice
        self.canvas_align_hist.clear_overlays()
        self.canvas_align_atlas.clear_overlays()
        hp = self._align_hist_pts.get(idx, [])
        ap = self._align_atlas_pts.get(idx, [])
        if hp:
            self.canvas_align_hist.add_scatter([p[0] for p in hp], [p[1] for p in hp], "w")
            self._add_numbered_points(self.canvas_align_hist, hp, fill=(20, 20, 20, 210))
        if ap:
            self.canvas_align_atlas.add_scatter([p[0] for p in ap], [p[1] for p in ap], "r")
            self._add_numbered_points(self.canvas_align_atlas, ap, fill=(160, 0, 0, 215))

    @staticmethod
    def _add_numbered_points(canvas: ImageCanvas, points, *, fill) -> None:
        """Overlay 1-based labels beside control points so pairs are easy to match."""
        for i, (x, y) in enumerate(points, start=1):
            canvas.add_text(
                str(i),
                float(x) + 6.0,
                float(y) - 6.0,
                color="w",
                fill=pg.mkBrush(*fill),
                border=pg.mkPen(255, 255, 255, 190),
            )

    def _align_click_hist(self, x: float, y: float) -> None:
        self._align_hist_pts.setdefault(self._cur_align_slice, []).append((x, y))
        self._align_redraw_points()

    def _align_click_atlas(self, x: float, y: float) -> None:
        self._align_atlas_pts.setdefault(self._cur_align_slice, []).append((x, y))
        self._align_redraw_points()

    def _align_clear(self) -> None:
        self._align_hist_pts[self._cur_align_slice] = []
        self._align_atlas_pts[self._cur_align_slice] = []
        self._align_redraw_points()

    def _align_apply_points(self) -> None:
        idx = self._cur_align_slice
        hp = self._align_hist_pts.get(idx, [])
        ap = self._align_atlas_pts.get(idx, [])
        if len(hp) < 3 or len(hp) != len(ap):
            self._log("Need >= 3 matching point pairs (histology and atlas).")
            return
        T = alignment.fit_affine_from_points(np.array(ap), np.array(hp))
        self._set_tform(idx, T)
        self._log(f"Applied control-point affine to slice {idx + 1}.")
        self._align_overlay(idx)
        self._autosave_tforms()

    def _align_auto(self) -> None:
        idx = self._cur_align_slice
        if idx >= len(self.histology_ccf) or idx >= len(self.slice_images):
            self._log("Match atlas slices first (need histology_ccf).")
            return
        hp = self._align_hist_pts.get(idx, [])
        ap = self._align_atlas_pts.get(idx, [])
        if len(hp) >= 3 and len(hp) == len(ap):
            T = alignment.fit_affine_from_points(np.array(ap), np.array(hp))
            self._set_tform(idx, T)
            self._log(f"Slice {idx + 1}: auto-align used {len(hp)} existing landmark pairs.")
            self._align_overlay(idx)
            self._autosave_tforms()
            return
        if hp or ap:
            self._log("Auto-align ignored incomplete landmark points. Need the same "
                      "number of histology and atlas points, with at least 3 pairs.")
            return
        hist = self.slice_images[idx]
        atlas_tv = self.histology_ccf[idx].get("tv_slices")
        if atlas_tv is None or np.asarray(atlas_tv).size == 0:
            self._log("Atlas slice has no tv_slices; re-run Match before auto-align.")
            return

        def job():
            # Runs accelerated shape alignment in-process; only the legacy ECC
            # fallback is isolated because it can abort natively.
            return alignment.auto_align_isolated(hist, atlas_tv)

        def done(result):
            T, status = result
            if "low confidence" in str(status).lower():
                self._log(f"Slice {idx + 1}: {status}")
                self._log("Auto-align did not overwrite the current transform. Add "
                          "landmark pairs and press Auto-align or Apply points.")
                return
            self._set_tform(idx, T)
            self._log(f"Slice {idx + 1}: {status}")
            self._align_overlay(idx)
            self._autosave_tforms()

        self._run_bg(job, done,
                     busy_msg="Auto-aligning with accelerated shape registration...")

    def _ensure_histology_ccf_slice(
        self,
        at: Optional[hatlas.AllenCCFAtlas] = None,
        idx: int = 0,
        spacing: int = 1,
        force: bool = False,
    ) -> bool:
        """Build/store ``histology_ccf[idx]`` from the current match spec if needed."""
        if not force and idx < len(self.histology_ccf) and self.histology_ccf[idx]:
            sl = self.histology_ccf[idx]
            if all(k in sl and np.asarray(sl[k]).size for k in ("tv_slices", "av_slices")):
                return True
        if idx >= len(self.slice_specs) or self.slice_specs[idx] is None:
            return False
        at = at or self._ensure_atlas()
        if at is None:
            return False
        while len(self.histology_ccf) <= idx:
            self.histology_ccf.append({})
        try:
            self.histology_ccf[idx] = matching.build_histology_ccf(
                at, [self.slice_specs[idx]], spacing=spacing
            )[0]
        except Exception as exc:
            self._log(f"Could not rebuild atlas slice {idx + 1}: {exc}")
            return False
        sl = self.histology_ccf[idx]
        if all(k in sl and np.asarray(sl[k]).size for k in ("tv_slices", "av_slices")):
            return True
        return False

    def _auto_adjust_current_match_slice(self, at: hatlas.AllenCCFAtlas, idx: int) -> None:
        """Run the existing intensity auto-align after an Auto Match result."""
        if idx >= len(self.slice_images):
            return
        if not self._ensure_histology_ccf_slice(at, idx, spacing=1):
            self._log("Auto-adjust skipped: no matched plane is available for this slice.")
            return
        hist = self.slice_images[idx]
        atlas_tv = self.histology_ccf[idx].get("tv_slices")
        if atlas_tv is None or np.asarray(atlas_tv).size == 0:
            self._log("Auto-adjust skipped: atlas plane has no template image.")
            return

        def job():
            return alignment.auto_align_isolated(hist, atlas_tv)

        def done(result):
            T, status = result
            if "low confidence" in str(status).lower():
                self._log(f"Slice {idx + 1}: auto-adjust after Auto match skipped: {status}")
                self.lbl_match_autosave.setText(f"Auto-adjust skipped for slice {idx + 1}")
                return
            self._set_tform(idx, T)
            self._autosave_tforms()
            self._log(f"Slice {idx + 1}: auto-adjust after Auto match: {status}")
            self.lbl_match_autosave.setText(f"Auto-adjusted and autosaved slice {idx + 1}")
            if self.nav.currentIndex() == getattr(self, "_page_align", -1):
                self._cur_align_slice = idx
                self._align_overlay(idx)

        self._run_bg(
            job,
            done,
            busy_msg=f"Auto-adjusting slice {idx + 1} after Auto match...",
        )

    def _set_tform(self, idx: int, T: np.ndarray) -> None:
        while len(self.tforms) <= idx:
            self.tforms.append(np.eye(3))
        self.tforms[idx] = T

    def _align_overlay(self, idx: int) -> None:
        if idx >= len(self.tforms) or idx >= len(self.slice_images):
            return
        if not self._ensure_histology_ccf_slice(idx=idx, spacing=1):
            self._log(f"Slice {idx + 1}: atlas overlay unavailable; re-run Match atlas for this slice.")
            return
        try:
            av = self.histology_ccf[idx]["av_slices"]
            hist = self.slice_images[idx]
            shape = hist.shape[:2]
            # use_cv2=False: GUI-thread warp must not be able to abort natively.
            warped = alignment.warp_atlas(av, self.tforms[idx], shape, nearest=True, use_cv2=False)
            bound = alignment.atlas_boundaries(warped)
            self._show_histology_slice("align", self.canvas_align_hist, idx, self.slice_images[idx])
            self._align_redraw_points()
            self.canvas_align_hist.add_mask_overlay(bound, (60, 180, 255))
        except Exception as exc:
            self._log(f"Could not draw atlas overlay: {exc}")

    def _write_tforms(self) -> bool:
        if self.folder is None or not self.tforms:
            return False
        while len(self.tforms) < len(self.slice_images):
            self.tforms.append(np.eye(3))
        io_formats.save_tforms(self.folder / "atlas2histology_tform.mat", self.tforms)
        return True

    def _autosave_tforms(self) -> None:
        """Persist alignment immediately after an apply/auto-align."""
        if self._write_tforms():
            self._log("Auto-saved atlas2histology_tform.mat.")
            self._refresh_status()

    def _align_save(self) -> None:
        if not self._write_tforms():
            self._log("Nothing to save.")
            return
        self._log("Saved atlas2histology_tform.mat.")
        self._refresh_status()

    # ------------------------------------------------------ Trace actions
    def _trace_set_probe(self, v: int) -> None:
        self._remember_visible_histology_levels("trace")
        self._trace_commit_roi()
        self._active_probe = int(v)
        self._pending_click = []
        self._trace_show()

    def _trace_step(self, d: int) -> None:
        if not self.slice_images:
            return
        self._remember_visible_histology_levels("trace")
        self._trace_commit_roi()
        self._cur_trace_slice = int(np.clip(self._cur_trace_slice + d, 0, len(self.slice_images) - 1))
        self._pending_click = []
        self._trace_show()

    def _trace_key(self) -> Tuple[int, int]:
        return self._cur_trace_slice, self._active_probe

    def _trace_visible_center(self, h: int, w: int) -> Tuple[float, float] | None:
        try:
            (x0, x1), (y0, y1) = self.canvas_trace.view.viewRange()
        except Exception:
            return None
        vals = np.asarray([x0, x1, y0, y1], dtype=float)
        if not np.isfinite(vals).all():
            return None
        x = float(np.clip((x0 + x1) / 2.0, 0, max(w - 1, 0)))
        y = float(np.clip((y0 + y1) / 2.0, 0, max(h - 1, 0)))
        return x, y

    def _trace_reference_line(self, exclude_key: Tuple[int, int] | None = None) -> tuple[np.ndarray, int, int] | None:
        current = self._trace_key()
        exclude_key = current if exclude_key is None else exclude_key
        roi_pts = self._trace_roi_points()
        if roi_pts is not None and current != exclude_key:
            return roi_pts, current[0], current[1]

        candidates: list[tuple[tuple[int, int, int, int], np.ndarray, int, int]] = []
        for (s, p), pts in self.probe_points.items():
            if (s, p) == exclude_key:
                continue
            arr = np.asarray(pts, dtype=float)
            if arr.shape != (2, 2) or not np.isfinite(arr).all():
                continue
            same_slice = 0 if s == self._cur_trace_slice else 1
            same_probe = 0 if p == self._active_probe else 1
            probe_dist = abs(p - self._active_probe)
            slice_dist = abs(s - self._cur_trace_slice)
            candidates.append(((same_slice, same_probe, probe_dist, slice_dist), arr, s, p))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        _rank, pts, s, p = candidates[0]
        return pts, s, p

    @staticmethod
    def _trace_fit_line_to_image(pts: np.ndarray, h: int, w: int) -> np.ndarray:
        out = np.asarray(pts, dtype=float).copy()
        if out.shape != (2, 2):
            return out
        max_x = float(max(w - 1, 0))
        max_y = float(max(h - 1, 0))
        span_x = float(out[:, 0].max() - out[:, 0].min())
        span_y = float(out[:, 1].max() - out[:, 1].min())
        if span_x <= max_x:
            if out[:, 0].min() < 0:
                out[:, 0] -= out[:, 0].min()
            if out[:, 0].max() > max_x:
                out[:, 0] -= out[:, 0].max() - max_x
        if span_y <= max_y:
            if out[:, 1].min() < 0:
                out[:, 1] -= out[:, 1].min()
            if out[:, 1].max() > max_y:
                out[:, 1] -= out[:, 1].max() - max_y
        out[:, 0] = np.clip(out[:, 0], 0, max_x)
        out[:, 1] = np.clip(out[:, 1], 0, max_y)
        return out

    def _trace_default_line(self, center: Optional[Tuple[float, float]] = None) -> np.ndarray:
        idx = min(self._cur_trace_slice, len(self.slice_images) - 1)
        img = self.slice_images[idx]
        h, w = img.shape[:2]
        ref = self._trace_reference_line()
        if ref is not None:
            ref_pts, ref_slice, ref_probe = ref
            vec = ref_pts[1] - ref_pts[0]
            if not np.isfinite(vec).all() or float(np.linalg.norm(vec)) < 2.0:
                vec = np.array([0.0, max(40.0, min(120.0, 0.20 * h))], dtype=float)
            if center is None:
                c = ref_pts.mean(axis=0)
                if ref_slice == idx and ref_probe != self._active_probe:
                    perp = np.array([-vec[1], vec[0]], dtype=float)
                    norm = float(np.linalg.norm(perp))
                    if norm > 1e-6:
                        perp /= norm
                        sign = 1.0 if self._active_probe >= ref_probe else -1.0
                        c = c + sign * perp * max(10.0, min(45.0, 0.20 * float(np.linalg.norm(vec))))
                x = float(np.clip(c[0], 0, max(w - 1, 0)))
                y = float(np.clip(c[1], 0, max(h - 1, 0)))
            else:
                x = float(np.clip(center[0], 0, max(w - 1, 0)))
                y = float(np.clip(center[1], 0, max(h - 1, 0)))
            pts = np.array([[x, y], [x, y]], dtype=float)
            pts[0] -= 0.5 * vec
            pts[1] += 0.5 * vec
            return self._trace_fit_line_to_image(pts, h, w)

        if center is None:
            visible = self._trace_visible_center(h, w)
            if visible is not None:
                x, y = visible
            else:
                x = float(np.clip(w * (0.42 + 0.045 * ((self._active_probe - 1) % 6)), 0, max(w - 1, 0)))
                y = h * 0.56
        else:
            x = float(np.clip(center[0], 0, max(w - 1, 0)))
            y = float(np.clip(center[1], 0, max(h - 1, 0)))
        length = max(40.0, min(140.0, 0.22 * h))
        y0 = float(np.clip(y - 0.5 * length, 0, max(h - 1, 0)))
        y1 = float(np.clip(y + 0.5 * length, 0, max(h - 1, 0)))
        if abs(y1 - y0) < 1.0:
            y0, y1 = 0.0, float(max(h - 1, 1))
        return np.array([[x, y0], [x, y1]], dtype=float)

    def _trace_roi_points(self) -> Optional[np.ndarray]:
        roi = self._trace_roi
        if roi is None:
            return None
        pts = []
        for handle in roi.getHandles():
            p = roi.mapToParent(handle.pos())
            pts.append([float(p.x()), float(p.y())])
        if len(pts) != 2:
            return None
        return np.asarray(pts, dtype=float)

    def _trace_commit_roi(self) -> None:
        pts = self._trace_roi_points()
        if pts is not None:
            self.probe_points[self._trace_key()] = pts
            self._save_probe_lines()

    _PROBE_LINES_FN = "histology_probe_lines.json"

    def _save_probe_lines(self) -> None:
        """Persist drawn probe lines so a reopened run shows them immediately."""
        if self.folder is None:
            return
        lines = []
        for (s, p), pts in self.probe_points.items():
            arr = np.asarray(pts, dtype=float) if pts is not None else None
            if arr is None or arr.shape != (2, 2) or not np.isfinite(arr).all():
                continue
            lines.append({"slice": int(s), "probe": int(p), "points": arr.tolist()})
        try:
            with open(self.folder / self._PROBE_LINES_FN, "w") as f:
                json.dump({"lines": lines}, f, indent=2)
        except OSError as exc:
            self._log(f"Could not save probe lines: {exc}")

    def _load_probe_lines(self) -> None:
        if self.folder is None:
            return
        fp = self.folder / self._PROBE_LINES_FN
        if not fp.exists():
            return
        try:
            with open(fp) as f:
                data = json.load(f)
        except (OSError, ValueError) as exc:
            self._log(f"Could not load probe lines: {exc}")
            return
        pts_map: Dict[Tuple[int, int], np.ndarray] = {}
        for ln in data.get("lines", []):
            try:
                arr = np.asarray(ln["points"], dtype=float)
                if arr.shape == (2, 2) and np.isfinite(arr).all():
                    pts_map[(int(ln["slice"]), int(ln["probe"]))] = arr
            except (KeyError, ValueError, TypeError):
                continue
        if pts_map:
            self.probe_points = pts_map
            self._log(f"Restored {len(pts_map)} probe line(s).")

    def _trace_roi_finished(self, *args) -> None:
        self._trace_roi_changed(*args)
        self._save_probe_lines()

    def _trace_show(self) -> None:
        if not self.slice_images:
            self.canvas_trace.set_image(None)
            self.canvas_trace.clear_overlays()
            return
        idx = min(self._cur_trace_slice, len(self.slice_images) - 1)
        self._show_histology_slice("trace", self.canvas_trace, idx, self.slice_images[idx], preserve_view=True)
        self.canvas_trace.clear_overlays()
        self._trace_roi = None
        active_pts = None
        for (s, p), pts in self.probe_points.items():
            if s == idx and pts is not None and len(pts):
                color = PROBE_QCOLORS[(p - 1) % len(PROBE_QCOLORS)]
                if p == self._active_probe:
                    active_pts = np.asarray(pts, dtype=float)
                else:
                    self.canvas_trace.add_line(pts[:, 0], pts[:, 1], color, 4)
                    self.canvas_trace.add_text(
                        f"S{p}",
                        float(pts[0, 0]) + 5.0,
                        float(pts[0, 1]) - 5.0,
                        color="w",
                        fill=pg.mkBrush(color.red(), color.green(), color.blue(), 190),
                        border=pg.mkPen(255, 255, 255, 160),
                    )
        if active_pts is not None and len(active_pts) == 2:
            color = PROBE_QCOLORS[(self._active_probe - 1) % len(PROBE_QCOLORS)]
            self._trace_roi = self.canvas_trace.add_line_roi(active_pts, color, 3)
            self._trace_roi.sigRegionChanged.connect(self._trace_roi_changed)
            self._trace_roi.sigRegionChangeFinished.connect(self._trace_roi_finished)
            self.canvas_trace.add_text(
                f"S{self._active_probe}",
                float(active_pts[0, 0]) + 5.0,
                float(active_pts[0, 1]) - 5.0,
                color="w",
                fill=pg.mkBrush(color.red(), color.green(), color.blue(), 210),
                border=pg.mkPen(255, 255, 255, 190),
            )
            self._trace_update_controls(active_pts)
        else:
            self._trace_update_controls(None)
        self.lbl_trace.setText(f"slice {idx + 1} / {len(self.slice_images)}")

    def _trace_click(self, x: float, y: float) -> None:
        key = self._trace_key()
        if key in self.probe_points:
            return
        self.probe_points[key] = self._trace_default_line((x, y))
        self._save_probe_lines()
        self._trace_show()
        self._log(f"Created editable shank {self._active_probe} line on slice {key[0] + 1}.")

    def _trace_new_line(self) -> None:
        if not self.slice_images:
            return
        key = self._trace_key()
        self.probe_points[key] = self._trace_default_line()
        self._save_probe_lines()
        self._trace_show()
        self._log(f"Created editable shank {self._active_probe} line on slice {key[0] + 1}.")

    def _trace_roi_changed(self, *_args) -> None:
        if self._trace_updating_roi:
            return
        pts = self._trace_roi_points()
        if pts is None:
            return
        self.probe_points[self._trace_key()] = pts
        self._trace_update_controls(pts)

    def _trace_update_controls(self, pts: Optional[np.ndarray]) -> None:
        self._trace_updating_controls = True
        try:
            enabled = pts is not None and len(pts) == 2
            for spin in self._trace_coord_spins.values():
                spin.setEnabled(enabled)
            if enabled:
                vals = {
                    "x1": float(pts[0, 0]), "y1": float(pts[0, 1]),
                    "x2": float(pts[1, 0]), "y2": float(pts[1, 1]),
                }
                for key, val in vals.items():
                    self._trace_coord_spins[key].setValue(val)
                self.lbl_trace_line.setText(
                    f"shank {self._active_probe} on slice {self._cur_trace_slice + 1}"
                )
            else:
                for spin in self._trace_coord_spins.values():
                    spin.setValue(0.0)
                self.lbl_trace_line.setText("no line for selected shank")
        finally:
            self._trace_updating_controls = False

    def _trace_controls_changed(self) -> None:
        if self._trace_updating_controls or not self.slice_images:
            return
        pts = np.array([
            [self._trace_coord_spins["x1"].value(), self._trace_coord_spins["y1"].value()],
            [self._trace_coord_spins["x2"].value(), self._trace_coord_spins["y2"].value()],
        ], dtype=float)
        self.probe_points[self._trace_key()] = pts
        self._save_probe_lines()
        if self._trace_roi is None:
            self._trace_show()
            return
        self._trace_updating_roi = True
        try:
            state = self._trace_roi.saveState()
            state["pos"] = (0.0, 0.0)
            state["points"] = [tuple(pts[0]), tuple(pts[1])]
            self._trace_roi.setState(state)
        finally:
            self._trace_updating_roi = False

    def _trace_clear(self) -> None:
        self.probe_points.pop((self._cur_trace_slice, self._active_probe), None)
        self._pending_click = []
        self._save_probe_lines()
        self._trace_show()

    @staticmethod
    def _trace_slice_has_ccf_mapping(sl: Dict[str, np.ndarray]) -> bool:
        """Return True when a matched atlas slice has every array needed for probe_ccf."""
        needed = ("plane_ap", "plane_ml", "plane_dv", "tv_slices", "av_slices")
        return all(k in sl and np.asarray(sl[k]).size for k in needed)

    def _trace_build_inputs(
        self,
        at: hatlas.AllenCCFAtlas,
    ) -> Optional[Tuple[Dict[Tuple[int, int], np.ndarray], List[Dict[str, np.ndarray]], List[np.ndarray], int]]:
        """Validate traced shanks and build dense per-slice inputs for probe_ccf."""
        if not self.probe_points:
            self._log("Draw at least one shank track first.")
            return None
        if not self.tforms:
            self._log("Need atlas2histology transforms first. Run Align and save tforms.")
            return None

        valid_points: Dict[Tuple[int, int], np.ndarray] = {}
        stale_keys: List[Tuple[int, int]] = []
        bad_keys: List[Tuple[int, int]] = []
        missing_match_slices: List[int] = []

        for (s, p), pts in sorted(self.probe_points.items()):
            arr = np.asarray(pts, dtype=float)
            if arr.shape != (2, 2) or not np.isfinite(arr).all():
                bad_keys.append((s, p))
                continue
            if s < 0 or s >= len(self.slice_images):
                stale_keys.append((s, p))
                continue

            if s >= len(self.histology_ccf) or not self._trace_slice_has_ccf_mapping(self.histology_ccf[s]):
                if not self._ensure_histology_ccf_slice(at=at, idx=s, spacing=1, force=True):
                    missing_match_slices.append(s)
                    continue
            if s >= len(self.histology_ccf) or not self._trace_slice_has_ccf_mapping(self.histology_ccf[s]):
                missing_match_slices.append(s)
                continue
            valid_points[(s, p)] = arr.copy()

        for key in stale_keys + bad_keys:
            self.probe_points.pop(key, None)
        if stale_keys or bad_keys:
            self._save_probe_lines()
            self._trace_show()
        if stale_keys:
            self._log(f"Removed {len(stale_keys)} stale shank line(s) outside the current slice list.")
        if bad_keys:
            self._log(f"Skipped {len(bad_keys)} invalid shank line(s).")
        if missing_match_slices:
            shown = ", ".join(str(i + 1) for i in sorted(set(missing_match_slices)))
            self._log(f"Match atlas slices first for traced slice(s): {shown}.")
            return None
        if not valid_points:
            self._log("No valid shank tracks remain to build probe_ccf.")
            return None

        max_slice = max(s for s, _p in valid_points)
        while len(self.histology_ccf) <= max_slice:
            self.histology_ccf.append({})
        missing_tform_slices: List[int] = []
        while len(self.tforms) <= max_slice:
            self.tforms.append(np.eye(3))
            missing_tform_slices.append(len(self.tforms) - 1)
        if missing_tform_slices:
            shown = ", ".join(str(i + 1) for i in missing_tform_slices)
            self._log(f"Using identity alignment for slice(s) without saved tforms: {shown}.")

        # tracing.build_probe_ccf indexes by absolute slice number, so the copied
        # arrays must be dense up to the highest traced slice even when only a few
        # slices contain shank lines.
        pts0 = {(s, p - 1): arr for (s, p), arr in valid_points.items()}
        histology_ccf = [dict(sl) for sl in self.histology_ccf[:max_slice + 1]]
        tforms = [np.asarray(t, dtype=float).copy() for t in self.tforms[:max_slice + 1]]
        n_probes = max(p for _s, p in valid_points)
        return pts0, histology_ccf, tforms, n_probes

    def _trace_build(self) -> None:
        self._trace_commit_roi()
        if self.folder is None:
            return
        # Load the atlas on the GUI thread (mmap is instant) so the worker is pure
        # compute + save; this also caches it and surfaces a slow first load.
        at = self._ensure_atlas()
        if at is None:
            return
        inputs = self._trace_build_inputs(at)
        if inputs is None:
            return
        pts0, histology_ccf, tforms, n_probes = inputs
        folder = self.folder
        emit_log = self.log_requested.emit  # thread-safe logging from the worker
        if hasattr(self, "btn_trace_build"):
            self.btn_trace_build.setEnabled(False)

        def job():
            emit_log("[probe_ccf] worker started; sampling trajectories through the CCF...")
            s = time.perf_counter()
            probes = tracing.build_probe_ccf(pts0, histology_ccf, tforms, at, n_probes)
            emit_log(f"[probe_ccf] sampled {len(probes)} probe(s) in {time.perf_counter() - s:.2f}s.")
            s = time.perf_counter()
            io_formats.save_probe_ccf(folder / "probe_ccf.mat", probes)
            io_formats.export_probe_ccf_csv(folder, probes)
            emit_log(f"[probe_ccf] wrote probe_ccf.mat + CSV in {time.perf_counter() - s:.2f}s "
                     f"(folder: {folder}).")
            return probes

        def done(probes):
            self._log(f"Saved probe_ccf.mat ({len(probes)} probes) + CSV.")
            self._refresh_status()
            self._draw_trajectory_areas(probes)
            self._save_trajectory_3d_gif(probes)

        def finished(_payload):
            if hasattr(self, "btn_trace_build"):
                self.btn_trace_build.setEnabled(True)

        self._run_bg(
            job,
            done,
            busy_msg="Building probe_ccf in the background...",
            on_finished=finished,
        )

    def _draw_trajectory_areas(self, probes) -> None:
        # Defensive: anything that escapes here runs on the GUI thread and would
        # otherwise reach the excepthook (or, for non-finite coordinates handed to
        # pyqtgraph, abort the Qt paint engine). Sanitize and contain per probe.
        try:
            self.trace_areas.clear()
        except Exception as exc:
            self._log(f"Could not reset trajectory chart: {exc}")
            return
        for i, p in enumerate(probes):
            try:
                self._draw_one_trajectory(i, p)
            except Exception as exc:
                self._log(f"Could not draw trajectory for probe {i + 1}: {exc}")
        try:
            self.trace_3d.render(self.atlas, probes)
        except Exception as exc:
            self._log(f"Could not draw 3D probe trajectories: {exc}")

    def _draw_one_trajectory(self, i: int, p: dict) -> None:
        ta = p.get("trajectory_areas")
        _crumb(f"draw_trajectory probe {i} addPlot")
        plt = self.trace_areas.addPlot(row=0, col=i)
        plt.setTitle(f"Shank {i + 1}")
        plt.invertY(True)
        plt.hideAxis("bottom")
        if ta is None or len(ta) == 0:
            return
        total = 0.0
        for j in range(len(ta)):
            d0 = float(ta.iloc[j].get("depth_start_um", 0))
            d1 = float(ta.iloc[j].get("depth_end_um", 0))
            if not (np.isfinite(d0) and np.isfinite(d1)) or d1 <= d0:
                continue  # never hand non-finite/degenerate spans to pyqtgraph
            total = max(total, d1)
            hexc = str(ta.iloc[j].get("color_hex_triplet", "808080"))
            try:
                col = QtGui.QColor(int(hexc[0:2], 16), int(hexc[2:4], 16), int(hexc[4:6], 16))
            except (ValueError, IndexError):
                col = QtGui.QColor(128, 128, 128)
            plt.addItem(pg.BarGraphItem(x=[0], y0=[d0], y1=[d1], width=1, brush=col))
            # Only label spans tall enough to read, to keep the item count sane.
            if (d1 - d0) >= 60.0:
                acr = str(ta.iloc[j].get("acronym", ""))
                if acr:
                    txt = pg.TextItem(acr, color="k", anchor=(0, 0.5))
                    txt.setPos(0.05, (d0 + d1) / 2.0)
                    plt.addItem(txt)
        _crumb(f"draw_trajectory probe {i} done")

    def _show_trajectory_3d_popup(self, probes) -> None:
        if self.atlas is None:
            return
        if self._trajectory_3d_dialog is None:
            dlg = QtWidgets.QDialog(self)
            dlg.setWindowTitle("Shank trajectories 3D")
            dlg.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)
            dlg.resize(760, 680)

            layout = QtWidgets.QVBoxLayout(dlg)
            layout.setContentsMargins(10, 10, 10, 10)
            layout.setSpacing(8)

            canvas = Trajectory3DCanvas()
            status = QtWidgets.QLabel("")
            status.setObjectName("SectionHint")
            status.setWordWrap(True)

            layout.addWidget(canvas, 1)
            layout.addWidget(status, 0)

            self._trajectory_3d_dialog = dlg
            self._trajectory_3d_popup = canvas
            self._trajectory_3d_status = status

        if self._trajectory_3d_popup is not None:
            self._trajectory_3d_popup.set_theme(self._plot_theme)
            self._trajectory_3d_popup.render(self.atlas, probes)
        if self._trajectory_3d_status is not None and self.folder is not None:
            self._trajectory_3d_status.setText(
                f"GIF: {self.folder / 'probe_trajectories_3d.gif'}"
            )
        self._trajectory_3d_dialog.show()
        self._trajectory_3d_dialog.raise_()
        self._trajectory_3d_dialog.activateWindow()

    @staticmethod
    def _copy_probe_geometry_for_gif(probes) -> List[dict]:
        copied = []
        for probe in probes or []:
            areas = probe.get("trajectory_areas")
            if hasattr(areas, "copy"):
                areas = areas.copy()
            elif isinstance(areas, dict):
                areas = dict(areas)
            copied.append({
                "trajectory_coords": np.asarray(
                    probe.get("trajectory_coords", np.zeros((0, 3))), dtype=float
                ).copy(),
                "points": np.asarray(probe.get("points", np.zeros((0, 3))), dtype=float).copy(),
                "trajectory_areas": areas,
            })
        return copied

    def _save_trajectory_3d_gif(self, probes) -> None:
        if self.folder is None or self.atlas is None:
            return
        out_path = self.folder / "probe_trajectories_3d.gif"
        atlas_path = str(self.atlas.atlas_path)
        probes_copy = self._copy_probe_geometry_for_gif(probes)
        theme = self._plot_theme

        def job():
            at = hatlas.AllenCCFAtlas(atlas_path)
            return Trajectory3DCanvas.save_gif(out_path, at, probes_copy, theme=theme)

        def done(path):
            self._log(f"Saved 3D trajectory GIF: {path}")
            if self._trajectory_3d_status is not None:
                self._trajectory_3d_status.setText(f"GIF saved: {path}")

        self._run_bg(
            job,
            done,
            busy_msg=f"Saving 3D trajectory GIF to {out_path}...",
        )

    # --------------------------------------------------- Channel map / IBL
    def _ibl_kwargs(self) -> dict:
        return {
            "ibl_python": self.ed_pyexe.text().strip() or None,
            "iblapps_path": self.ed_iblapps.text().strip() or None,
            # run_bridge streams subprocess output from a worker thread; route it
            # through the queued signal so the log widget is only touched on the
            # GUI thread (writing a QPlainTextEdit off-thread crashes Qt natively).
            "log": self.log_requested.emit,
        }

    def _extract_alf(self, *, compute_rms: bool = False) -> None:
        if self.folder is None:
            self._log("Load a session folder first.")
            return
        ks = self.ed_ks.text().strip()
        ephys = self.ed_ephys.text().strip()
        if not ks or not ephys:
            self._log("ALF extraction needs both Kilosort and ephys folders set on Setup.")
            return
        args = ["extract_alf", ks, ephys, str(self.folder)]
        if compute_rms:
            args.append("--rms")
        kw = self._ibl_kwargs()

        def job():
            return ibl_launch.run_bridge(args, **kw)[0]

        def done(rc):
            if rc == 0:
                self._log("ALF extraction done.")
                self._refresh_status()
                self._load_channel_table()
            else:
                self._log("ALF extraction failed.")

        self._run_bg(job, done, busy_msg="Running fast ALF extraction via IBL bridge...")

    def _compute_rms_qc_maps(self) -> None:
        """Run only the optional slow RMS/QC extraction path for the IBL GUI."""
        self._extract_alf(compute_rms=True)

    def _prepare_for_ibl(self) -> None:
        """Run the complete preparation pipeline needed before opening the IBL GUI."""
        if self.folder is None:
            self._log("Load a session folder first.")
            return
        if not (self.folder / "probe_ccf.mat").exists():
            self._log("Need probe_ccf.mat first. Finish Trace probes and save probe_ccf.")
            return
        ks = self.ed_ks.text().strip()
        ephys = self.ed_ephys.text().strip()
        if not ks or not ephys:
            self._log("Prepare for IBL needs both Kilosort and ephys folders set on Setup.")
            return

        align = self.cb_alignment.currentText()
        args = [
            "all",
            str(self.folder),
            "--alignment", align,
            "--ks", ks,
            "--ephys", ephys,
        ]
        kw = self._ibl_kwargs()

        def job():
            rc, _ = ibl_launch.run_bridge(args, **kw)
            return rc

        def done(rc):
            if rc != 0:
                self._log("Prepare for IBL failed.")
                return
            self._log("Prepare for IBL complete: ALF, xyz_picks, and channel map are ready.")
            self._log("RMS/QC maps were skipped for speed. Use 'Optional RMS/QC maps' only if you need those IBL GUI panels.")
            self._on_channels_done(0)

        self._run_bg(
            job,
            done,
            busy_msg="Preparing IBL inputs (fast ALF + xyz_picks + channel map)...",
        )

    def _gen_xyz(self) -> None:
        if self.folder is None or not (self.folder / "probe_ccf.mat").exists():
            self._log("Need probe_ccf.mat (Trace stage) first.")
            return
        folder = self.folder
        kw = self._ibl_kwargs()

        def job():
            try:
                from ..histology import ibl_bridge
                written = ibl_bridge.compute_xyz_picks(
                    folder / "probe_ccf.mat",
                    folder,
                    mode="fast",
                )
                return {"rc": 0, "mode": "fast", "count": len(written)}
            except Exception as exc:
                self.log_requested.emit(
                    f"Fast xyz_picks failed ({exc}); falling back to exact IBL mode."
                )
                rc, _ = ibl_launch.run_bridge(
                    ["xyz_picks", str(folder), "--mode", "ibl"],
                    **kw,
                )
                return {"rc": rc, "mode": "ibl", "count": 0}

        def done(result):
            rc = int(result.get("rc", 1))
            if rc == 0:
                mode = result.get("mode", "fast")
                count = result.get("count", 0)
                self._log(f"xyz_picks done ({mode}, {count} file(s)).")
            else:
                self._log("xyz_picks failed.")

        self._run_bg(job, done, busy_msg="Generating xyz_picks locally...")

    def _gen_channels(self, alignment_override: Optional[str] = None) -> None:
        if self.folder is None:
            return
        align = alignment_override or self.cb_alignment.currentText()
        kw = self._ibl_kwargs()
        args = ["channels", str(self.folder), "--alignment", align]
        ks = self.ed_ks.text().strip()
        if ks:
            args += ["--ks", ks]  # lets the bridge reuse channel_positions.npy

        def job():
            rc, _ = ibl_launch.run_bridge(args, **kw)
            return rc

        self._run_bg(job, lambda rc: self._on_channels_done(rc),
                     busy_msg=f"Generating channel map ({align}) via IBL bridge...")

    def _gen_all(self) -> None:
        if self.folder is None:
            return
        args = ["all", str(self.folder), "--alignment", self.cb_alignment.currentText()]
        ks = self.ed_ks.text().strip()
        ephys = self.ed_ephys.text().strip()
        if ks:
            args += ["--ks", ks]  # geometry reuse; extraction only if --ephys is added too
        if ks and ephys:
            args += ["--ephys", ephys]
        kw = self._ibl_kwargs()
        self._run_bg(lambda: ibl_launch.run_bridge(args, **kw)[0],
                     lambda rc: self._on_channels_done(rc),
                     busy_msg="Running full channel-map pipeline via IBL bridge...")

    def _on_channels_done(self, rc: int) -> None:
        if rc != 0:
            self._log("Channel map generation failed.")
            return
        self._log("Channel map generated.")
        self._refresh_status()
        self._load_channel_table()
        if self.folder is not None and (self.folder / "clusters.channels.npy").exists():
            self._plot_unit_distribution()  # auto-show the summary when units exist

    def _finalize_channels(self) -> None:
        """Rebuild per-shank channel regions from the latest IBL GUI alignments."""
        if self.folder is None:
            self._log("Load a session folder first.")
            return
        prev = sorted(self.folder.glob("prev_alignments*.json"))
        if not prev:
            self._log("No saved IBL alignments found. In the IBL GUI, align each shank "
                      "and press Upload first (writes prev_alignments_shankN.json).")
            return
        self._log(f"Finalizing channel regions from {len(prev)} saved alignment file(s) "
                  "(latest per shank).")
        self._gen_channels(alignment_override="latest")

    def _propose_alignment(self) -> None:
        if self.folder is None:
            self._log("Load a session folder first.")
            return
        if not (self.folder / "clusters.channels.npy").exists():
            self._log("Propose alignment needs ALF cluster files. Run the channel map "
                      "with 'Run ALF extraction first' checked.")
            return
        args = ["propose_align", str(self.folder)]
        atlas = self.ed_atlas.text().strip()
        if atlas:
            args += ["--atlas", atlas]
        kw = self._ibl_kwargs()

        def job():
            return ibl_launch.run_bridge(args, **kw)[0]

        def done(rc):
            if rc != 0:
                self._log("Alignment proposal failed.")
                return
            self._log(f"Alignment proposal written. Report: "
                      f"{self.folder / 'alignment_report.md'}")
            self._log("In the IBL GUI, pick the 'auto_...' entry from the alignment "
                      "drop-down and press Get Data to review it.")

        self._run_bg(job, done,
                     busy_msg="Proposing alignment from firing vs atlas structure...")

    def _load_channel_table(self) -> None:
        fn = self.folder / "channel_locations_all_shanks.json"
        if not fn.exists():
            return
        with open(fn) as f:
            data = json.load(f)
        rows = [(k, v) for k, v in data.items() if k != "origin"]
        self.tbl_channels.setRowCount(len(rows))
        for r, (k, v) in enumerate(rows):
            self.tbl_channels.setItem(r, 0, QtWidgets.QTableWidgetItem(str(k)))
            self.tbl_channels.setItem(r, 1, QtWidgets.QTableWidgetItem(str(v.get("axial", ""))))
            self.tbl_channels.setItem(r, 2, QtWidgets.QTableWidgetItem(str(v.get("lateral", ""))))
            self.tbl_channels.setItem(r, 3, QtWidgets.QTableWidgetItem(str(v.get("brain_region", ""))))

    # ------------------------------------------------ Unit distribution plots
    def _plot_unit_distribution(self) -> None:
        if self.folder is None:
            self._log("Load a session folder first.")
            return
        folder = self.folder
        atlas_path = self.ed_atlas.text().strip() or None

        def job():
            return HistologyTab._compute_unit_distribution(folder, atlas_path)

        def done(data):
            if data is None:
                self._log("Unit distribution needs ALF cluster files "
                          "(clusters.channels.npy / clusters.depths.npy). Re-run the channel "
                          "map with 'Run ALF extraction first' checked.")
                return
            self._draw_unit_distribution(data)
            self._log(f"Plotted {data['n_units']} units across "
                      f"{len(data['region_order'])} region(s).")

        self._run_bg(job, done, busy_msg="Loading unit distribution...")

    @staticmethod
    def _compute_unit_distribution(folder, atlas_path):
        """Load per-unit depth/region/firing from the ALF products (worker thread)."""
        folder = Path(folder)
        need = ["clusters.channels.npy", "clusters.depths.npy", "channel_locations_all_shanks.json"]
        if not all((folder / f).exists() for f in need):
            return None
        ch = np.asarray(np.load(folder / "clusters.channels.npy")).astype(int)
        depths = np.asarray(np.load(folder / "clusters.depths.npy")).astype(float)
        n = int(min(len(ch), len(depths)))
        ch, depths = ch[:n], depths[:n]
        loc = json.loads((folder / "channel_locations_all_shanks.json").read_text())

        acr = np.empty(n, dtype=object)
        rid = np.full(n, -1, dtype=int)
        lat = np.full(n, np.nan)
        for c in range(n):
            info = loc.get(str(int(ch[c])))
            if info is None:
                acr[c] = "?"
                continue
            acr[c] = str(info.get("brain_region", "?")) or "?"
            rid[c] = int(info.get("brain_region_id", -1))
            lat[c] = float(info.get("lateral", np.nan))

        counts = np.ones(n, dtype=float)
        sf = folder / "spikes.clusters.npy"
        if sf.exists():
            sclu = np.asarray(np.load(sf))
            sclu = sclu[(sclu >= 0) & (sclu < n)].astype(int)
            if sclu.size:
                counts = np.bincount(sclu, minlength=n)[:n].astype(float)

        # Firing RATE (Hz) = spike count / recording duration (ALF spikes.times is in
        # seconds), so dot size compares units fairly. Falls back to raw count when the
        # spike-times file is absent (sizes are then proportional within the session).
        duration = None
        rates = counts.copy()
        stimes_f = folder / "spikes.times.npy"
        if stimes_f.exists():
            try:
                stimes = np.asarray(np.load(stimes_f), dtype=float)
                if stimes.size:
                    duration = float(np.nanmax(stimes) - np.nanmin(stimes))
                    if duration > 0:
                        rates = counts / duration
                    else:
                        duration = None
            except Exception:
                duration = None

        id2col = {}
        try:
            base = hatlas.resolve_atlas_path(atlas_path)
            st = hatlas.load_structure_tree(base / hatlas._STRUCTURE_FN)
            for _, r in st.iterrows():
                h = str(r["color_hex_triplet"])
                try:
                    id2col[int(r["id"])] = (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
                except (ValueError, IndexError):
                    pass
        except Exception:
            pass

        regions: Dict[str, dict] = {}
        for c in range(n):
            d = regions.setdefault(str(acr[c]), {"count": 0, "rid": int(rid[c]), "depths": []})
            d["count"] += 1
            d["depths"].append(float(depths[c]))
        region_order = sorted(regions, key=lambda a: float(np.mean(regions[a]["depths"])))
        return {
            "n_units": n, "depths": depths, "lateral": lat, "rid": rid, "counts": counts,
            "acr": acr, "rates": rates, "duration": duration,
            "id2col": id2col, "regions": regions, "region_order": region_order,
        }

    @staticmethod
    def _style_plot(p, title: str) -> None:
        p.setTitle(title, color=(30, 30, 30), size="10pt")
        for axn in ("left", "bottom"):
            ax = p.getAxis(axn)
            ax.setPen(pg.mkPen((90, 90, 90)))
            ax.setTextPen(pg.mkPen((40, 40, 40)))

    # Qualitative, perceptually-distinct palette (tab20-style, no near-white). The
    # Allen CCF colours paint neighbouring midbrain nuclei near-identical pink, so a
    # categorical palette is what actually separates regions by eye.
    _REGION_PALETTE = [
        (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
        (148, 103, 189), (140, 86, 75), (227, 119, 194), (23, 190, 207),
        (188, 189, 34), (23, 90, 160), (174, 199, 232), (255, 152, 150),
        (152, 223, 138), (255, 187, 120), (197, 176, 213), (196, 156, 148),
    ]

    def _draw_unit_distribution(self, data) -> None:
        try:
            self.units_plot.clear()
        except Exception as exc:
            self._log(f"Could not reset unit plot: {exc}")
            return

        depths = np.asarray(data["depths"], float)
        lat = np.nan_to_num(np.asarray(data["lateral"]), nan=0.0)
        acr = np.asarray(data["acr"], dtype=object)
        rates = np.asarray(data.get("rates", data["counts"]), float)
        region_order = data["region_order"]
        has_rate = bool(data.get("duration"))
        unit_word = "Hz" if has_rate else "spikes"
        size_word = "firing rate" if has_rate else "spike count"

        # One distinct colour per region, used in BOTH panels so a region reads the
        # same everywhere; "?" (unmapped) stays neutral grey.
        pal = HistologyTab._REGION_PALETTE
        region_colors = {a: pal[i % len(pal)] for i, a in enumerate(region_order)}
        region_colors["?"] = (120, 120, 120)

        # Dot size encodes firing rate, log-scaled so a wide Hz range stays legible.
        rmax = float(np.nanmax(rates)) if rates.size else 1.0
        lrmax = float(np.log10(rmax + 1.0)) or 1.0

        def rate_to_size(r):
            return 6.0 + 16.0 * (np.log10(np.asarray(r, float) + 1.0) / lrmax)

        sizes = rate_to_size(rates)

        region_count = len(region_order)
        size_count = 1 + min(3, len(rates)) if rates.size else 0
        legend_cols = max(1, min(region_count + size_count, 12))
        leg = pg.LegendItem(
            offset=(0, 0),
            labelTextColor=(40, 40, 40),
            brush=pg.mkBrush(255, 255, 255, 0),
            pen=None,
            frame=False,
            colCount=legend_cols,
        )
        self.units_plot.addItem(leg, row=0, col=0, colspan=2)
        for a in region_order:
            c = region_colors.get(a, (120, 120, 120))
            swatch = pg.ScatterPlotItem(symbol="o", size=10, brush=pg.mkBrush(*c, 235),
                                        pen=pg.mkPen((40, 40, 40), width=0.4))
            leg.addItem(swatch, f"{a} ({data['regions'][a]['count']})")
        if rates.size:
            spacer = pg.ScatterPlotItem(symbol="o", size=0.1, pen=None,
                                        brush=pg.mkBrush(255, 255, 255, 0))
            leg.addItem(spacer, f"size: {size_word} ({unit_word})")
            refs = np.unique(np.round(np.nanpercentile(rates, [15, 55, 95]), 1))
            for r in refs:
                ref = pg.ScatterPlotItem(symbol="o", size=float(rate_to_size(r)),
                                         brush=pg.mkBrush(135, 135, 135, 230),
                                         pen=pg.mkPen((40, 40, 40), width=0.4))
                leg.addItem(ref, f"{r:g} {unit_word}")

        # Panel 1: spatial scatter (lateral vs depth), colour = region, size = rate.
        p1 = self.units_plot.addPlot(row=1, col=0)
        self._style_plot(p1, f"{data['n_units']} units   (colour = region, size = {size_word})")
        p1.setLabel("left", "depth (um)")
        p1.setLabel("bottom", "lateral (um)")
        p1.showGrid(x=False, y=True, alpha=0.15)
        jitter = (np.random.default_rng(0).random(len(lat)) - 0.5) * 14.0
        brushes = [pg.mkBrush(*region_colors.get(str(acr[i]), (120, 120, 120)), 215)
                   for i in range(len(depths))]
        p1.addItem(pg.ScatterPlotItem(
            x=(lat + jitter).tolist(), y=depths.tolist(),
            size=sizes.tolist(), brush=brushes, pen=pg.mkPen((40, 40, 40), width=0.4)))

        # Panel 2: units per region (horizontal bars), same region colours.
        order = region_order
        if order:
            p2 = self.units_plot.addPlot(row=1, col=1)
            self._style_plot(p2, "units per region")
            p2.setLabel("bottom", "# units")
            ypos = np.arange(len(order))
            widths = np.array([data["regions"][a]["count"] for a in order], float)
            p2.addItem(pg.BarGraphItem(
                x0=np.zeros(len(order)), x1=widths, y=ypos, height=0.7,
                brushes=[pg.mkBrush(*region_colors.get(a, (120, 120, 120)), 235) for a in order],
                pen=pg.mkPen((70, 70, 70), width=0.5)))
            p2.getAxis("left").setTicks([[(i, a) for i, a in enumerate(order)]])
            wmax = float(widths.max()) if widths.size else 1.0
            for i, wv in enumerate(widths):
                t = pg.TextItem(str(int(wv)), color=(40, 40, 40), anchor=(0, 0.5))
                t.setPos(wv + wmax * 0.02, i)
                p2.addItem(t)
            p2.setXRange(0, wmax * 1.15)

    def _launch_ibl(self) -> None:
        if self.folder is None:
            self._log("Load a session folder first.")
            return
        if not (self.folder / "channels.localCoordinates.npy").exists():
            self._log("Tip: run the Channel map step first so the IBL GUI can auto-load "
                      "(it needs channels.localCoordinates.npy + xyz_picks).")
        try:
            ibl_launch.launch_ibl_gui(self.folder, **self._ibl_kwargs())
            self._log("IBL GUI launching in a separate window; it will auto-load this session.")
        except Exception as exc:
            self._log(f"Could not launch IBL GUI: {exc}")

    # ----------------------------------------------------------- settings
    def _persist_settings(self) -> None:
        self.settings.setValue("histology/last_folder", self.ed_folder.text().strip())
        self.settings.setValue("histology/raw_path", self.ed_raw.text().strip())
        self.settings.setValue("histology/atlas_path", self.ed_atlas.text().strip())
        self.settings.setValue("histology/ks_path", self.ed_ks.text().strip())
        self.settings.setValue("histology/ephys_path", self.ed_ephys.text().strip())
        self.settings.setValue("histology/iblapps_path", self.ed_iblapps.text().strip())
        self.settings.setValue("histology/python_exe", self.ed_pyexe.text().strip())
        self._log("Saved histology paths.")

    def _restore_settings(self) -> None:
        self.ed_folder.setText(str(self.settings.value("histology/last_folder", "")))
        self.ed_raw.setText(str(self.settings.value("histology/raw_path", "")))
        self.ed_atlas.setText(str(self.settings.value("histology/atlas_path", hatlas.DEFAULT_ATLAS_PATH)))
        self.ed_ks.setText(str(self.settings.value("histology/ks_path", "")))
        self.ed_ephys.setText(str(self.settings.value("histology/ephys_path", "")))
        self.ed_iblapps.setText(str(self.settings.value("histology/iblapps_path", ibl_launch.DEFAULT_IBLAPPS_PATH)))
        self.ed_pyexe.setText(str(self.settings.value("histology/python_exe", "")))
