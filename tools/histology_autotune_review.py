"""Generate review overlays for histology auto-match/auto-align tuning.

This is an offline diagnostic helper: it does not write AP_histology output
files, only PNG/CSV artifacts for visual inspection.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from neuropyguin.histology import alignment, atlas as hatlas, matching, slice_prep


def _norm_rgb(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        v = arr.astype(float)
        finite = v[np.isfinite(v)]
        lo, hi = np.percentile(finite, [0.5, 99.5]) if finite.size else (0.0, 1.0)
        g = np.clip((np.nan_to_num(v) - lo) / max(float(hi - lo), 1e-9), 0, 1)
        return np.dstack([g, g, g])
    rgb = arr[..., :3].astype(float)
    if rgb.max(initial=0) > 1.5:
        rgb /= 255.0
    return np.clip(rgb, 0, 1)


def _resize_rgb(rgb: np.ndarray, width: int) -> Image.Image:
    h, w = rgb.shape[:2]
    height = max(1, int(round(h * width / max(1, w))))
    im = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    return im.resize((width, height), Image.Resampling.LANCZOS)


def _mask_rgb(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, bool)
    rgb = np.zeros((*mask.shape, 3), dtype=float)
    rgb[mask] = [0.0, 0.8, 1.0]
    return rgb


def _downscale_image(image: np.ndarray, max_width: int) -> tuple[np.ndarray, float]:
    arr = np.asarray(image)
    if arr.ndim < 2 or arr.shape[1] <= max_width:
        return arr, 1.0
    scale = float(max_width) / float(arr.shape[1])
    height = max(1, int(round(arr.shape[0] * scale)))
    im = Image.fromarray(arr.astype(np.uint8))
    return np.asarray(im.resize((max_width, height), Image.Resampling.LANCZOS)), scale


def _atlas_boundary_overlay(
    histology: np.ndarray,
    atlas_tv: np.ndarray,
    T: np.ndarray,
    atlas_av: np.ndarray | None = None,
) -> np.ndarray:
    rgb = _norm_rgb(histology)
    if atlas_av is not None:
        atlas_mask = alignment.atlas_boundaries(np.nan_to_num(atlas_av, nan=0.0))
    else:
        atlas_mask = matching.atlas_shape_mask(atlas_tv)
    warped = alignment.warp_atlas(
        atlas_mask.astype(float),
        T,
        rgb.shape[:2],
        nearest=True,
        use_cv2=False,
    ) > 0.5
    edge = warped ^ ndi.binary_erosion(warped)
    out = rgb.copy()
    out[edge] = [0.0, 1.0, 1.0]
    return out


def _panel(
    title: str,
    histology: np.ndarray,
    hist_mask: np.ndarray,
    atlas_overlay: np.ndarray,
    width: int,
) -> Image.Image:
    hist_thumb = _resize_rgb(_norm_rgb(histology), width)
    mask_thumb = _resize_rgb(_mask_rgb(hist_mask), width)
    ov_thumb = _resize_rgb(atlas_overlay, width)
    panel_h = max(hist_thumb.height, mask_thumb.height, ov_thumb.height) + 36
    panel = Image.new("RGB", (width * 3, panel_h), "white")
    draw = ImageDraw.Draw(panel)
    draw.text((8, 8), title, fill=(0, 0, 0))
    panel.paste(hist_thumb, (0, 36))
    panel.paste(mask_thumb, (width, 36))
    panel.paste(ov_thumb, (width * 2, 36))
    return panel


def _write_contact_sheet(panels: Iterable[Image.Image], out_path: Path) -> None:
    panels = list(panels)
    if not panels:
        return
    w = max(p.width for p in panels)
    h = sum(p.height for p in panels)
    sheet = Image.new("RGB", (w, h), "white")
    y = 0
    for panel in panels:
        sheet.paste(panel, (0, y))
        y += panel.height
    sheet.save(out_path)


def _candidate_sheet(
    image: np.ndarray,
    atlas_obj: hatlas.AllenCCFAtlas,
    aps: Iterable[int],
    out_path: Path,
    width: int,
) -> None:
    panels: list[Image.Image] = []
    small_image, _scale = _downscale_image(image, max_width=720)
    hist_mask = matching.histology_shape_mask(small_image)
    hist_box = _bbox(hist_mask)
    for ap in aps:
        sl = atlas_obj.grab_atlas_slice(
            hatlas.coronal_slice_point(ap, atlas_obj),
            hatlas.coronal_camera_vector(0, 0),
            spacing=3,
        )
        atlas_mask = matching.atlas_shape_mask(sl["tv_slices"])
        T = _bbox_tform(atlas_mask, hist_box)
        overlay = _atlas_boundary_overlay(small_image, sl["tv_slices"], T, sl.get("av_slices"))
        thumb = _resize_rgb(overlay, width)
        panel = Image.new("RGB", (width, thumb.height + 24), "white")
        draw = ImageDraw.Draw(panel)
        draw.text((5, 5), f"AP {ap}", fill=(0, 0, 0))
        panel.paste(thumb, (0, 24))
        panels.append(panel)
    cols = 4
    rows = (len(panels) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * width, rows * max(p.height for p in panels)), "white")
    for i, panel in enumerate(panels):
        row, col = divmod(i, cols)
        sheet.paste(panel, (col * width, row * panel.height))
    sheet.save(out_path)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    rows = np.flatnonzero(np.any(mask, axis=1))
    cols = np.flatnonzero(np.any(mask, axis=0))
    if rows.size == 0 or cols.size == 0:
        return None
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def _bbox_tform(atlas_mask: np.ndarray, hist_box: tuple[int, int, int, int] | None) -> np.ndarray:
    atlas_box = _bbox(atlas_mask)
    if atlas_box is None or hist_box is None:
        return np.eye(3)
    ar0, ar1, ac0, ac1 = atlas_box
    hr0, hr1, hc0, hc1 = hist_box
    aw, ah = max(1, ac1 - ac0), max(1, ar1 - ar0)
    hw, hh = max(1, hc1 - hc0), max(1, hr1 - hr0)
    sx = hw / aw
    sy = hh / ah
    ax = (ac0 + ac1 - 1) / 2.0
    ay = (ar0 + ar1 - 1) / 2.0
    hx = (hc0 + hc1 - 1) / 2.0
    hy = (hr0 + hr1 - 1) / 2.0
    return np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [hx - ax * sx, hy - ay * sy, 1.0]])


def run(
    raw_dir: Path,
    atlas_dir: Path,
    out_dir: Path,
    width: int = 360,
    candidates_only: bool = False,
    write_candidates: bool = True,
    center_ap: int | None = None,
    search_radius: int | None = None,
    atlas_to_histology_scale: float | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = slice_prep.list_raw_images(raw_dir)
    at = hatlas.AllenCCFAtlas(str(atlas_dir))
    rows: list[dict[str, object]] = []
    panels: list[Image.Image] = []

    for i, path in enumerate(paths, start=1):
        image = slice_prep.load_image(path)
        hist_mask = matching.histology_shape_mask(image)
        row: dict[str, object] = {"slice": i, "file": path.name}
        scale = atlas_to_histology_scale
        if scale is None:
            px = slice_prep.pixel_size_um(path)
            if px is not None and px > 0:
                scale = hatlas.CCF_VOXEL_UM / float(px)
        if candidates_only:
            ap = -1
            overlay = _norm_rgb(image)
            row.update({"ap": "", "score": "", "confidence": "", "scale": scale or "", "top": "", "align_status": "candidates-only"})
        else:
            try:
                result = matching.automatch_coronal_ap(
                    image,
                    at,
                    center_ap=center_ap,
                    search_radius=search_radius,
                    atlas_to_histology_scale=scale,
                )
                ap = int(result["ap"])
                sl = at.grab_atlas_slice(
                    hatlas.coronal_slice_point(ap, at),
                    hatlas.coronal_camera_vector(0, 0),
                    spacing=1,
                )
                T, status = alignment.auto_align_isolated(image, sl["tv_slices"], timeout=120)
                overlay = _atlas_boundary_overlay(image, sl["tv_slices"], T, sl.get("av_slices"))
                row.update(
                    {
                        "ap": ap,
                        "score": f"{float(result.get('score', 0.0)):.4f}",
                        "confidence": f"{float(result.get('confidence', 0.0)):.4f}",
                        "scale": f"{float(scale):.4f}" if scale else "",
                        "top": result.get("top", []),
                        "align_status": status,
                    }
                )
            except Exception as exc:
                ap = -1
                overlay = _norm_rgb(image)
                row.update({"ap": ap, "score": "", "confidence": "", "scale": scale or "", "top": "", "align_status": f"ERROR: {exc}"})
        rows.append(row)
        scale_txt = f" scale={row.get('scale')}" if row.get("scale") else ""
        title = f"{i}: {path.name}  AP={ap}  score={row['score']}  conf={row['confidence']}{scale_txt}"
        panel = _panel(title, image, hist_mask, overlay, width)
        panel.save(out_dir / f"slice_{i:02d}_overlay.png")
        panels.append(panel)
        if write_candidates:
            _candidate_sheet(
                image,
                at,
                range(250, 1001, 50),
                out_dir / f"slice_{i:02d}_candidate_ap_250_1000.png",
                max(220, width // 2),
            )

    with open(out_dir / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["slice", "file", "ap", "score", "confidence", "scale", "top", "align_status"],
        )
        writer.writeheader()
        writer.writerows(rows)
    _write_contact_sheet(panels, out_dir / "contact_sheet.png")
    print(out_dir)
    for row in rows:
        print(row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--atlas", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--width", type=int, default=360)
    parser.add_argument("--candidates-only", action="store_true")
    parser.add_argument("--no-candidates", action="store_true")
    parser.add_argument("--center-ap", type=int, default=None)
    parser.add_argument("--search-radius", type=int, default=None)
    parser.add_argument("--atlas-to-histology-scale", type=float, default=None)
    args = parser.parse_args()
    run(
        args.raw,
        args.atlas,
        args.out,
        width=args.width,
        candidates_only=args.candidates_only,
        write_candidates=not args.no_candidates,
        center_ap=args.center_ap,
        search_radius=args.search_radius,
        atlas_to_histology_scale=args.atlas_to_histology_scale,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
