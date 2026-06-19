"""Histology slice image preprocessing.

Native port of ``create_slice_images.m`` plus the small reorientation helpers
(``rotate_center_slices.m``, ``flip_slices.m``, ``reorder_slices.m``):

* load multi-page / RGB TIFFs and downsample
* per-channel white balance and colourisation -> RGB
* segment individual slices on a slide and crop them
* rotate / centre / flip / reorder slices
* save ``slice_*.tif`` (the inputs to atlas matching)

The interactive parts (clicking slices on a slide, confirming contrast/colour)
live in the tab; this module holds the deterministic image operations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage as ndi

try:
    import tifffile
    _HAS_TIFFFILE = True
except Exception:  # pragma: no cover
    _HAS_TIFFFILE = False

from PIL import Image


# ---------------------------------------------------------------------------
# Loading / saving
# ---------------------------------------------------------------------------

def list_tiffs(folder: str | Path) -> List[Path]:
    """Natural-sorted list of TIFFs in a folder."""
    folder = Path(folder)
    files = list(folder.glob("*.tif")) + list(folder.glob("*.tiff"))
    return _natsort(files)


def _natsort(paths: Sequence[Path]) -> List[Path]:
    import re

    def key(p: Path):
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]

    return sorted(paths, key=key)


def load_image(path: str | Path) -> np.ndarray:
    """Load a TIFF as an array. RGB -> (H, W, 3); multi-channel -> (H, W, C)."""
    path = Path(path)
    if _HAS_TIFFFILE:
        arr = tifffile.imread(str(path))
    else:  # PIL fallback (handles RGB and multi-frame)
        im = Image.open(str(path))
        frames = []
        try:
            i = 0
            while True:
                im.seek(i)
                frames.append(np.array(im))
                i += 1
        except EOFError:
            pass
        arr = frames[0] if len(frames) == 1 else np.stack(frames, axis=-1)
    arr = np.asarray(arr)
    # Normalise channel axis to last for multi-channel stacks like (C, H, W).
    if arr.ndim == 3 and arr.shape[0] <= 4 and arr.shape[0] < arr.shape[-1]:
        arr = np.moveaxis(arr, 0, -1)
    return arr


def downsample(image: np.ndarray, factor: float) -> np.ndarray:
    """Resize by 1/factor (nearest-ish, channel-aware)."""
    if factor == 1:
        return image
    zoom = [1.0 / factor, 1.0 / factor] + ([1.0] if image.ndim == 3 else [])
    return ndi.zoom(image, zoom, order=1)


def is_rgb(image: np.ndarray) -> bool:
    return image.ndim == 3 and image.shape[-1] == 3 and image.dtype == np.uint8


def save_slices(slices_rgb: Sequence[np.ndarray], save_dir: str | Path) -> List[Path]:
    """Write each RGB slice (float 0..1 or uint8) to ``slice_<n>.tif``."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for i, im in enumerate(slices_rgb, start=1):
        arr = im
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        fn = save_dir / f"slice_{i}.tif"
        if _HAS_TIFFFILE:
            tifffile.imwrite(str(fn), arr)
        else:
            Image.fromarray(arr).save(str(fn))
        out.append(fn)
    return out


# ---------------------------------------------------------------------------
# White balance / colourisation  (port of create_slice_images.m)
# ---------------------------------------------------------------------------

def estimate_channel_contrast(channel_image: np.ndarray) -> Tuple[float, float]:
    """Estimate [cmin, cmax] for a single channel via the AP histogram heuristic."""
    img = channel_image[channel_image > 0]
    if img.size == 0:
        return 0.0, 1.0
    max_v = float(img.max())
    hist, _ = np.histogram(img, bins=np.arange(0, max_v + 1))
    smoothed = ndi.uniform_filter1d(hist.astype(np.float64), size=50)
    deriv = np.concatenate([[0], np.diff(smoothed)])
    bg_down = int(np.argmin(deriv))
    pos = np.flatnonzero(deriv[bg_down:] > 0)
    bg_signal_min = (pos[0] + bg_down) if pos.size else bg_down
    rel = np.argmax(smoothed[bg_signal_min:]) if bg_signal_min < len(smoothed) else 0
    signal_median = rel + bg_signal_min
    cutoff = smoothed[min(signal_median, len(smoothed) - 1)] * 0.01
    below = np.flatnonzero(smoothed[signal_median:] < cutoff)
    signal_high = (below[0] + signal_median + 1) if below.size else len(smoothed)
    return float(bg_signal_min), float(signal_high)


def combine_channels_rgb(
    channels: Sequence[np.ndarray],
    contrasts: Sequence[Tuple[float, float]],
    colors: Sequence[Sequence[float]],
) -> np.ndarray:
    """Rescale each channel by its contrast, tint by its colour, and sum to RGB."""
    h, w = channels[0].shape[:2]
    rgb = np.zeros((h, w, 3), dtype=np.float64)
    for ch, (cmin, cmax), color in zip(channels, contrasts, colors):
        scaled = np.clip((ch.astype(np.float64) - cmin) / max(cmax - cmin, 1e-9), 0, 1)
        rgb += scaled[..., None] * np.asarray(color, dtype=np.float64)[None, None, :]
    return np.clip(rgb, 0, 1)


# ---------------------------------------------------------------------------
# Slice segmentation on a slide  (port of update_slide)
# ---------------------------------------------------------------------------

def segment_slices(slide_rgb: np.ndarray, min_slice: int = 1000) -> Tuple[np.ndarray, int]:
    """Find candidate slice objects on a slide. Returns (labels, n_objects)."""
    bw = np.nanmean(slide_rgb, axis=2) if slide_rgb.ndim == 3 else slide_rgb.astype(float)
    lo, hi = np.nanmin(bw), np.nanmax(bw)
    hist, edges = np.histogram(bw, bins=np.linspace(lo, hi, 100))
    deriv = np.concatenate([[0], np.diff(ndi.uniform_filter1d(hist.astype(float), 3))])
    bg_down = int(np.argmin(deriv))
    pos = np.flatnonzero(deriv[bg_down:] > 0)
    bg_signal_min = (pos[0] + bg_down) if pos.size else bg_down
    threshold = edges[min(bg_signal_min, len(edges) - 1)] * 0.5

    mask = ndi.binary_fill_holes(bw > threshold)
    labels, n = ndi.label(mask)
    # Drop objects smaller than min_slice.
    sizes = ndi.sum(np.ones_like(labels), labels, index=np.arange(1, n + 1))
    keep = np.where(sizes >= min_slice)[0] + 1
    out = np.zeros_like(labels)
    for new_id, old_id in enumerate(keep, start=1):
        out[labels == old_id] = new_id
    return out, len(keep)


def extract_slice(slide_rgb: np.ndarray, label_mask: np.ndarray, label_id: int,
                  dilate: int = 30) -> np.ndarray:
    """Crop the bounding box of object ``label_id`` from a slide (port of extract_slice_rgb)."""
    obj = label_mask == label_id
    rows = np.any(obj, axis=1)
    cols = np.any(obj, axis=0)
    if dilate:
        rows = ndi.binary_dilation(rows, iterations=dilate)
        cols = ndi.binary_dilation(cols, iterations=dilate)
    r0, r1 = np.flatnonzero(rows)[[0, -1]]
    c0, c1 = np.flatnonzero(cols)[[0, -1]]
    return slide_rgb[r0:r1 + 1, c0:c1 + 1]


# ---------------------------------------------------------------------------
# Reorientation helpers
# ---------------------------------------------------------------------------

def rotate_center(image: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate a slice image about its centre (keeps shape, fills with 0)."""
    return ndi.rotate(image, angle_deg, axes=(1, 0), reshape=True, order=1, mode="constant")


def flip(image: np.ndarray, horizontal: bool = True) -> np.ndarray:
    """Flip a slice horizontally (left/right) or vertically."""
    return image[:, ::-1] if horizontal else image[::-1]


def reorder(slices: List, order: Sequence[int]) -> List:
    """Reorder a list of slices by ``order`` (list of 0-based indices)."""
    return [slices[i] for i in order]
