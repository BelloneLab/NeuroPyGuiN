"""Histology localization for NeuroPyGuiN.

Native Python reimplementation of the AP_histology (petersaj) Neuropixels
localization workflow, plus an optional bridge to the IBL ephys-alignment GUI.

Pipeline stages (each in its own module):

    slice_prep   -> extract / clean / orient histology slice images
    matching     -> choose the Allen CCF plane for every slice  (histology_ccf)
    alignment    -> warp atlas <-> histology                    (atlas2histology_tform)
    tracing      -> draw probe tracks, sample regions           (probe_ccf)
    ibl_bridge   -> xyz_picks, ALF extraction, channel maps     (channel_locations_*)
    ibl_launch   -> launch the unmodified IBL alignment GUI

The shared atlas access lives in :mod:`atlas`, file IO in :mod:`io_formats`.
"""

from __future__ import annotations

__all__ = [
    "atlas",
    "io_formats",
    "slice_prep",
    "matching",
    "alignment",
    "tracing",
    "ibl_bridge",
    "ibl_launch",
]
