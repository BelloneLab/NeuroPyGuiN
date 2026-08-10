"""``neuropyguin histology ...`` - everything the Histology tab can do headlessly.

The tab walks a session through five stages:

    Preprocess  raw slide images  -> per-slice RGB images saved in the session
    Match       pick the Allen CCF plane per slice -> histology_ccf.mat
    Align       warp atlas <-> histology           -> atlas2histology_tform.mat
    Trace       draw probe tracks, sample regions  -> probe_ccf.mat (+ CSV)
    IBL         ALF extraction, xyz_picks, channel map, alignment proposal

Point-and-click stages (drawing probe tracks, nudging control points) have no
sensible non-interactive equivalent, so the CLI covers the automatic paths:
slice preparation, silhouette-based AP matching, automatic affine alignment,
CSV export of whatever has been saved, and the whole IBL bridge. Commands that
require the interactive step say so and point at the GUI.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ._common import (
    CommandError,
    Reporter,
    format_table,
    require_dir,
    setting_str,
)


#: Sidecar written next to ``histology_ccf.mat`` holding the per-slice planes.
MATCH_SPECS_FILENAME = "histology_match_specs.json"


def _settings_default(key: str, fallback: str = "") -> str:
    """Read one histology path from the GUI settings, falling back cleanly."""
    try:
        from ._common import open_settings

        return setting_str(open_settings(), key, fallback)
    except Exception:
        return fallback


def _atlas_path(args) -> Optional[str]:
    """Resolve the Allen CCF folder from the flag, the GUI settings, or the default."""
    if getattr(args, "atlas", None):
        return str(args.atlas)
    stored = _settings_default("histology/atlas_path", "")
    return stored or None


def _load_atlas(args):
    """Open the Allen CCF volumes, turning a missing download into a clean error."""
    from ..histology.atlas import AllenCCFAtlas

    try:
        return AllenCCFAtlas(_atlas_path(args))
    except FileNotFoundError as exc:
        raise CommandError(str(exc)) from exc


def _ibl_kwargs(args) -> Dict[str, Any]:
    """Build the keyword arguments the IBL bridge launcher expects."""
    return {
        "ibl_python": getattr(args, "ibl_python", None) or _settings_default("histology/python_exe") or None,
        "iblapps_path": getattr(args, "iblapps", None) or _settings_default("histology/iblapps_path") or None,
    }


def _run_bridge(args, bridge_args: List[str], reporter: Reporter) -> int:
    """Run one ``ibl_bridge`` subcommand in the configured IBL environment."""
    from ..histology import ibl_launch

    kwargs = _ibl_kwargs(args)
    code, _output = ibl_launch.run_bridge(bridge_args, log=reporter.log, **kwargs)
    return int(code)


# ---------------------------------------------------------------------------
# Session status and slice preparation
# ---------------------------------------------------------------------------


def cmd_status(args, reporter: Reporter) -> int:
    """Report which histology artefacts exist for a session folder."""
    from ..histology import slice_prep

    folder = require_dir(args.folder, "session folder")
    artefacts = [
        "histology_ccf.mat",
        "atlas2histology_tform.mat",
        "probe_ccf.mat",
        MATCH_SPECS_FILENAME,
        "channels.localCoordinates.npy",
        "clusters.channels.npy",
        "channel_locations_all_shanks.json",
        "alignment_report.md",
    ]
    rows = [
        {"artefact": name, "present": (folder / name).exists(), "path": str(folder / name)}
        for name in artefacts
    ]
    slices = slice_prep.list_saved_slices(folder)
    picks = sorted(p.name for p in folder.glob("xyz_picks*.json"))
    alignments = sorted(p.name for p in folder.glob("prev_alignments*.json"))

    reporter.emit(
        {
            "folder": str(folder),
            "n_saved_slices": len(slices),
            "xyz_picks": picks,
            "saved_alignments": alignments,
            "artefacts": rows,
        },
        text=[
            f"Session: {folder}",
            f"Saved slices: {len(slices)}",
            f"xyz_picks   : {', '.join(picks) or '(none)'}",
            f"alignments  : {', '.join(alignments) or '(none)'}",
            "",
            *format_table(rows, ["present", "artefact"]),
        ],
    )
    return 0


def cmd_prep(args, reporter: Reporter) -> int:
    """Turn raw slide images into per-slice RGB images inside the session folder.

    Each raw image is downsampled, converted to RGB (single-channel stacks are
    contrast-stretched and tinted with the default channel colours), segmented
    into slice objects, and each object is cropped out. The results are saved
    with :func:`slice_prep.save_slices`, which is exactly what the Preprocess
    stage's "Save slices" button writes.
    """
    import numpy as np

    from ..histology import slice_prep

    raw_folder = require_dir(args.raw_folder, "raw image folder")
    out_folder = Path(args.output or args.raw_folder).expanduser()
    out_folder.mkdir(parents=True, exist_ok=True)

    images = slice_prep.list_raw_images(raw_folder)
    if not images:
        raise CommandError(f"No readable images found in {raw_folder}")
    reporter.log(f"Found {len(images)} raw image(s) in {raw_folder}")

    # Default channel tints, matching the Preprocess stage: green, red, blue.
    default_colors = [(0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)]

    collected: List[np.ndarray] = []
    per_image: List[Dict[str, Any]] = []
    for path in images:
        image = slice_prep.load_image(path)
        if args.downsample and args.downsample != 1:
            image = slice_prep.downsample(image, args.downsample)

        if slice_prep.is_rgb(image):
            slide = np.clip(image.astype(np.float64) / max(float(image.max()), 1.0), 0, 1)
        else:
            channels = [image] if image.ndim == 2 else [image[..., i] for i in range(image.shape[-1])]
            contrasts = [slice_prep.estimate_channel_contrast(ch) for ch in channels]
            colors = (default_colors * 3)[: len(channels)]
            slide = slice_prep.combine_channels_rgb(channels, contrasts, colors)

        labels, n_objects = slice_prep.segment_slices(slide, min_slice=args.min_slice)
        for label_id in range(1, n_objects + 1):
            collected.append(slice_prep.extract_slice(slide, labels, label_id, dilate=args.dilate))
        per_image.append({"image": str(path), "slices": int(n_objects)})
        reporter.log(f"  {path.name}: {n_objects} slice(s)")

    if not collected:
        raise CommandError(
            "Segmentation found no slice objects. Try lowering --min-slice or check the input images."
        )

    if args.dry_run:
        reporter.emit({"dry_run": True, "total_slices": len(collected), "images": per_image})
        return 0

    written = slice_prep.save_slices(collected, out_folder)
    reporter.emit(
        {
            "session_folder": str(out_folder),
            "total_slices": len(written),
            "images": per_image,
            "files": [str(p) for p in written],
        },
        text=[f"Saved {len(written)} slice image(s) to {out_folder}"],
    )
    return 0


# ---------------------------------------------------------------------------
# Match: pick the atlas plane per slice
# ---------------------------------------------------------------------------


def _load_slices(folder: Path):
    """Load the saved per-slice images, raising a helpful error when absent."""
    from ..histology import slice_prep

    paths = slice_prep.list_saved_slices(folder)
    if not paths:
        raise CommandError(
            f"No saved slices found in {folder}. Run 'histology prep' (or the Preprocess stage) first."
        )
    return [(p, slice_prep.load_image(p)) for p in paths]


def _write_match_specs(folder: Path, specs: List[Optional[Dict[str, Any]]]) -> Path:
    """Persist per-slice match planes in the sidecar the GUI reads on load."""
    import numpy as np

    payload = []
    for spec in specs:
        if spec is None:
            payload.append(None)
            continue
        payload.append(
            {
                "slice_point": np.asarray(spec["slice_point"], float).ravel().tolist(),
                "camera_vector": np.asarray(spec["camera_vector"], float).ravel().tolist(),
                "ap": int(spec.get("ap", 0)),
                "lr": int(spec.get("lr", 0)),
                "si": int(spec.get("si", 0)),
                "mode": str(spec.get("mode", "TV")),
            }
        )
    target = folder / MATCH_SPECS_FILENAME
    target.write_text(json.dumps({"specs": payload}, indent=2) + "\n", encoding="utf-8")
    return target


def _read_match_specs(folder: Path) -> List[Optional[Dict[str, Any]]]:
    """Load the per-slice match planes written by :func:`_write_match_specs`."""
    import numpy as np

    path = folder / MATCH_SPECS_FILENAME
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    specs: List[Optional[Dict[str, Any]]] = []
    for entry in data.get("specs") or []:
        if not entry:
            specs.append(None)
            continue
        specs.append(
            {
                "slice_point": np.asarray(entry["slice_point"], float),
                "camera_vector": np.asarray(entry["camera_vector"], float),
                "ap": int(entry.get("ap", 0)),
                "lr": int(entry.get("lr", 0)),
                "si": int(entry.get("si", 0)),
                "mode": str(entry.get("mode", "TV")),
            }
        )
    return specs


def cmd_match(args, reporter: Reporter) -> int:
    """Predict the coronal AP plane for every saved slice from its silhouette.

    Runs the same deterministic shape matcher the "Auto match" button uses, then
    writes the resulting planes to the match-specs sidecar so the GUI (and
    ``histology build-ccf``) can pick them up.
    """
    from ..histology import atlas as hatlas
    from ..histology.matching import automatch_coronal_ap

    folder = require_dir(args.folder, "session folder")
    atlas = _load_atlas(args)
    slices = _load_slices(folder)
    reporter.log(f"Matching {len(slices)} slice(s) against the atlas...")

    existing = _read_match_specs(folder)
    specs: List[Optional[Dict[str, Any]]] = list(existing) + [None] * max(0, len(slices) - len(existing))
    rows: List[Dict[str, Any]] = []
    center_ap: Optional[int] = args.center_ap

    for index, (path, image) in enumerate(slices):
        if specs[index] is not None and not args.overwrite:
            rows.append({"slice": index, "ap": specs[index].get("ap"), "score": "", "note": "kept"})
            continue
        try:
            result = automatch_coronal_ap(
                image,
                atlas,
                center_ap=center_ap,
                search_radius=args.search_radius,
                ap_step=args.ap_step,
                workers=args.workers,
            )
        except Exception as exc:  # noqa: BLE001 - one bad slice must not stop the batch
            reporter.warn(f"Slice {index} ({path.name}): {exc}")
            rows.append({"slice": index, "ap": "", "score": "", "note": str(exc)})
            continue

        ap = int(result["ap"])
        specs[index] = {
            "slice_point": hatlas.coronal_slice_point(ap, atlas),
            "camera_vector": hatlas.coronal_camera_vector(args.tilt_lr, args.tilt_si),
            "ap": ap,
            "lr": int(args.tilt_lr),
            "si": int(args.tilt_si),
            "mode": args.mode,
        }
        rows.append(
            {
                "slice": index,
                "ap": ap,
                "score": round(float(result.get("score", 0.0)), 4),
                "note": f"confidence {float(result.get('confidence', 0.0)):.2f}",
            }
        )
        reporter.log(f"  slice {index} ({path.name}): AP {ap}")
        if args.sequential:
            # Slices in a series are close together, so use the previous result
            # as the prior for the next search. This is both faster and steadier.
            center_ap = ap

    written = _write_match_specs(folder, specs)
    reporter.emit(
        {"folder": str(folder), "match_specs": str(written), "slices": rows},
        text=format_table(rows, ["slice", "ap", "score", "note"]),
    )
    return 0


def cmd_build_ccf(args, reporter: Reporter) -> int:
    """Write ``histology_ccf.mat`` (and its CSV) from the saved match planes."""
    from ..histology import io_formats
    from ..histology.matching import build_histology_ccf

    folder = require_dir(args.folder, "session folder")
    atlas = _load_atlas(args)
    specs = _read_match_specs(folder)
    if not specs or any(spec is None for spec in specs):
        missing = [i for i, spec in enumerate(specs) if spec is None]
        raise CommandError(
            "Every slice needs an assigned plane first. "
            + (f"Missing: {missing}. " if missing else "")
            + "Run 'histology match' or assign the planes in the GUI."
        )

    reporter.log(f"Grabbing {len(specs)} atlas plane(s) at spacing {args.spacing}...")
    histology_ccf = build_histology_ccf(atlas, specs, spacing=args.spacing)
    out_path = folder / "histology_ccf.mat"
    io_formats.save_histology_ccf(out_path, histology_ccf)
    csv_path = io_formats.export_histology_ccf_csv(folder, histology_ccf)
    reporter.emit(
        {"histology_ccf": str(out_path), "csv": str(csv_path), "n_slices": len(histology_ccf)},
        text=[f"Wrote {out_path}", f"Wrote {csv_path}"],
    )
    return 0


# ---------------------------------------------------------------------------
# Align: atlas <-> histology affine
# ---------------------------------------------------------------------------


def cmd_align(args, reporter: Reporter) -> int:
    """Compute the atlas-to-histology affine for every slice automatically.

    Uses the shape-constrained auto-aligner (run in a child process so a
    pathological slice cannot abort the run) and writes the per-slice 3x3
    transforms to ``atlas2histology_tform.mat``.
    """
    import numpy as np

    from ..histology import io_formats
    from ..histology.alignment import auto_align_isolated

    folder = require_dir(args.folder, "session folder")
    ccf_path = folder / "histology_ccf.mat"
    if not ccf_path.exists():
        raise CommandError(
            f"Missing {ccf_path}. Run 'histology match' then 'histology build-ccf' first."
        )

    slices = _load_slices(folder)
    histology_ccf = io_formats.load_histology_ccf(ccf_path)
    count = min(len(slices), len(histology_ccf))
    if count == 0:
        raise CommandError("No slices to align.")

    reporter.log(f"Auto-aligning {count} slice(s)...")
    tforms: List[np.ndarray] = []
    rows: List[Dict[str, Any]] = []
    for index in range(count):
        _path, image = slices[index]
        gray = image.mean(axis=2) if image.ndim == 3 else image
        transform, status = auto_align_isolated(
            gray, np.asarray(histology_ccf[index]["tv_slices"]), timeout=args.timeout
        )
        tforms.append(np.asarray(transform, dtype=float))
        rows.append({"slice": index, "status": status or "ok"})
        reporter.log(f"  slice {index}: {status or 'ok'}")

    out_path = folder / "atlas2histology_tform.mat"
    io_formats.save_tforms(out_path, tforms)
    reporter.emit(
        {"tform_file": str(out_path), "n_slices": len(tforms), "slices": rows},
        text=[f"Wrote {out_path}", "", *format_table(rows, ["slice", "status"])],
    )
    return 0


def cmd_export(args, reporter: Reporter) -> int:
    """Export the saved probe and atlas results to CSV files."""
    from ..histology import io_formats

    folder = require_dir(args.folder, "session folder")
    written: Dict[str, Any] = {}

    ccf_path = folder / "histology_ccf.mat"
    if ccf_path.exists():
        histology_ccf = io_formats.load_histology_ccf(ccf_path)
        written["histology_ccf_csv"] = str(io_formats.export_histology_ccf_csv(folder, histology_ccf))

    probe_path = folder / "probe_ccf.mat"
    if probe_path.exists():
        # export_probe_ccf_csv needs the full probe dicts; the saved points are
        # enough to re-emit the per-probe coordinate tables.
        points = io_formats.load_probe_ccf_points(probe_path)
        probes = [{"points": p} for p in points]
        exported = io_formats.export_probe_ccf_csv(folder, probes)
        written["probe_ccf_csv"] = {k: str(v) for k, v in exported.items()}

    if not written:
        raise CommandError(
            f"Nothing to export from {folder}: neither histology_ccf.mat nor probe_ccf.mat exists."
        )
    reporter.emit({"folder": str(folder), "written": written})
    return 0


# ---------------------------------------------------------------------------
# IBL bridge
# ---------------------------------------------------------------------------


def cmd_alf(args, reporter: Reporter) -> int:
    """Extract the ALF files the histology and IBL alignment steps consume."""
    folder = require_dir(args.folder, "session folder")
    ks_dir = args.ks or _settings_default("histology/ks_path")
    ephys = args.ephys or _settings_default("histology/ephys_path")
    if not ks_dir or not ephys:
        raise CommandError("ALF extraction needs both --ks and --ephys (or the GUI's saved paths).")

    bridge_args = ["extract_alf", str(ks_dir), str(ephys), str(folder)]
    if args.rms:
        bridge_args.append("--rms")
    if args.mode:
        bridge_args += ["--mode", args.mode]

    code = _run_bridge(args, bridge_args, reporter)
    reporter.emit({"folder": str(folder), "exit_code": code, "ok": code == 0})
    return code


def cmd_xyz(args, reporter: Reporter) -> int:
    """Convert ``probe_ccf.mat`` into per-shank ``xyz_picks`` JSON files.

    The fast path runs in-process and needs no IBL environment. With
    ``--mode ibl`` (or on a fast-path failure) the exact IBL surface-intersection
    extractor is invoked through the bridge instead.
    """
    folder = require_dir(args.folder, "session folder")
    probe_ccf = folder / "probe_ccf.mat"
    if not probe_ccf.exists():
        raise CommandError(f"Missing {probe_ccf}. Trace the probe tracks in the GUI first.")

    if args.mode in {"fast", "auto"}:
        from ..histology import ibl_bridge

        try:
            written = ibl_bridge.compute_xyz_picks(probe_ccf, folder, res=args.res, mode="fast")
            reporter.emit(
                {"mode": "fast", "files": [str(p) for p in written], "count": len(written)},
                text=[f"Wrote {len(written)} xyz_picks file(s)", *[f"  {p}" for p in written]],
            )
            return 0
        except Exception as exc:  # noqa: BLE001
            if args.mode == "fast":
                raise CommandError(f"Fast xyz_picks failed: {exc}") from exc
            reporter.warn(f"Fast xyz_picks failed ({exc}); falling back to the IBL extractor.")

    code = _run_bridge(args, ["xyz_picks", str(folder), "--mode", "ibl"], reporter)
    files = sorted(str(p) for p in folder.glob("xyz_picks*.json"))
    reporter.emit({"mode": "ibl", "exit_code": code, "files": files})
    return code


def cmd_channels(args, reporter: Reporter) -> int:
    """Compute per-channel CCF locations for every shank plus the merged file."""
    folder = require_dir(args.folder, "session folder")
    bridge_args = ["channels", str(folder), "--alignment", args.alignment]
    ks_dir = args.ks or _settings_default("histology/ks_path")
    if ks_dir:
        bridge_args += ["--ks", str(ks_dir)]

    code = _run_bridge(args, bridge_args, reporter)
    merged = folder / "channel_locations_all_shanks.json"
    reporter.emit(
        {
            "folder": str(folder),
            "alignment": args.alignment,
            "exit_code": code,
            "merged_file": str(merged) if merged.exists() else "",
        }
    )
    return code


def cmd_pipeline(args, reporter: Reporter) -> int:
    """Run the whole IBL bridge pipeline (optional ALF extraction + channel map)."""
    folder = require_dir(args.folder, "session folder")
    bridge_args = ["all", str(folder), "--alignment", args.alignment]
    ks_dir = args.ks or _settings_default("histology/ks_path")
    ephys = args.ephys or _settings_default("histology/ephys_path")
    if ks_dir:
        bridge_args += ["--ks", str(ks_dir)]
    if args.extract:
        if not (ks_dir and ephys):
            raise CommandError("--extract needs both --ks and --ephys (or the GUI's saved paths).")
        bridge_args += ["--ephys", str(ephys)]
        if args.rms:
            bridge_args.append("--rms")

    code = _run_bridge(args, bridge_args, reporter)
    reporter.emit({"folder": str(folder), "exit_code": code, "ok": code == 0})
    return code


def cmd_propose(args, reporter: Reporter) -> int:
    """Propose a depth alignment by scoring firing against atlas structure."""
    folder = require_dir(args.folder, "session folder")
    if not (folder / "clusters.channels.npy").exists():
        raise CommandError(
            f"Missing ALF cluster files in {folder}. Run 'histology alf' (or 'histology pipeline --extract') first."
        )
    bridge_args = ["propose_align", str(folder)]
    atlas = _atlas_path(args)
    if atlas:
        bridge_args += ["--atlas", str(atlas)]

    code = _run_bridge(args, bridge_args, reporter)
    report = folder / "alignment_report.md"
    reporter.emit(
        {
            "folder": str(folder),
            "exit_code": code,
            "report": str(report) if report.exists() else "",
        },
        text=[
            f"Alignment proposal exit code: {code}",
            f"Report: {report}" if report.exists() else "No report written.",
            "In the IBL GUI, pick the 'auto_...' entry from the alignment drop-down.",
        ],
    )
    return code


def cmd_finalize(args, reporter: Reporter) -> int:
    """Rebuild the channel regions from the latest saved IBL GUI alignments."""
    folder = require_dir(args.folder, "session folder")
    saved = sorted(folder.glob("prev_alignments*.json"))
    if not saved:
        raise CommandError(
            "No saved IBL alignments found. Align each shank in the IBL GUI and press Upload first."
        )
    reporter.log(f"Finalizing from {len(saved)} saved alignment file(s).")
    args.alignment = "latest"
    return cmd_channels(args, reporter)


def cmd_gui(args, reporter: Reporter) -> int:
    """Launch the offline IBL ephys-alignment GUI on a session folder."""
    from ..histology import ibl_launch

    folder = require_dir(args.folder, "session folder")
    process = ibl_launch.launch_ibl_gui(
        folder, log=reporter.log, auto_load=not args.no_auto_load, **_ibl_kwargs(args)
    )
    if args.wait:
        code = process.wait()
        reporter.emit({"pid": process.pid, "exit_code": code})
        return int(code)
    reporter.emit(
        {"pid": process.pid, "folder": str(folder), "log": str(folder / "ibl_gui.log")},
        text=[f"IBL alignment GUI started (pid {process.pid}) on {folder}"],
    )
    return 0


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def cmd_atlas_info(args, reporter: Reporter) -> int:
    """Report whether the Allen CCF atlas files are present and usable."""
    from ..histology.atlas import atlas_files_present, resolve_atlas_path

    path = resolve_atlas_path(_atlas_path(args))
    present = atlas_files_present(str(path))
    payload: Dict[str, Any] = {"atlas_path": str(path), "files_present": present}
    if present:
        atlas = _load_atlas(args)
        payload["shape_ap_dv_ml"] = list(atlas.shape)
        payload["n_structures"] = int(len(atlas.structure_tree))
    else:
        payload["hint"] = "Download the Allen CCF volumes from https://osf.io/fv7ed/overview"
    reporter.emit(payload)
    return 0 if present else 1


def cmd_hardware(args, reporter: Reporter) -> int:
    """Report the acceleration available to the histology matching code."""
    from ..histology import acceleration

    reporter.emit(
        {
            "summary": acceleration.hardware_summary(),
            "numba": acceleration.numba_available(),
            "cuda": acceleration.cuda_available(),
            "cuda_device": acceleration.cuda_device_name(),
            "workers": acceleration.auto_worker_count(),
        },
        text=[acceleration.hardware_summary()],
    )
    return 0


# ---------------------------------------------------------------------------
# Parser wiring
# ---------------------------------------------------------------------------


def _add_atlas_option(parser) -> None:
    """Attach the shared ``--atlas`` option to a subcommand parser."""
    parser.add_argument("--atlas", help="Allen CCF folder (default: the GUI setting).")


def _add_ibl_options(parser) -> None:
    """Attach the shared IBL environment options to a subcommand parser."""
    parser.add_argument("--ibl-python", dest="ibl_python", help="Python executable of the IBL environment.")
    parser.add_argument("--iblapps", help="Path to the iblapps checkout.")


def register(subparsers) -> None:
    """Attach the ``histology`` command group to the top level parser."""
    group = subparsers.add_parser(
        "histology",
        help="Slice preparation, atlas matching, alignment and the IBL bridge.",
        description="Histology tab equivalents. Interactive stages (probe tracing, manual "
        "control points) stay in the GUI; everything automatic is available here.",
    )
    commands = group.add_subparsers(dest="command", required=True)

    p = commands.add_parser("status", help="Show which histology artefacts a session has.")
    p.add_argument("folder", help="Histology session folder.")
    p.set_defaults(func=cmd_status)

    p = commands.add_parser("prep", help="Segment raw slide images into per-slice images.")
    p.add_argument("raw_folder", help="Folder holding the raw slide images.")
    p.add_argument("-o", "--output", metavar="DIR", help="Session folder to save slices into.")
    p.add_argument("--downsample", type=float, default=1.0, help="Downsample factor (default: 1 = none).")
    p.add_argument("--min-slice", type=int, default=1000, help="Minimum object size in pixels (default: 1000).")
    p.add_argument("--dilate", type=int, default=30, help="Bounding-box padding in pixels (default: 30).")
    p.add_argument("--dry-run", action="store_true", help="Report the segmentation without saving.")
    p.set_defaults(func=cmd_prep)

    p = commands.add_parser("match", help="Predict the coronal AP plane for every saved slice.")
    p.add_argument("folder", help="Histology session folder.")
    p.add_argument("--center-ap", type=int, help="Prior AP position to search around.")
    p.add_argument("--search-radius", type=int, help="Search radius around the prior, in AP steps.")
    p.add_argument("--ap-step", type=int, default=10, help="Coarse AP search step (default: 10).")
    p.add_argument("--tilt-lr", type=float, default=0.0, help="Left/right tilt in degrees.")
    p.add_argument("--tilt-si", type=float, default=0.0, help="Superior/inferior tilt in degrees.")
    p.add_argument("--mode", default="TV", help="Atlas render mode recorded with the plane.")
    p.add_argument("--workers", type=int, help="Parallel workers (default: automatic).")
    p.add_argument("--sequential", action="store_true", help="Use each match as the prior for the next slice.")
    p.add_argument("--overwrite", action="store_true", help="Re-match slices that already have a plane.")
    _add_atlas_option(p)
    p.set_defaults(func=cmd_match)

    p = commands.add_parser("build-ccf", help="Write histology_ccf.mat from the matched planes.")
    p.add_argument("folder", help="Histology session folder.")
    p.add_argument("--spacing", type=int, default=1, help="Atlas sampling spacing (default: 1).")
    _add_atlas_option(p)
    p.set_defaults(func=cmd_build_ccf)

    p = commands.add_parser("align", help="Auto-align the atlas to each slice and save the transforms.")
    p.add_argument("folder", help="Histology session folder.")
    p.add_argument("--timeout", type=float, default=180.0, help="Per-slice alignment timeout (s).")
    p.set_defaults(func=cmd_align)

    p = commands.add_parser("export", help="Export saved histology and probe results to CSV.")
    p.add_argument("folder", help="Histology session folder.")
    p.set_defaults(func=cmd_export)

    p = commands.add_parser("alf", help="Extract the ALF files used by the alignment workflow.")
    p.add_argument("folder", help="Histology session folder.")
    p.add_argument("--ks", help="Kilosort output folder.")
    p.add_argument("--ephys", help="Raw ephys folder holding the AP binary.")
    p.add_argument("--mode", choices=["auto", "fast", "ibl"], help="Extractor to use (default: auto).")
    p.add_argument("--rms", action="store_true", help="Also compute the slow per-channel RMS/QC map.")
    _add_ibl_options(p)
    p.set_defaults(func=cmd_alf)

    p = commands.add_parser("xyz", help="Generate per-shank xyz_picks from probe_ccf.mat.")
    p.add_argument("folder", help="Histology session folder.")
    p.add_argument("--mode", choices=["auto", "fast", "ibl"], default="auto", help="Extractor to use.")
    p.add_argument("--res", type=int, default=10, help="Atlas resolution in um (default: 10).")
    _add_ibl_options(p)
    p.set_defaults(func=cmd_xyz)

    p = commands.add_parser("channels", help="Compute per-channel CCF locations.")
    p.add_argument("folder", help="Histology session folder.")
    p.add_argument("--alignment", choices=["original", "latest"], default="original", help="Track to use.")
    p.add_argument("--ks", help="Kilosort folder, so channel geometry is reused.")
    _add_ibl_options(p)
    p.set_defaults(func=cmd_channels)

    p = commands.add_parser("pipeline", help="Run the full IBL bridge pipeline for a session.")
    p.add_argument("folder", help="Histology session folder.")
    p.add_argument("--alignment", choices=["original", "latest"], default="original", help="Track to use.")
    p.add_argument("--ks", help="Kilosort output folder.")
    p.add_argument("--ephys", help="Raw ephys folder holding the AP binary.")
    p.add_argument("--extract", action="store_true", help="Run ALF extraction as part of the pipeline.")
    p.add_argument("--rms", action="store_true", help="Also compute the slow per-channel RMS/QC map.")
    _add_ibl_options(p)
    p.set_defaults(func=cmd_pipeline)

    p = commands.add_parser("propose", help="Propose a depth alignment from firing versus atlas structure.")
    p.add_argument("folder", help="Histology session folder.")
    _add_atlas_option(p)
    _add_ibl_options(p)
    p.set_defaults(func=cmd_propose)

    p = commands.add_parser("finalize", help="Rebuild channel regions from the latest IBL GUI alignments.")
    p.add_argument("folder", help="Histology session folder.")
    p.add_argument("--ks", help="Kilosort folder, so channel geometry is reused.")
    _add_ibl_options(p)
    p.set_defaults(func=cmd_finalize)

    p = commands.add_parser("gui", help="Launch the offline IBL ephys-alignment GUI.")
    p.add_argument("folder", help="Histology session folder.")
    p.add_argument("--wait", action="store_true", help="Block until the GUI exits.")
    p.add_argument("--no-auto-load", action="store_true", help="Open the stock GUI with a manual folder picker.")
    _add_ibl_options(p)
    p.set_defaults(func=cmd_gui)

    p = commands.add_parser("atlas-info", help="Check that the Allen CCF atlas files are available.")
    _add_atlas_option(p)
    p.set_defaults(func=cmd_atlas_info)

    p = commands.add_parser("hardware", help="Report numba/CUDA acceleration for slice matching.")
    p.set_defaults(func=cmd_hardware)
