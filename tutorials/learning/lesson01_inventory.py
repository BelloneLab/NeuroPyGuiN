"""Lesson 1: look at your dataset before loading it.

What this script does
---------------------
It walks through one dataset folder and, for every file, reports what is
inside: array shapes, table columns, metadata keys, sampling rates. It never
modifies anything and never loads a large file fully into memory.

Why this is step one
--------------------
You cannot write a correct loader for data you have not looked at. Most
analysis bugs in freely moving electrophysiology are not maths errors, they
are wrong assumptions about files: wrong sampling rate, frames counted in a
different unit, a tracking table with one extra row, a time column in
milliseconds instead of seconds. This script turns assumptions into facts.

How to use it in Spyder
-----------------------
1. Set DATASET_PATH below to your small dataset folder.
2. Run one cell at a time with Ctrl+Enter (cells start with ``# %%``),
   or the whole file with F5.
3. Open the Variable Explorer (top right) and inspect ``inventory``.
4. Read the printed report and answer the questions at the bottom.
"""

# %% USER SETTINGS: the only lines you normally edit.
from pathlib import Path

DATASET_PATH = Path(r"D:\path\to\your\small_dataset")  # <- EDIT ME
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "NeuroPyGuiN_output" / "learning"
MAX_TABLE_PREVIEW_ROWS = 3  # Rows shown when previewing a CSV/TSV table.
MAX_TEXT_PREVIEW_LINES = 5  # Lines shown when previewing a plain text file.


# %% LIBRARIES
import json  # Read .json files (often used for session metadata).

import numpy as np  # Arrays. np.load(mmap_mode="r") reads headers only.
import pandas as pd  # Tables with named columns (CSV, TSV, DeepLabCut h5).

# Optional libraries: the script still works if they are missing.
try:
    import h5py  # Low-level reader for .h5/.hdf5/.nwb files.
except ImportError:
    h5py = None
try:
    from scipy.io import whosmat  # Lists variables in a MATLAB .mat file.
except ImportError:
    whosmat = None


# %% HELPER FUNCTIONS: one small function per file type.
def human_size(n_bytes):
    """Convert a byte count to a readable string such as '12.3 MB'."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} PB"


def read_spikeglx_meta(path):
    """Read a SpikeGLX .meta file into a dictionary.

    A .meta file is plain text with one 'key=value' per line. It describes the
    .bin file of the same name: sampling rate, channel count, duration, etc.
    """
    meta = {}
    for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            meta[key.strip().lstrip("~")] = value.strip()
    return meta


def describe_npy(path):
    """Describe a NumPy array WITHOUT loading it.

    mmap_mode="r" maps the file instead of reading it, so even a 50 GB array
    costs no memory. allow_pickle=False refuses files that could run code.
    """
    arr = np.load(path, mmap_mode="r", allow_pickle=False)
    text = f"array shape={arr.shape} dtype={arr.dtype}"
    if arr.size and arr.ndim == 1 and arr.dtype.kind in "iuf":
        head = np.asarray(arr[: min(5, arr.size)])  # Only these values are read.
        text += f" first={head.tolist()}"
    return text


def describe_table(path):
    """Describe a CSV/TSV table: number of columns, their names, first rows."""
    sep = "\t" if path.suffix.lower() == ".tsv" else None  # None = auto-detect.
    preview = pd.read_csv(path, sep=sep, engine="python", nrows=MAX_TABLE_PREVIEW_ROWS)
    with open(path, "rb") as handle:  # Count rows by counting lines (cheap).
        n_lines = sum(1 for _ in handle)
    return (f"table ~{n_lines - 1} rows x {preview.shape[1]} cols, "
            f"columns={list(preview.columns)[:12]}\n{preview.to_string()}")


def describe_hdf5(path):
    """List the groups/datasets inside an HDF5 file (.h5, .hdf5, .nwb).

    DeepLabCut saves pose tracking as .h5; NWB files are also HDF5 inside.
    """
    if h5py is None:
        return "HDF5 file (install h5py to inspect)"
    lines = []

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset) and len(lines) < 25:
            lines.append(f"  {name}: shape={obj.shape} dtype={obj.dtype}")

    with h5py.File(path, "r") as handle:  # "r" = read-only.
        handle.visititems(visit)
    return "HDF5 datasets:\n" + "\n".join(lines)


def describe_mat(path):
    """List variables of a MATLAB .mat file without loading their contents."""
    if whosmat is None:
        return "MATLAB file (install scipy to inspect)"
    try:
        variables = whosmat(str(path))
    except NotImplementedError:  # MATLAB v7.3 files are actually HDF5.
        return describe_hdf5(path)
    return "MATLAB variables: " + ", ".join(f"{n} {s} {c}" for n, s, c in variables)


def describe_json(path):
    """Show the top-level keys of a JSON file."""
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(data, dict):
        return f"JSON keys={list(data)[:20]}"
    return f"JSON {type(data).__name__} of length {len(data)}"


def describe_meta(path):
    """Summarise the SpikeGLX fields that matter for analysis."""
    meta = read_spikeglx_meta(path)
    wanted = ("typeThis", "imSampRate", "niSampRate", "nSavedChans",
              "fileTimeSecs", "imDatPrb_type", "fileCreateTime")
    found = {k: meta[k] for k in wanted if k in meta}
    return f"SpikeGLX meta {found}"


def describe_bin(path):
    """A raw binary has no header; its .meta twin tells us how to read it."""
    meta_path = path.with_suffix(".meta")
    if not meta_path.is_file():
        return "raw binary, no .meta next to it (cannot interpret safely)"
    meta = read_spikeglx_meta(meta_path)
    rate = float(meta.get("imSampRate", meta.get("niSampRate", "nan")))
    n_chan = int(meta.get("nSavedChans", 0))
    if n_chan == 0 or not np.isfinite(rate):
        return "raw binary, .meta incomplete"
    n_samples = path.stat().st_size // (2 * n_chan)  # int16 = 2 bytes per value.
    return (f"raw int16 binary: {n_chan} channels x {n_samples} samples "
            f"@ {rate:.4f} Hz = {n_samples / rate:.2f} s")


def describe_text(path):
    """Show the first lines of a plain text file (e.g. CatGT edge files)."""
    with open(path, encoding="utf-8", errors="replace") as handle:
        lines = [next(handle, "").rstrip() for _ in range(MAX_TEXT_PREVIEW_LINES)]
    return "text:\n  " + "\n  ".join(line for line in lines if line)


# Map each file extension to the function that understands it.
DESCRIBERS = {
    ".npy": describe_npy, ".csv": describe_table, ".tsv": describe_table,
    ".h5": describe_hdf5, ".hdf5": describe_hdf5, ".nwb": describe_hdf5,
    ".mat": describe_mat, ".json": describe_json, ".meta": describe_meta,
    ".bin": describe_bin, ".txt": describe_text, ".py": describe_text,
}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov"}


def build_inventory(root):
    """Walk every file under ``root`` and return one row per file.

    Errors are caught per file and reported, so one unreadable file does not
    hide the description of all the others.
    """
    rows = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        suffix = path.suffix.lower()
        try:
            if suffix in DESCRIBERS:
                description = DESCRIBERS[suffix](path)
            elif suffix in VIDEO_EXTENSIONS:
                description = "video (not opened; we only need its frame timestamps)"
            else:
                description = "unknown type (tell me what produces it)"
        except Exception as error:  # Report and continue.
            description = f"COULD NOT READ: {type(error).__name__}: {error}"
        rows.append({
            "relative_path": str(path.relative_to(root)),
            "extension": suffix,
            "size": human_size(path.stat().st_size),
            "description": description,
        })
    return pd.DataFrame(rows)


# %% RUN: build the inventory and print it.
if not DATASET_PATH.is_dir():
    raise FileNotFoundError(f"Set DATASET_PATH at the top of the script. Not found: {DATASET_PATH}")

inventory = build_inventory(DATASET_PATH)  # Inspect this in the Variable Explorer.

print(f"\nDataset: {DATASET_PATH}\n{len(inventory)} files\n" + "=" * 70)
for _, row in inventory.iterrows():
    print(f"\n[{row['extension'] or 'no ext'}] {row['relative_path']}  ({row['size']})")
    print("  " + row["description"].replace("\n", "\n  "))


# %% SAVE: keep a copy of the report next to your other outputs.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
report_path = OUTPUT_DIR / f"inventory_{DATASET_PATH.name}.csv"
inventory.to_csv(report_path, index=False)
print(f"\nInventory saved to {report_path}")


# %% YOUR TURN: answer these from the report before we write the loader.
# 1. Which file holds spike times, and in what unit (samples or seconds)?
# 2. Which file holds positions? How many rows, and what are its columns?
# 3. Is there one timestamp per video frame? Where does it come from
#    (camera TTL recorded on the NIDQ/imec, or just frame_index / fps)?
# 4. Which file holds behaviour, and is it per frame (a label every frame)
#    or per bout (start, stop, label)?
# 5. Compare durations: recording (from .meta or last spike) versus video
#    (n_frames / fps). Do they agree? If not, by how much?
