"""
two_edge_align.py
=================

Put NI event times on the probe (imec) clock for recordings that have NO 1 Hz
sync pulse, using one session TTL that was recorded on BOTH streams.

WHY THIS EXISTS
---------------
The probe (imec) and the NI card each have their own clock. The two clocks run
at slightly different speeds (a few parts per million), so NI times slowly
drift away from probe times (about 10 ms per 30 min on our rig). Normally TPrime
fixes this with a 1 Hz sync pulse recorded on both streams. Our old recordings
(August / September 2026) had no sync pulse, so TPrime cannot be used.

But the session TTL (high during the whole session) was recorded on both
streams: on the imec SY channel (bit 6) and on an NI digital line (P0.0). Its
rising edge and its falling edge are two moments we know in BOTH clocks.
Two known points define a straight line, and the line converts any NI time
into probe time:

    t_probe = imec_rise + (t_ni - ni_rise) * (imec_fall - imec_rise) / (ni_fall - ni_rise)
              ---------   ------------------   ------------------------------------------
              start       time since start      speed ratio between the two clocks
              (probe)     (NI clock)            (measured, about 1.000001)

On the current rig, this straight line stays within 0.12 ms of the true sync
over a whole session (measured on 2026-09-23), well under 1 ms.

WHAT IT WRITES
--------------
Files with the same names and format as CatGT + TPrime, so the rest of the
NeuroPyGuiN pipeline finds and uses them without any change:

    <out>/<run>_g0_tcat.nidq.xd_8_<bit>_0.txt        rising edges, NI clock
    <out>/<run>_g0_tcat.nidq.xd_8_<bit>_0.adj.txt    rising edges, PROBE clock  <- use these
    <out>/<run>_g0_tcat.nidq.xid_8_<bit>_0.txt       falling edges, NI clock
    <out>/<run>_g0_tcat.nidq.xid_8_<bit>_0.adj.txt   falling edges, PROBE clock <- use these
    <out>/<label>_<same name>                        readable copies (e.g. barcode_...)
    <out>/<run>_g0_two_edge_alignment.txt            report: edges used, clock ratio, checks

Each event file has one time per line, in seconds from the start of the file.

HOW TO RUN (Anaconda Prompt, env "neuropyguin")
-----------------------------------------------
One session (output goes to the usual processed folder automatically):

    python tools\\two_edge_align.py "B:\\NPX\\rawData\\AOL\\20260904\\51569\\RewardObs\\1\\51562_1_RewardObs_RewardObs_g0"

Several sessions at once: list several run folders.
Choose another output folder:        --out "D:\\some\\folder"
Change which NI lines get a label:   --labels "3:syncttl,4:barcode"
Change the reference lines:          --ni-ref-bit 0 --imec-ref-bit 6

The raw data is only read, never modified.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np


# =============================================================================
# SETTINGS (defaults match the OLD rig wiring, Aug/Sep 2026)
# =============================================================================

# Line that carries the session TTL on each stream.
NI_REF_BIT = 0      # NI digital line P0.0
IMEC_REF_BIT = 6    # bit 6 of the imec SY channel (the SMA input of the base station)

# Readable names for the NI lines. A labelled copy of each event file is made,
# e.g. "barcode_<run>_g0_tcat.nidq.xd_8_4_0.adj.txt".
# Old rig: P0.0 = session TTL, P0.3 = SyncTTL from PyKaboo, P0.4 = barcode.
# (On the NEW rig the barcode is P0.1 and the trigger P0.4: change this if needed.)
DEFAULT_LABELS = {0: "trigger", 3: "syncttl", 4: "barcode"}

# Raw data root and processed data root, used to build the default output
# folder exactly like NeuroPyGuiN's "mirror" layout:
#   B:\NPX\rawData\<path>\<run>_g0  ->  I:\processedData\<path>\spike_sorting\catgt_<run>_g0
RAW_ROOT = Path(r"B:\NPX\rawData")
PROCESSED_ROOT = Path(r"I:\processedData")

# Safety limit: the two clocks never differ by more than this. A bigger value
# means the two edges are not the same physical event on both streams.
MAX_PLAUSIBLE_PPM = 100.0

# Read the big imec file in blocks of this many seconds (keeps memory low).
IMEC_BLOCK_SECONDS = 60


# =============================================================================
# Reading SpikeGLX files
# =============================================================================

def read_meta(meta_path: Path) -> dict:
    """Read a SpikeGLX .meta file into a dictionary {key: value as text}."""
    meta = {}
    for line in meta_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            meta[key.strip().lstrip("~")] = value.strip()
    return meta


def find_session_files(run_dir: Path) -> dict:
    """Locate the NI and imec0 files of one run folder (<run>_g0).

    Returns paths and the run name (folder name without the "_g0" ending).
    """
    run_dir = Path(run_dir)
    ni_meta = sorted(run_dir.glob("*.nidq.meta"))
    ap_meta = sorted(run_dir.glob("*_imec0/*.imec0.ap.meta"))
    if not ni_meta:
        raise FileNotFoundError(f"No .nidq.meta in {run_dir}")
    if not ap_meta:
        raise FileNotFoundError(f"No imec0 .ap.meta in {run_dir}")
    folder = run_dir.name                       # e.g. "51562_1_RewardObs_RewardObs_g0"
    run_name, _, gate = folder.rpartition("_g")  # -> "51562_1_RewardObs_RewardObs", "0"
    return {
        "ni_meta": ni_meta[0], "ni_bin": ni_meta[0].with_suffix(".bin"),
        "ap_meta": ap_meta[0], "ap_bin": ap_meta[0].with_suffix(".bin"),
        "run_name": run_name, "gate": gate or "0",
    }


def ni_digital_word(ni_bin: Path, ni_meta: dict):
    """Return (digital word samples, sampling rate, channel index) of the NI file.

    The NI file stores channels in the order MN, MA, XA, DW (see snsMnMaXaDw).
    The digital word is the first channel after all analog channels; with
    8 analog channels it is channel 8, which is why CatGT calls it "word 8".
    """
    n_chan = int(ni_meta["nSavedChans"])
    mn, ma, xa, dw = (int(v) for v in ni_meta["snsMnMaXaDw"].split(","))
    if dw < 1:
        raise ValueError("This NI file has no digital word.")
    word_index = mn + ma + xa
    data = np.memmap(ni_bin, dtype=np.int16, mode="r").reshape(-1, n_chan)
    word = data[:, word_index].astype(np.uint16)
    return word, float(ni_meta["niSampRate"]), word_index


# =============================================================================
# Finding edges
# =============================================================================

def edges_of_bit(word: np.ndarray, bit: int, rate: float):
    """Times (s) of rising and falling edges of one bit of a digital word.

    An edge time is the first sample in the new state, divided by the sampling
    rate (same convention as CatGT).
    """
    level = ((word >> bit) & 1).astype(np.int8)
    change = np.diff(level)
    rising = (np.flatnonzero(change == 1) + 1) / rate
    falling = (np.flatnonzero(change == -1) + 1) / rate
    return rising, falling


def imec_reference_edges(ap_bin: Path, ap_meta: dict, bit: int):
    """First rising edge and last falling edge of one SY bit in the imec file.

    The imec file is huge (tens of GB), so instead of reading it all we read
    blocks: forward from the start until the first rising edge, then backward
    from the end until the last falling edge.
    """
    n_chan = int(ap_meta["nSavedChans"])
    rate = float(ap_meta["imSampRate"])
    sy = n_chan - 1                                  # SY channel is the last one (384)
    data = np.memmap(ap_bin, dtype=np.int16, mode="r").reshape(-1, n_chan)
    n = data.shape[0]
    block = int(IMEC_BLOCK_SECONDS * rate)

    def level(a, b):
        return ((data[a:b, sy].astype(np.uint16) >> bit) & 1).astype(np.int8)

    # Forward search for the first rising edge.
    first_rise = None
    start = 0
    while start < n - 1 and first_rise is None:
        stop = min(n, start + block + 1)             # +1 overlap so no edge falls between blocks
        hits = np.flatnonzero(np.diff(level(start, stop)) == 1)
        if hits.size:
            first_rise = (start + hits[0] + 1) / rate
        start = stop - 1

    # Backward search for the last falling edge.
    last_fall = None
    stop = n
    while stop > 1 and last_fall is None:
        start = max(0, stop - block - 1)
        hits = np.flatnonzero(np.diff(level(start, stop)) == -1)
        if hits.size:
            last_fall = (start + hits[-1] + 1) / rate
        stop = start + 1

    return first_rise, last_fall, rate, n / rate


# =============================================================================
# The alignment itself
# =============================================================================

def make_ni_to_probe(ni_rise, ni_fall, imec_rise, imec_fall):
    """Return a function that converts NI times (s) into probe times (s).

    It is the straight line through the two known points
    (ni_rise -> imec_rise) and (ni_fall -> imec_fall).
    """
    ratio = (imec_fall - imec_rise) / (ni_fall - ni_rise)   # clock speed ratio

    def ni_to_probe(t_ni):
        return imec_rise + (np.asarray(t_ni, dtype=float) - ni_rise) * ratio

    return ni_to_probe, ratio


def write_times(path: Path, times) -> None:
    """Write one time per line, 6 decimals (microseconds), like CatGT/TPrime."""
    np.savetxt(path, np.asarray(times, dtype=float), fmt="%.6f")


def default_output_dir(run_dir: Path, run_name: str, gate: str) -> Path:
    """Mirror the raw path into the processed tree, like NeuroPyGuiN does."""
    session_dir = run_dir.parent
    try:
        relative = session_dir.relative_to(RAW_ROOT)
    except ValueError:
        raise ValueError(
            f"{run_dir} is not under {RAW_ROOT}; please give an output folder with --out"
        )
    return PROCESSED_ROOT / relative / "spike_sorting" / f"catgt_{run_name}_g{gate}"


def align_session(run_dir: Path, out_dir: Path | None, labels: dict,
                  ni_ref_bit: int, imec_ref_bit: int) -> Path:
    """Align one run folder. Returns the output folder."""
    files = find_session_files(run_dir)
    run, gate = files["run_name"], files["gate"]
    out_dir = Path(out_dir) if out_dir else default_output_dir(Path(run_dir), run, gate)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {run}_g{gate}\n    output: {out_dir}")

    ni_meta = read_meta(files["ni_meta"])
    ap_meta = read_meta(files["ap_meta"])
    warnings = []

    # --- 1. The two reference edges on each stream ---------------------------
    word, ni_rate, word_index = ni_digital_word(files["ni_bin"], ni_meta)
    ni_rises, ni_falls = edges_of_bit(word, ni_ref_bit, ni_rate)
    if ni_rises.size == 0 or ni_falls.size == 0:
        raise RuntimeError(f"No session TTL found on NI P0.{ni_ref_bit}.")
    if ni_rises.size != 1 or ni_falls.size != 1:
        warnings.append(
            f"NI P0.{ni_ref_bit} has {ni_rises.size} rising and {ni_falls.size} falling edges "
            f"(expected 1 and 1). Using the first rising and the last falling edge."
        )
    ni_rise, ni_fall = ni_rises[0], ni_falls[-1]

    imec_rise, imec_fall, imec_rate, imec_duration = imec_reference_edges(
        files["ap_bin"], ap_meta, imec_ref_bit)
    if imec_rise is None or imec_fall is None:
        raise RuntimeError(f"No session TTL found on imec SY bit {imec_ref_bit}.")

    # --- 2. Sanity checks -----------------------------------------------------
    if not (ni_fall > ni_rise and imec_fall > imec_rise):
        raise RuntimeError("The falling edge comes before the rising edge: wrong lines?")
    ni_span, imec_span = ni_fall - ni_rise, imec_fall - imec_rise
    ppm = (imec_span / ni_span - 1.0) * 1e6
    if abs(ppm) > MAX_PLAUSIBLE_PPM:
        raise RuntimeError(
            f"Clock ratio of {ppm:.1f} ppm is not plausible: the edges found on the two "
            f"streams are probably not the same physical event."
        )
    start_offset_ms = (imec_rise - ni_rise) * 1e3
    if abs(start_offset_ms) > 1000:
        warnings.append(f"The two streams start {start_offset_ms:.0f} ms apart (unusual).")

    ni_to_probe, ratio = make_ni_to_probe(ni_rise, ni_fall, imec_rise, imec_fall)
    drift_ms = (ni_span * ratio - ni_span) * 1e3
    print(f"    session TTL  NI: {ni_rise:.6f} -> {ni_fall:.6f} s   "
          f"imec: {imec_rise:.6f} -> {imec_fall:.6f} s")
    print(f"    clock ratio {ratio:.9f} ({ppm:+.2f} ppm), drift corrected over the session: "
          f"{drift_ms:+.2f} ms")

    # --- 3. Every active NI line: raw and probe-clock event files --------------
    stem = f"{run}_g{gate}_tcat.nidq"
    written = []
    for bit in range(8):
        rises, falls = edges_of_bit(word, bit, ni_rate)
        if rises.size == 0 and falls.size == 0:
            continue                                   # line never changed: skip it
        for kind, times in (("xd", rises), ("xid", falls)):
            base = f"{stem}.{kind}_{word_index}_{bit}_0"
            raw_file = out_dir / f"{base}.txt"
            adj_file = out_dir / f"{base}.adj.txt"
            write_times(raw_file, times)
            write_times(adj_file, ni_to_probe(times))
            written.append((bit, kind, times.size, adj_file.name))
            label = labels.get(bit)
            if label:                                  # readable copies, like NeuroPyGuiN
                suffix = "" if kind == "xd" else "_off"
                shutil.copyfile(raw_file, out_dir / f"{label}{suffix}_{raw_file.name}")
                shutil.copyfile(adj_file, out_dir / f"{label}{suffix}_{adj_file.name}")

    for bit, kind, count, name in written:
        edge = "rising " if kind == "xd" else "falling"
        print(f"    P0.{bit} {edge} {count:6d} events -> {name}")

    # --- 4. Report --------------------------------------------------------------
    report = out_dir / f"{run}_g{gate}_two_edge_alignment.txt"
    lines = [
        "Two-edge alignment of NI events onto the imec0 (probe) clock",
        f"script: {Path(__file__).name}",
        f"run folder: {run_dir}",
        f"NI file: {files['ni_bin']}  (rate {ni_rate} Hz, digital word = channel {word_index})",
        f"imec file: {files['ap_bin']}  (rate {imec_rate} Hz, duration {imec_duration:.3f} s)",
        "",
        f"reference line NI: P0.{ni_ref_bit}   imec: SY bit {imec_ref_bit}",
        f"NI   rising {ni_rise:.6f} s   falling {ni_fall:.6f} s   span {ni_span:.6f} s",
        f"imec rising {imec_rise:.6f} s   falling {imec_fall:.6f} s   span {imec_span:.6f} s",
        f"clock ratio (imec/NI) {ratio:.9f}  =  {ppm:+.3f} ppm",
        f"start offset imec - NI {start_offset_ms:+.3f} ms; drift corrected over session {drift_ms:+.3f} ms",
        "",
        "formula: t_probe = imec_rise + (t_ni - ni_rise) * ratio",
        "",
        "event files written (probe clock = .adj.txt):",
    ] + [f"  P0.{b} {k:3s} {c:6d}  {n}" for b, k, c, n in written]
    lines += ["", "warnings:"] + ([f"  {w}" for w in warnings] or ["  none"])
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for w in warnings:
        print(f"    WARNING: {w}")
    print(f"    report: {report.name}")
    return out_dir


# =============================================================================
# Command line
# =============================================================================

def parse_labels(text: str) -> dict:
    """'3:syncttl,4:barcode' -> {3: 'syncttl', 4: 'barcode'}"""
    labels = {}
    for item in filter(None, (p.strip() for p in text.split(","))):
        bit, name = item.split(":", 1)
        labels[int(bit)] = name.strip()
    return labels


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Align NI events to the imec clock using the session TTL's two edges.")
    parser.add_argument("run_dirs", nargs="+", help="Run folder(s), e.g. ...\\<run>_g0")
    parser.add_argument("--out", help="Output folder (only with one run folder). "
                                      "Default: processed mirror .../spike_sorting/catgt_<run>_g0")
    parser.add_argument("--labels", help="Readable names, e.g. '0:trigger,3:syncttl,4:barcode'")
    parser.add_argument("--ni-ref-bit", type=int, default=NI_REF_BIT)
    parser.add_argument("--imec-ref-bit", type=int, default=IMEC_REF_BIT)
    args = parser.parse_args(argv)

    if args.out and len(args.run_dirs) > 1:
        parser.error("--out can only be used with a single run folder")
    labels = parse_labels(args.labels) if args.labels else DEFAULT_LABELS

    failed = []
    for run_dir in args.run_dirs:
        try:
            align_session(Path(run_dir), args.out, labels, args.ni_ref_bit, args.imec_ref_bit)
        except Exception as exc:                       # keep going with the other sessions
            print(f"    FAILED: {exc}")
            failed.append(run_dir)
    if failed:
        print(f"\n{len(failed)} session(s) failed: " + ", ".join(map(str, failed)))
        return 1
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
