"""Read saved Phy curation and measure firing rates, without changing the session.

Start with the USER SETTINGS below, then read the numbered sections in
``analyze_session``. The companion ``README_curated_spikes.md`` explains the
data model and gives Python/MATLAB examples. ``# %%`` marks IDE code sections.

This is a Kilosort/Phy tutorial, not a new spike sorter. It needs NumPy, pandas,
SciPy and matplotlib, already installed in this computer's neuropyguin env.
"""

# %% USER SETTINGS: these are the values you normally edit.
from pathlib import Path

# Use the most recently saved Phy example found on this computer, not the newest
# uncurated recording. A processed session parent is also accepted, but it must
# contain exactly one sorter. Never silently choose between multiple sorters.
SESSION_PATH = Path(
    "I:/processedData/CA2/Batch1/20261021/52488/spike_sorting/"
    "catgt_52488_1_CA2Oxy_FreeSocialInteract_g0/"
    "52488_1_CA2Oxy_FreeSocialInteract_g0_imec0/imec0_ks4"
)
OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "NeuroPyGuiN_output/phy_spikes_tutorial"
WINDOW_DURATION_S = 5 * 60.0  # Seconds, measured from the first saved AP sample.
BIN_WIDTH_S = 1.0  # Each column measures one second, by default.
SAVE_FIGURES = True
FIGURE_SIZE_INCHES = (10, 8)
FIGURE_DPI = 180
HEATMAP_COLORMAP = "viridis"

# If you preserved a genuine pre-Phy spike_clusters.npy, put its path here.
# Otherwise compare with spike_templates.npy, which Phy does not modify.
# That template reference is NOT a versioned backup of every processing step.
ORIGINAL_CLUSTERS_PATH = None


# %% LIBRARIES: imports are Python's equivalent of making tools available.
import ast  # Safely read literal settings in params.py WITHOUT executing it.
from datetime import datetime, timezone
import hashlib  # Record fingerprints so an extraction is reproducible.
import json  # Save the analysis settings and audit in a human-readable format.

import numpy as np  # Numerical arrays, the closest equivalent to MATLAB arrays.
import pandas as pd  # Tables with named columns, similar to MATLAB tables.
from scipy.io import savemat  # Write ordinary MATLAB .mat files.


def find_sorter_folder(session_path):
    """Accept a sorter itself or a parent containing exactly one valid sorter.

    A folder's name is not enough: identify it by its actual spike files.
    Ambiguity is an error because selecting the wrong session is worse than
    asking the user to supply a more precise path.
    """
    root = Path(session_path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Session folder not found: {root}")
    if (root / "spike_times.npy").is_file() and (root / "spike_clusters.npy").is_file():
        return root
    candidates = sorted(
        p.parent for p in root.rglob("spike_times.npy")
        if (p.parent / "spike_clusters.npy").is_file()
    )
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one sorter under {root}, found {len(candidates)}. "
            "Set SESSION_PATH to the exact imec*_ks* folder."
        )
    return candidates[0]


def read_recording_info(sorter_folder):
    """Read sampling rate and recording length without loading the AP binary.

    params.py is Python code. We use its syntax tree and ast.literal_eval,
    rather than exec(), so importing a recording cannot execute its code.
    The binary's byte count gives its duration, including silent periods.
    The time of the final detected spike is NOT the recording duration.
    """
    params_path = sorter_folder / "params.py"
    tree = ast.parse(params_path.read_text(encoding="utf-8-sig"))
    params = {}
    wanted = {"dat_path", "sample_rate", "n_channels_dat", "dtype", "offset"}
    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                if isinstance(target, ast.Name) and target.id in wanted:
                    params[target.id] = ast.literal_eval(statement.value)

    # A relative dat_path is interpreted relative to params.py, not your IDE.
    dat_value = params.get("dat_path")
    if not isinstance(dat_value, str):
        raise ValueError("params.py must specify one binary file as a dat_path string.")
    binary_path = Path(dat_value)
    if not binary_path.is_absolute():
        binary_path = sorter_folder / binary_path
    binary_path = binary_path.resolve()
    if not binary_path.is_file():
        raise FileNotFoundError(f"The binary referenced by params.py is missing: {binary_path}")

    meta_path = binary_path.with_suffix(".meta")
    meta = {}
    if meta_path.is_file():
        for line in meta_path.read_text(encoding="utf-8-sig").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                meta[key.strip()] = value.strip()

    # Prefer the AP stream's recorded sampling rate, as NeuroPyGuiN also does.
    # Do not assume 30000 Hz: some recordings use a calibrated, slightly different rate.
    params_rate = float(params["sample_rate"])
    sample_rate_hz = float(meta.get("imSampRate", params_rate))
    n_channels = int(params["n_channels_dat"])
    dtype = np.dtype(params.get("dtype", "int16"))
    offset_bytes = int(params.get("offset", 0))
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("Sampling rate must be a finite positive number.")
    if n_channels <= 0 or offset_bytes < 0 or dtype.kind not in "iu":
        raise ValueError("Invalid binary channel count, offset, or integer data type.")
    if "nSavedChans" in meta and int(meta["nSavedChans"]) != n_channels:
        raise ValueError("params.py and AP metadata disagree about the saved channel count.")
    n_data_bytes = binary_path.stat().st_size - offset_bytes
    bytes_per_sample = n_channels * dtype.itemsize
    if n_data_bytes <= 0 or n_data_bytes % bytes_per_sample:
        raise ValueError("Binary size is not a positive whole number of multichannel samples.")
    n_samples = n_data_bytes // bytes_per_sample
    return {
        "sample_rate_hz": sample_rate_hz,
        "params_sample_rate_hz": params_rate,
        "sample_rate_source": "AP metadata imSampRate" if "imSampRate" in meta else "params.py",
        "recording_duration_s": n_samples / sample_rate_hz,
        "n_recording_samples": n_samples,
        "binary_path": str(binary_path),
        "meta_path": str(meta_path) if meta_path.is_file() else None,
        "n_channels": n_channels,
    }


def load_integer_vector(path):
    """Load one integer per spike, keeping even zero/one-spike arrays 1-D.

    mmap_mode='r' opens a read-only memory map instead of copying the whole
    file immediately. allow_pickle=False avoids executing pickled objects.
    Do not flatten arbitrary multi-column arrays: that would mix up spikes.
    """
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1 or array.dtype.kind not in "iu":
        raise ValueError(f"Expected a 1-D integer vector (or N x 1), got {array.shape}: {path}")
    if np.any(array < 0):
        raise ValueError(f"Negative sample indices or cluster IDs are not supported: {path}")
    return array


def read_labels(path, label_column, separator="\t"):
    """Read a cluster table and use cluster_id as its explicit lookup key.

    A cluster ID is a name, not its row number. IDs can have gaps after merges
    and splits. A pandas Series indexed by ID prevents off-by-one mistakes.
    """
    table = pd.read_csv(path, sep=separator)
    required = {"cluster_id", label_column}
    if not required.issubset(table.columns):
        raise ValueError(f"{path} must contain columns {sorted(required)}")
    ids = pd.to_numeric(table["cluster_id"], errors="raise")
    if ids.isna().any() or (ids < 0).any() or (ids % 1 != 0).any() or ids.duplicated().any():
        raise ValueError(f"Cluster IDs must be unique nonnegative integers: {path}")
    labels = table[label_column].fillna("unsorted").astype(str).str.strip().str.lower()
    labels = labels.replace("", "unsorted")
    return pd.Series(labels.to_numpy(), index=ids.to_numpy(dtype=np.int64), name=label_column)


def fingerprint(path):
    """Record file size, timestamp and SHA256, without modifying the file.

    Small sequential blocks avoid loading an entire file just to hash it.
    This records inputs for this extraction; it does not recover a lost backup.
    """
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.stat()
    return {
        "path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "sha256": digest.hexdigest(),
    }


def calculate_firing_rates(spike_samples, spike_clusters, good_ids, sample_rate_hz,
                           duration_s, bin_width_s):
    """Return counts, rates and edges for the half-open interval [0, duration).

    Rows are good neurons, in good_ids order. Columns are time bins.
    Rate = spike count / bin duration in seconds. No smoothing is applied.
    A final partial bin is allowed and uses its actual duration, not bin_width_s.
    """
    if not all(np.isfinite(x) and x > 0 for x in (sample_rate_hz, duration_s, bin_width_s)):
        raise ValueError("Sampling rate, window duration and bin width must be finite and positive.")
    if len(spike_samples) != len(spike_clusters):
        raise ValueError("There must be exactly one cluster ID per spike time.")
    if len(np.unique(good_ids)) != len(good_ids):
        raise ValueError("good_ids must not contain duplicates.")

    starts = np.arange(0.0, duration_s, bin_width_s)
    bin_edges_s = np.append(starts[starts < duration_s], duration_s)
    bin_durations_s = np.diff(bin_edges_s)
    # Bin in sample coordinates to avoid rounding a spike's time into the wrong
    # bin. In floating-point arithmetic, 0.3 can become 0.30000000000000004.
    # Snap a boundary to an integer sample only within machine-rounding error;
    # genuinely fractional sample boundaries are kept as fractional values.
    sample_edges = bin_edges_s * sample_rate_hz
    nearest_sample = np.rint(sample_edges)
    rounding_tolerance = 8 * np.finfo(float).eps * np.maximum(1, np.abs(sample_edges))
    sample_edges = np.where(np.abs(sample_edges - nearest_sample) <= rounding_tolerance,
                            nearest_sample, sample_edges)
    # Python/MATLAB distinction: '&' combines two elementwise Boolean tests.
    # Filter the analysis endpoint explicitly: np.histogram otherwise includes
    # the rightmost edge in its final bin. A spike at exactly 300 s is excluded.
    in_window = (spike_samples >= 0) & (spike_samples < sample_edges[-1])
    window_samples = spike_samples[in_window]
    window_clusters = spike_clusters[in_window]
    counts = np.zeros((len(good_ids), len(bin_durations_s)), dtype=np.int64)

    # Deliberately use a readable loop. Each iteration handles one neuron.
    # enumerate() supplies both the matrix row and the neuron's actual ID.
    for row, cluster_id in enumerate(good_ids):
        neuron_samples = window_samples[window_clusters == cluster_id]
        counts[row, :] = np.histogram(neuron_samples, bins=sample_edges)[0]
    # np.newaxis gives durations shape (1, number_of_bins). NumPy broadcasts
    # that row across all neurons, dividing every column by its own duration.
    rates_hz = counts / bin_durations_s[np.newaxis, :]
    # axis=1 sums columns separately for each neuron: MATLAB sum(counts, 2).
    mean_rates_hz = counts.sum(axis=1) / duration_s

    # Internal validation: counting in bins must not lose or duplicate spikes.
    expected = np.count_nonzero(np.isin(window_clusters, good_ids))
    if int(counts.sum()) != expected:
        raise RuntimeError("Binned spike counts failed the conservation check.")
    return counts, rates_hz, mean_rates_hz, bin_edges_s


def save_overview(output_dir, rates_hz, mean_rates_hz, bin_edges_s, label_counts):
    """Save descriptive figures only; there is no hypothesis test in this task.

    matplotlib is imported here so loading the spike functions needs no GUI.
    Agg renders files without opening a window or interrupting an acquisition.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context({"font.size": 10, "svg.fonttype": "none", "pdf.fonttype": 42}):
        fig, axes = plt.subplots(3, 1, figsize=FIGURE_SIZE_INCHES, layout="constrained",
                                 gridspec_kw={"height_ratios": [1, 2.2, 1]})
        names = list(label_counts)
        values = list(label_counts.values())
        bars = axes[0].bar(names, values, width=0.45, color=["#777777", "#368c8a", "#d55e00"][:len(names)])
        axes[0].bar_label(bars, padding=3)
        axes[0].set_ylim(0, max(values + [1]) * 1.25)
        axes[0].set_ylabel("Good clusters")
        axes[0].set_title("Automatic labels versus saved Phy labels (descriptive comparison)")
        if rates_hz.shape[0]:
            image = axes[1].imshow(
                rates_hz, origin="lower", aspect="auto", interpolation="nearest",
                extent=[bin_edges_s[0], bin_edges_s[-1], -0.5, len(rates_hz) - 0.5],
                cmap=HEATMAP_COLORMAP, vmin=0,
            )
            fig.colorbar(image, ax=axes[1], label="Firing rate (spikes/s)", pad=0.015)
            axes[2].hist(mean_rates_hz, bins=min(25, max(1, len(mean_rates_hz))),
                         color="#368c8a", rwidth=0.85)
        else:
            axes[1].text(0.5, 0.5, "No active clusters labeled good", ha="center",
                         va="center", transform=axes[1].transAxes)
        axes[1].set_xlabel("Time from first saved AP sample (s)")
        axes[1].set_ylabel("Matrix row (0-based)")
        axes[1].set_title(f"{len(rates_hz)} saved-good clusters; no smoothing")
        axes[2].set_xlabel(f"Mean firing rate over {bin_edges_s[-1]:g} s (spikes/s)")
        axes[2].set_ylabel("Good clusters")
        for ax in axes:
            ax.spines[["top", "right"]].set_visible(False)
        for extension in ("png", "pdf", "svg"):
            fig.savefig(output_dir / f"curation_and_firing_rates.{extension}", dpi=FIGURE_DPI)
        plt.close(fig)


# %% MAIN LESSON: follow the numbered steps in this function.
def analyze_session(session_path, output_root, duration_s=WINDOW_DURATION_S,
                    bin_width_s=BIN_WIDTH_S, make_figures=SAVE_FIGURES,
                    original_clusters_path=ORIGINAL_CLUSTERS_PATH):
    """Extract saved curation, audit it, export matrices and return IDE variables.

    Inputs are read only. Every run gets a NEW output directory. Keep Phy saved
    and preferably closed; changes made during extraction invalidate the run.
    """
    # 1. Identify the exact source and the recording's clock and duration.
    sorter = find_sorter_folder(session_path)
    info = read_recording_info(sorter)
    sample_rate_hz = info["sample_rate_hz"]
    if not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("Analysis duration must be finite and positive.")
    if info["recording_duration_s"] < duration_s:
        raise ValueError(
            f"Recording is only {info['recording_duration_s']:.3f} s, shorter than "
            f"the requested {duration_s:g} s. Do not pad missing recording time with zeros."
        )
    source_paths = [sorter / name for name in (
        "spike_times.npy", "spike_clusters.npy", "cluster_group.tsv", "params.py"
    )]
    # No fallback to KSLabel/Bombcell: missing Phy labels must be visible to you.
    for path in source_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required saved Phy/Kilosort input is missing: {path}")
    reference = Path(original_clusters_path).resolve() if original_clusters_path else sorter / "spike_templates.npy"
    optional = [reference, sorter / "cluster_KSLabel.tsv", sorter / "bombcell_labels.csv"]
    if original_clusters_path and not reference.is_file():
        raise FileNotFoundError(f"Explicit original-cluster backup not found: {reference}")
    if info["meta_path"]:
        optional.append(Path(info["meta_path"]))
    source_paths = list(dict.fromkeys(source_paths + [p for p in optional if p.is_file()]))
    before = {str(p): fingerprint(p) for p in source_paths}
    # Recheck the settings after fingerprinting so a concurrent save cannot
    # pair old sampling information with a newer copy of the source files.
    if read_recording_info(sorter) != info:
        raise RuntimeError("Recording settings changed during setup. Save/close Phy and rerun.")
    binary_before = Path(info["binary_path"]).stat()

    # 2. One row per spike: these two arrays describe the SAME ordered events.
    spike_times_samples = load_integer_vector(sorter / "spike_times.npy")
    spike_cluster_ids = load_integer_vector(sorter / "spike_clusters.npy")
    if len(spike_times_samples) != len(spike_cluster_ids):
        raise ValueError("spike_times and spike_clusters have different lengths.")
    if np.any(spike_times_samples[1:] < spike_times_samples[:-1]):
        raise ValueError("Spike sample times must be sorted in nondecreasing order.")
    if len(spike_times_samples) and spike_times_samples[-1] >= info["n_recording_samples"]:
        raise ValueError("A spike lies outside the binary recording's sample range.")
    active_ids, n_spikes_per_cluster = np.unique(spike_cluster_ids, return_counts=True)

    # 3. Read the SAVED manual groups. Indexing by ID is crucial here.
    saved_labels = read_labels(sorter / "cluster_group.tsv", "group")
    active_labels = saved_labels.reindex(active_ids).fillna("unsorted")
    good_cluster_ids = active_ids[active_labels.to_numpy() == "good"]
    missing_labels = active_ids[~np.isin(active_ids, saved_labels.index)]

    # 4. Separate two kinds of curation evidence: labels versus membership.
    # Changing good -> noise edits cluster_group.tsv, not necessarily the spike
    # assignments. Therefore unchanged spike_clusters is NOT a failed save.
    comparison = pd.DataFrame({
        "cluster_id": active_ids, "n_spikes_full_recording": n_spikes_per_cluster,
        "saved_phy_group": active_labels.to_numpy(),
    })
    label_counts = {}
    audit = {"session_folder": str(sorter), **info, "n_spikes": len(spike_times_samples),
             "n_active_clusters": len(active_ids), "n_good_clusters": len(good_cluster_ids),
             "unlabeled_active_cluster_ids": missing_labels.tolist(),
             "inactive_labeled_cluster_ids": saved_labels.index.difference(active_ids).tolist(),
             "good_selection_source": "cluster_group.tsv ONLY",
             "original_assignment_reference": str(reference) if reference.is_file() else None,
             "original_reference_is_explicit_backup": bool(original_clusters_path),
             "assignment_comparison": "unavailable: no original reference found"}
    for filename, column, separator, name in (
        ("cluster_KSLabel.tsv", "KSLabel", "\t", "Kilosort"),
        ("bombcell_labels.csv", "bombcell_label", ",", "Bombcell"),
    ):
        path = sorter / filename
        if not path.is_file():
            continue
        automatic = read_labels(path, column, separator)
        mapped = automatic.reindex(active_ids)
        comparison[f"{name.lower()}_label"] = mapped.to_numpy()
        comparable = mapped.notna().to_numpy()
        changed = comparable & (mapped.to_numpy() != active_labels.to_numpy())
        comparison[f"changed_vs_{name.lower()}"] = changed
        audit[f"n_labels_changed_vs_{name.lower()}_common_ids"] = int(changed.sum())
        audit[f"n_current_ids_missing_from_{name.lower()}"] = int((~comparable).sum())
        # Original counts are calculated on their own original IDs. After merges
        # or splits, their cluster population can differ from the current one.
        label_counts[name] = int((automatic == "good").sum())
    label_counts["Saved Phy"] = len(good_cluster_ids)

    if reference.is_file():
        original = load_integer_vector(reference)
        if len(original) != len(spike_cluster_ids):
            raise ValueError("Original and current assignments have different spike counts; comparison is unsafe.")
        audit["n_spikes_with_changed_cluster_assignment"] = int(np.count_nonzero(original != spike_cluster_ids))
        audit["assignment_comparison"] = "compared elementwise in the same spike order"
    audit["interpretation_limit"] = (
        "Automatic label differences alone do not prove who edited them. Phy log history "
        "can corroborate manual edits. Without a genuine pre-curation backup, there is "
        "no complete version history. Unsaved edits in an open Phy window are not on disk."
    )

    # 5. Measure first-window firing rates using the current saved assignments.
    counts, firing_rates_hz, mean_firing_rates_hz, bin_edges_s = calculate_firing_rates(
        spike_times_samples, spike_cluster_ids, good_cluster_ids,
        sample_rate_hz, duration_s, bin_width_s,
    )
    bin_centers_s = (bin_edges_s[:-1] + bin_edges_s[1:]) / 2
    bin_durations_s = np.diff(bin_edges_s)

    # These are the full-session spikes for GOOD clusters, not just five minutes.
    # They form a flat event table, not a huge neurons x 30000Hz binary matrix.
    is_good_spike = np.isin(spike_cluster_ids, good_cluster_ids)
    good_spike_samples = spike_times_samples[is_good_spike]
    good_spike_cluster_ids = spike_cluster_ids[is_good_spike]
    good_spike_times_s = good_spike_samples.astype(np.float64) / sample_rate_hz
    # A list of differently sized arrays is analogous to a MATLAB cell array.
    good_spike_trains_s = [good_spike_times_s[good_spike_cluster_ids == cid]
                          for cid in good_cluster_ids]

    # 6. Refuse a mixed snapshot if Phy saved while the analysis was reading.
    after = {str(p): fingerprint(p) for p in source_paths}
    binary_after = Path(info["binary_path"]).stat()
    if before != after or (binary_before.st_size, binary_before.st_mtime_ns) != (
            binary_after.st_size, binary_after.st_mtime_ns):
        raise RuntimeError("Source files changed during extraction. Save/close Phy and rerun.")

    # 7. Save only NEW derived files, outside the source session tree.
    output_root = Path(output_root).expanduser().resolve()
    if output_root == sorter or sorter in output_root.parents:
        raise ValueError("Choose an OUTPUT_ROOT outside the source sorter folder.")
    if output_root == Path(info["binary_path"]).parent or Path(info["binary_path"]).parent in output_root.parents:
        raise ValueError("Choose an OUTPUT_ROOT outside the source recording folder.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output_dir = output_root / f"{sorter.parent.name}_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    matrices = {
        "firing_rates_hz": firing_rates_hz, "spike_counts": counts,
        "good_cluster_ids": good_cluster_ids, "mean_firing_rates_hz": mean_firing_rates_hz,
        "bin_edges_s": bin_edges_s, "bin_centers_s": bin_centers_s,
        "bin_durations_s": bin_durations_s, "sample_rate_hz": sample_rate_hz,
        "window_start_s": 0.0, "window_duration_s": duration_s,
        "source_sorter_folder": str(sorter), "time_origin": "first saved AP sample",
    }
    np.savez_compressed(output_dir / "firing_rates.npz", **matrices)
    # oned_as='column' makes ID and rate vectors convenient MATLAB columns.
    savemat(output_dir / "firing_rates.mat", matrices, do_compression=True, oned_as="column")
    np.savez_compressed(
        output_dir / "curated_good_spikes.npz", sample_indices=good_spike_samples,
        times_s=good_spike_times_s, cluster_ids=good_spike_cluster_ids,
        good_cluster_ids=good_cluster_ids, sample_rate_hz=sample_rate_hz,
    )
    rate_table = pd.DataFrame(firing_rates_hz, index=good_cluster_ids,
                              columns=[f"bin_{i:03d}" for i in range(counts.shape[1])])
    rate_table.index.name = "cluster_id"
    rate_table.to_csv(output_dir / "firing_rates.csv")
    pd.DataFrame({"bin_index_python": np.arange(counts.shape[1]),
                  "start_s": bin_edges_s[:-1], "end_s": bin_edges_s[1:],
                  "center_s": bin_centers_s, "duration_s": bin_durations_s}).to_csv(
        output_dir / "time_bins.csv", index=False)
    summary = pd.DataFrame({
        "matrix_row_python": np.arange(len(good_cluster_ids)),
        "matrix_row_matlab": np.arange(1, len(good_cluster_ids) + 1),
        "cluster_id": good_cluster_ids, "spikes_in_window": counts.sum(axis=1),
        "mean_firing_rate_hz": mean_firing_rates_hz,
        "spikes_full_recording": [len(train) for train in good_spike_trains_s],
    })
    summary.to_csv(output_dir / "neuron_summary.csv", index=False)
    comparison.to_csv(output_dir / "curation_comparison.csv", index=False)
    # Export only actual differences on comparable IDs. New cluster IDs are not
    # silently counted as relabels of a different original neuron.
    change_columns = [c for c in comparison.columns if c.startswith("changed_vs_")]
    changed_rows = comparison[change_columns].any(axis=1) if change_columns else np.zeros(len(comparison), dtype=bool)
    comparison.loc[changed_rows].to_csv(output_dir / "changed_cluster_labels.csv", index=False)
    for column in change_columns:
        # Separate tables keep Kilosort-to-Phy changes (which can include the
        # automatic Bombcell step) distinct from Bombcell-to-Phy changes.
        comparison.loc[comparison[column]].to_csv(output_dir / f"{column}.csv", index=False)
    pd.DataFrame({"label_source": list(label_counts),
                  "good_cluster_count": list(label_counts.values())}).to_csv(
        output_dir / "label_source_counts.csv", index=False)
    pd.DataFrame({"sample_index": spike_times_samples[:10],
                  "time_s": spike_times_samples[:10] / sample_rate_hz,
                  "cluster_id": spike_cluster_ids[:10]}).to_csv(output_dir / "spike_table_first_10_rows.csv", index=False)

    audit.update({"created_utc": datetime.now(timezone.utc).isoformat(),
                  "window_duration_s": duration_s, "bin_width_requested_s": bin_width_s,
                  "time_origin": "first saved AP sample, NOT behavioral TTL onset",
                  "firing_rate_matrix_shape": list(firing_rates_hz.shape),
                  "n_good_spikes_in_window": int(counts.sum()),
                  "n_good_spikes_full_recording": len(good_spike_samples),
                  "input_files": list(before.values()), "source_files_unchanged": True,
                  "good_counts_by_available_label_source": label_counts,
                  "descriptive_statistics_only": True,
                  "numpy_version": np.__version__, "pandas_version": pd.__version__})
    (output_dir / "analysis_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    if make_figures:
        save_overview(output_dir, firing_rates_hz, mean_firing_rates_hz, bin_edges_s, label_counts)

    print(f"\nSource: {sorter}")
    print(f"Saved Phy good clusters with spikes: {len(good_cluster_ids)}")
    print(f"Full recording: {info['recording_duration_s']:.3f} s; sampling rate: {sample_rate_hz:g} Hz")
    print(f"Rate matrix: {firing_rates_hz.shape} = neurons x time bins")
    print(f"Window: [0, {duration_s:g}) s on the saved AP clock; units: spikes/s")
    print(f"Cluster-assignment differences: {audit.get('n_spikes_with_changed_cluster_assignment', 'not available')}")
    for source in ("kilosort", "bombcell"):
        key = f"n_labels_changed_vs_{source}_common_ids"
        if key in audit:
            print(f"Label differences versus {source}: {audit[key]}")
    if len(missing_labels):
        print(f"WARNING: {len(missing_labels)} active clusters lack labels; treated as unsorted, not good.")
    if not np.isclose(sample_rate_hz, info["params_sample_rate_hz"], rtol=1e-9):
        print("NOTE: AP metadata and params.py rates differ; this analysis used the AP metadata rate.")
    print(f"Outputs (source files unchanged): {output_dir}\n")
    return {"spike_times_samples": spike_times_samples, "spike_cluster_ids": spike_cluster_ids,
            "sample_rate_hz": sample_rate_hz, "good_cluster_ids": good_cluster_ids,
            "good_spike_trains_s": good_spike_trains_s, "firing_rates_hz": firing_rates_hz,
            "mean_firing_rates_hz": mean_firing_rates_hz, "spike_counts": counts,
            "bin_edges_s": bin_edges_s, "output_dir": output_dir, "audit": audit}


# %% RUN: when imported by a test, this block does not execute.
if __name__ == "__main__":
    results = analyze_session(SESSION_PATH, OUTPUT_ROOT)
    # In Spyder or another IDE, inspect these variables after running the file.
    # Example: good_spike_trains_s[0] is the first GOOD neuron's full spike train.
    spike_times_samples = results["spike_times_samples"]
    spike_cluster_ids = results["spike_cluster_ids"]
    sample_rate_hz = results["sample_rate_hz"]
    good_cluster_ids = results["good_cluster_ids"]
    good_spike_trains_s = results["good_spike_trains_s"]
    firing_rates_hz = results["firing_rates_hz"]
    mean_firing_rates_hz = results["mean_firing_rates_hz"]
