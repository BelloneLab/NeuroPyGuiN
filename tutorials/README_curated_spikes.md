# Reading Your Saved Phy Spikes With Python

## Start Here

Open `curated_spikes_tutorial.py` in your IDE using the existing `neuropyguin`
Python environment. On this computer the interpreter is:

```text
C:\Users\bellone\anaconda3\envs\neuropyguin\python.exe
```

Run the entire file, or double-click `run_curated_spikes.cmd`. The launcher uses
the same environment and keeps its console open so you can read the results.
It does not install anything, rerun sorting, or modify your recording.

The editable settings are at the top. `SESSION_PATH` should point to one sorter
folder containing `spike_times.npy`, `spike_clusters.npy`, `cluster_group.tsv`
and `params.py`. A parent path also works if it contains exactly one sorter.
Do not point it at a raw-data folder and expect it to perform spike sorting.

The default example is mouse **52488, October 21, 2026**, CA2 Batch1. Its Phy
curation was saved on November 3. The newest preprocessing output found was an
October 22 recording without Phy edits; selecting it would not demonstrate
your manual changes. Neither original folder is renamed or altered.

## 1. What Is Actually Being Loaded?

Your recordings were sorted with **Kilosort 4**, not KlustaKwik. Phy can work
with several sorting formats; your dataset uses the Kilosort/NumPy format,
not one legacy `.kwik` file.

| File | One row represents | Meaning |
| --- | --- | --- |
| `spike_times.npy` | One detected spike | Integer index of that spike's AP sample |
| `spike_clusters.npy` | The same spike | Its current saved cluster ID, after Phy curation |
| `cluster_group.tsv` | One cluster | Your saved `good`, `mua`, `noise`, etc. classification |
| `params.py` | Recording settings | Binary path, sample rate and channel count |
| `spike_templates.npy` | The same spike | Original sorter template assignment, not edited by Phy |
| `cluster_KSLabel.tsv` | An original sorter cluster | Automatic Kilosort quality label |
| `bombcell_labels.csv` | An automatically classified cluster | Automatic Bombcell quality label |

NumPy's `.npy` format stores a numerical array together with its shape and data
type. It is not a text file. TSV means tab-separated text; CSV means
comma-separated text. pandas reads those text tables and preserves the named
columns.

The fundamental representation is an **event table**, conceptually:

```text
spike row     sample index     current cluster ID
0             123              7
1             145              42
2             190              7
```

These numbers illustrate the structure; they are not measurements from your
recording. The first and third events belong to neuron 7. NumPy keeps the time
and cluster columns in separate arrays with matching row order.

**A cluster ID is not a recording channel and is not necessarily an array row.**
After curation, IDs can have gaps and newly created IDs. Never use a cluster ID
as an index into a compact matrix unless an explicit lookup maps them.

## 2. The Core Extraction, Without the Safety Checks

This is the short conceptual version. The complete script adds validation,
provenance, an audit, exports, and figures.

```python
from pathlib import Path
import numpy as np
import pandas as pd

folder = Path("I:/your/processed/session/imec0_ks4")
spike_samples = np.load(folder / "spike_times.npy", allow_pickle=False).reshape(-1)
spike_clusters = np.load(folder / "spike_clusters.npy", allow_pickle=False).reshape(-1)
groups = pd.read_csv(folder / "cluster_group.tsv", sep="\t")

# Boolean indexing is comparable to MATLAB logical indexing.
good_ids = groups.loc[groups["group"] == "good", "cluster_id"].to_numpy()
good_ids = np.intersect1d(good_ids, np.unique(spike_clusters))

# fs must be read from THIS recording's metadata, not guessed.
# In the checked example it is 30000 samples/s.
fs = 30000.0
one_neuron_id = good_ids[0]
one_neuron_spikes_s = spike_samples[spike_clusters == one_neuron_id] / fs
```

The comparison `spike_clusters == one_neuron_id` produces one Boolean per spike.
Selecting those rows extracts every event currently assigned to that neuron.
Dividing sample indices by samples/second converts them to seconds.

The complete script reads sampling rate from the AP metadata when available,
otherwise from `params.py`, and reports any discrepancy. It reads the binary's
file size to establish the available duration without loading its voltage data.
It never guesses duration from the last spike, because a recording can end in
silence.

`params.py` is parsed as literal settings, not executed. Pickled `ops.npy`
objects are unnecessary for this lesson. Input NumPy arrays are opened read-only.

## 3. How Do We Know the Phy Changes Were Saved?

Phy does **not** generally create a second, separate "modified recording".
Saving overwrites its current cluster assignments and cluster-label tables.
These are the files downstream analysis must reload.

There are two distinct operations:

| Operation in Phy | Expected saved change |
| --- | --- |
| Reclassify `good` to `mua` or `noise` | `cluster_group.tsv` changes; spike membership can stay identical |
| Merge or split clusters | `spike_clusters.npy` changes; cluster IDs and labels can also change |

Therefore, identical `spike_clusters.npy` and `spike_templates.npy` **do not mean
your classification edits were lost**.

In the inspected mouse 52488 example:

| Label source | Good clusters |
| --- | ---: |
| Automatic Kilosort | 250 |
| Automatic Bombcell | 160 |
| Saved Phy | 124 |

Relative to Bombcell, **36 good clusters and one non-somatic cluster were saved
as noise**, giving 37 changed classifications. The Phy log independently records
the corresponding move-to-noise actions. The per-spike assignment arrays match,
consistent with classification changes rather than merges or splits.

Kilosort-to-Phy label differences alone include Bombcell's automatic changes,
so that comparison is NOT a pure measure of your manual work. Compare the saved
groups to Bombcell as well, and inspect the Phy log as corroborating evidence.

The script exports a per-cluster comparison and input SHA256 fingerprints.
Automatic labels and template assignments are useful references, **not a complete
pre-curation archive**. No such versioned archive was found beside this example.
If you have a real pre-Phy `spike_clusters.npy` backup, set
`ORIGINAL_CLUSTERS_PATH` to it. It must use the same spike count and row order.
For other sorters or previously modified reference files, provenance must be
checked separately. ID renumbering alone is not evidence of biological changes.

Always save in Phy before extraction. An unsaved edit visible only in a Phy
window cannot be recovered by reading disk files. The script checks that its
source files have not changed while it runs. Before future curation passes,
preserve a dated copy of the existing sorter outputs, including arrays and TSVs.

## 4. What Does the Firing-Rate Matrix Mean?

With the default settings, the interval is **[0, 300) seconds**:

- Rows: good neurons, ordered by ascending cluster ID.
- Columns: consecutive one-second bins, `[0,1)`, `[1,2)`, ..., `[299,300)`.
- Value: spikes in the bin divided by its duration in seconds, measured in Hz.
- A zero means no detected spikes in a recorded bin, not missing data.
- A spike exactly at 300 seconds is outside this analysis window.
- No smoothing or interpolation is applied to the firing-rate matrix.

For example, six spikes in a one-second bin give 6 Hz. Six spikes in a
half-second bin give 12 Hz. In a one-second-bin matrix the numbers equal the
counts, but their units and interpretation differ.

The implementation bins using sample indices, then divides by bin durations.
This also avoids putting a spike at exactly 0.3 s into the wrong sub-second bin
because of computer floating-point rounding.

For the default example the result has **124 rows and 300 columns**. The separate
vector `mean_firing_rates_hz` contains each neuron's total first-window count
divided by 300 seconds. For unequal-duration bins, do not take an unweighted
mean of bin rates: the script uses total count / total duration instead.

Time zero is the **first saved AP sample**, not the start of acquisition and not
necessarily the behavioral TTL onset. The metadata's `firstSample` is not added
to these local spike times. To analyze five minutes of behavior instead, first
map the behavioral start into this clock. This tutorial deliberately does not
attempt behavioral synchronization.

A recording shorter than the requested interval causes an explicit error;
missing time is not padded with zeros. Good labels for clusters that have no
spikes anywhere in the recording do not inflate the count. Clusters that are
present in the full recording but silent in the first five minutes stay as
zero-valued rows. Missing manual labels are treated as unsorted, not replaced
with automatic "good" labels.

## 5. Outputs and MATLAB Equivalents

Each execution creates a new timestamped folder beneath
`NeuroPyGuiN_output/phy_spikes_tutorial/`. It never overwrites earlier results.

| Output | Contents |
| --- | --- |
| `firing_rates.mat` | MATLAB matrices, IDs, bin timing, sampling rate and source path |
| `firing_rates.npz` | The same named arrays in a compressed NumPy archive |
| `firing_rates.csv` | One row per neuron; first column is its cluster ID |
| `time_bins.csv` | Exact time limits, centers and durations for each matrix column |
| `neuron_summary.csv` | Python/MATLAB row lookup, counts and mean Hz per good neuron |
| `curated_good_spikes.npz` | All saved-good spikes over the FULL recording, as flat event arrays |
| `curation_comparison.csv` | Current clusters and available automatic classifications |
| `changed_cluster_labels.csv` | Rows with a label difference against at least one reference |
| `changed_vs_bombcell.csv` | Only Bombcell-to-Phy differences, when that reference is available |
| `changed_vs_kilosort.csv` | Only Kilosort-to-Phy differences, when that reference is available |
| `label_source_counts.csv` | Good-cluster counts for each available label source |
| `spike_table_first_10_rows.csv` | First ten detected events, for inspecting the data structure |
| `analysis_audit.json` | Parameters, source fingerprints, comparisons and validation results |
| `curation_and_firing_rates.png/.pdf/.svg` | Label-count comparison, rate heatmap and mean-rate distribution |

The good-spike archive is intentionally NOT a pickled list of variable-length
arrays. It stores ordinary numeric arrays that can be loaded with
`allow_pickle=False`. In an IDE, `good_spike_trains_s` is also available as a list
of one array per neuron, similar to a MATLAB cell array.

Load results in Python:

```python
import numpy as np

with np.load("path/to/firing_rates.npz", allow_pickle=False) as result:
    rates = result["firing_rates_hz"]
    ids = result["good_cluster_ids"]
    first_neuron_first_ten_seconds = rates[0, :10]
    actual_cluster_id_of_first_row = ids[0]
```

Load the same matrix in MATLAB:

```matlab
result = load('path/to/firing_rates.mat');
rates = result.firing_rates_hz;
ids = result.good_cluster_ids;
first_neuron_first_ten_seconds = rates(1, 1:10);
actual_cluster_id_of_first_row = ids(1);
```

Python starts indexing at 0; MATLAB starts at 1. **Do not add one to cluster
IDs**: their values identify clusters, independently of the indexing language.
Python `rates[0, :10]` corresponds to MATLAB `rates(1, 1:10)`.

### A Small Python-to-MATLAB Dictionary

| Python expression or concept | MATLAB analogy |
| --- | --- |
| `import numpy as np` | Make numerical functions available under the short name `np` |
| `array.shape` | `size(array)` |
| `np.unique(cluster_ids)` | `unique(cluster_ids)` |
| `samples[clusters == 7]` | `samples(clusters == 7)` |
| `counts.sum(axis=1)` | `sum(counts, 2)`, one sum per neuron |
| `results["good_cluster_ids"]` | Access a named field, like `results.good_cluster_ids` in a struct |
| A Python list of spike-time arrays | A MATLAB cell array with one spike train per cell |
| A pandas DataFrame | A MATLAB table with named columns |

A NumPy vector with shape `(4854461,)` is **one-dimensional**. It is not
automatically a row or a column vector. The script explicitly chooses columns
when writing one-dimensional vectors into MATLAB. The rate matrix is genuinely
two-dimensional, with shape `(124, 300)`.

Python slices exclude the final index: `array[:10]` selects positions 0 through
9, exactly ten values. Indentation groups code into a loop or function, serving
the role that `end` plays in MATLAB. A line beginning with `#` is a comment;
triple-quoted text directly below `def` documents a function.

## 6. Why These Libraries and This Structure?

- **pathlib**: robust paths, including Windows paths, without hand-built separators.
- **NumPy**: directly reads the sorter's arrays and performs Boolean indexing and histograms.
- **pandas**: readable label tables with explicit cluster-ID matching and CSV output.
- **SciPy**: creates a MATLAB `.mat` file so you can inspect familiar matrices immediately.
- **matplotlib**: produces the same compact figure as PNG, PDF and editable SVG.
- **ast/json/hashlib**: standard-library tools for safe settings parsing, provenance and fingerprints.

The script has no dependency on the NeuroPyGuiN GUI, Phy itself, a GPU, or the
sorting engine at analysis time. It makes the core file-reading operations
visible instead of hiding them behind a large neuroscience framework.
This is appropriate for learning a small, already-sorted dataset. For large
multi-session pipelines, a structured framework can become useful later.

Functions keep loading, binning and plotting separately testable. The main
function follows numbered analysis steps; `# %%` sections help navigation in
IDEs. The variables assigned at the end remain visible in IDEs that retain
the script namespace, such as Spyder.

These are descriptive counts and rates, not a comparison between experimental
groups. No significance tests or p-values are warranted. A good label is a
curation decision, not a guarantee of biological identity or recording stability.

## Sources

- [Phy: saving and output files](https://phy.readthedocs.io/en/latest/outputs/)
- [Phy: sorter templates versus editable clusters](https://phy.readthedocs.io/en/latest/customization/)
- [Kilosort 4: exported Phy files](https://kilosort.readthedocs.io/en/stable/export_files.html)

The installed example's actual shapes are checked instead of assuming every
sorter version has identical file shapes. Unsupported multi-column per-spike
arrays cause an error rather than silent reshaping.
