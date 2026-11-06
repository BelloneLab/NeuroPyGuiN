# Workspace audit and analysis upgrade

Audit date: 2026-11-06. Baseline: `01f7752` on `main`.

## What changed

### Postprocessing

- Compact Units, Voltage, Timing, Events, Network, Methods, Cell types and
  Population navigation, with full descriptions in tooltips.
- Searchable unit list, curated-label filtering, clear session summary, selection
  utilities, compact settings and analysis status. Hidden pages no longer force
  every page wider than the display. Checked at 1366 x 768 and 1440 x 960.
- Debounced automatic redraws, batched selection updates, background PSTH/network
  computation and bounded cached waveform summaries.
- Population rate maps, PCA, factor analysis, real NEMBA TCN/PCA/FA/GPFA adapters,
  continuous behavior alignment and blocked ridge decoding. See
  [Population analysis](POPULATION_ANALYSIS.md) for scientific assumptions.
- Actual waveform extraction respects `channel_map`; sorter templates carry
  arbitrary-unit labels, never an unsupported calibrated-voltage claim.
- Time-window, Gaussian rate and raw downsampling controls now affect the plot.
- Dataset changes clear previous figures/results. Disk cache identity includes
  source file size/timestamps and calibration. Reload after external Phy edits.
- Correlograms and event histograms search the relevant time window instead of
  subtracting every spike from every reference event. Repeated unit retrieval
  uses a bounded, read-only cache. Source arrays are not sorted in place.
- PSTHs cover the requested interval exactly. Short terminal bins use their
  actual duration, and floating-point rounding cannot add a phantom tail bin.
- Singleton spike/channel arrays load correctly. Relative `params.py` raw paths
  resolve against the recording. Literal parsing does not execute dataset code.
- Cache writes are atomic and optional, including for read-only recordings.
  Unreadable resolver candidates are skipped, symlink trees are not traversed,
  and searches stop once the closest matching root has been evaluated.
- Exploratory CCG deviation thresholds no longer masquerade as significant
  synaptic connections. They remain descriptive, uncorrected screening scores.

### Figures and navigation

PDF and SVG preserve editable labels; PNG exports use 600 dpi. White-paper
presets support 89 and 183 mm width, 7 pt type and thin strokes. CSV tables and
JSON manifests accompany available plotted data. Existing bundles receive
versioned names. Unit-atlas exports report failures per unit rather than silently
omitting them. Large matrices and point clouds can be rasterized while labels
remain vector artwork.

These presets follow the official [Nature figure preparation guidance](https://research-figure-guide.nature.com/figures/building-and-exporting-figure-panels/)
and [editable-artwork specifications](https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/).
A journal-ready scientific figure still requires suitable experimental design,
independent replication, an appropriate legend and inspection at final size.
A dense multi-panel plot may need 183 mm rather than 89 mm.

The shell includes a Ctrl+K action finder, shared light/dark themes, visible
keyboard focus, clearer spin controls, scrollable navigation and display-aware
startup geometry.

### Curation: retain useful decisions, remove invalid calculations

Retained: Bombcell quality metrics and threshold review, Phy launch, manual label
inspection, optional raw waveform SNR inspection, batch recordings and splitting
concatenated sessions. These address distinct scientific or workflow needs.

Removed: automatic PSD augmentation during quality-table loading. It assumed a
30 kHz sample clock and computed a 30-80 Hz quantity from a signal with a 50 Hz
Nyquist limit. This was scientifically misleading and blocked the GUI. Existing
measured Bombcell values remain available.

Recording switches now clear stale labels, metrics and file watchers. Empty paths
no longer silently select the current directory. Batch selection survives refresh;
Select All redraws once. Histograms reject non-finite values and align bars with
the actual bins. Unit identifiers are validated before writes.

### Preprocessing

Each queue uses a settings snapshot. Active queues reject clear/remove mutations.
Empty pipelines and output paths are rejected. Successful recordings leave the
pending queue; failures remain for retry. Logs are bounded. Existing raw and
metadata files were not altered by this audit.

## Verification and reproducibility

Final application suite: **348 passed, 4 skipped** in 17.72 seconds. The four
skips require Allen CCF atlas or reference files that are not present. Five
warnings concern third-party deprecations or unavailable NVML diagnostics.

The existing `neuropyguin` conda environment was used. NEMBA 0.3.0 was installed
from commit `78ccb6b32b24e8c0ef0d17dcd6b839d20645e954` with `--no-deps`, preserving
installed dependencies. Actual NEMBA PCA and TCN calls returned finite 300 x 3
representations. The TCN check trained for 300 steps in about 13 seconds on this
machine. Optional FA/GPFA adapters share the verified API but were not separately
benchmarked here.

Run application regressions with:

```bash
QT_QPA_PLATFORM=offscreen conda run -n neuropyguin python -m pytest -q
conda run -n neuropyguin python tools/benchmark_postproc.py
```

`pytest.ini` scopes default discovery to the application's `tests/` directory.
Vendored external test trees require their own package setup and are outside this
suite. Windows-only path literals in helper tests were made native, retaining
folder-layout assertions on Linux and Windows.

New tests cover cache invalidation, read-only/corrupt caches, singleton loading,
raw path resolution, CCG pair-count equivalence, PSTH exposure, source immutability,
publication sizing/restoration, channel mapping, dataset isolation, queue safety,
behavior gaps, blocked model evaluation, process lifecycle and keyboard navigation.
The full-window smoke check rendered all four workflow tabs. Figure and UI checks
used explicit synthetic fixtures, not experimental observations.

## Performance measurements

Measurements are descriptive microbenchmarks from this machine. They do not
predict full-recording throughput, GPU sorting performance, or a UI-wide speedup.

| Operation | Before median | After median | Ratio |
| --- | ---: | ---: | ---: |
| CCG, 12,000 synthetic spikes | 39.098 ms | 12.697 ms | 3.08x |
| PSTH, 400 events | 6.608 ms | 5.319 ms | 1.24x |
| 100 repeated unit retrievals | 4.262 ms | 0.107 ms | 39.84x |
| Curation Select All, 500 units / 12 metrics | 677.746 ms | 90.672 ms | 7.47x |
| Cached waveform helper | 2.739 ms | 0.0132 ms | 207x |

The first three use a fixed seed, three repetitions and checked numerical
agreement with the baseline. Curation uses three repetitions; the waveform
helper uses 20. No inferential p-values are appropriate for these small local
timing samples. Raw timings and comparison figures are saved locally as CSV,
PNG, PDF and SVG under ignored `tools/_postproc_shots/` and
`tools/_verify_shots/workflow_audit/`.

## Remaining scope

This release provides a coherent spike and continuous-behavior workbench. It does
not implement every possible neuroscience analysis. Categorical behavior,
circular variables, GLMs, cross-session decoding, automated synchronization and
across-animal inference need dedicated designs. Spike timestamps alone do not
establish exact acquisition exposure at recording boundaries; event windows must
be chosen within verified recording coverage. No full experimental GPU sorting
job, C4 model download, or end-to-end histology registration was run in this audit.
Tests and rendering checks reduce risk; they cannot establish that every app bug
or every possible dataset edge case has been eliminated.
