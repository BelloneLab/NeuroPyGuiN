# Post-Processing Panel Overhaul - Progress & Audit

Living document tracking the audit and rebuild of the NeuroPyGuiN **Post Processing**
tab so that (a) its figures match NeuroPyxels (`npyx`) output, (b) the panel is
genuinely user-friendly, and (c) it works with real lab data (curated KS4 + a binary
behavior matrix).

Author: Andrianarivelo. Started 2026-06-29.

---

## 1. Goal (from the request)

> The NeuroPyxels integration is not good and the post-processing panel content is bad.
> Pull the latest NeuroPyxels, audit the post-processing of the app and its GUI
> friendliness, take control of the app, test with real spike-sorted + behavior data,
> make the figure output really match NeuroPyxels output, and check figure/analysis
> quality with screenshots, iterating until it is perfect.

Success = a post-processing panel a systems neuroscientist trusts: npyx-faithful ACG/CCG/
waveform/raw figures, a behavior-aligned PSTH that actually consumes our behavior format,
a curated (not kitchen-sink) advanced-analysis menu, and verified screenshots.

---

## 2. Confirmed facts (ground truth)

### Environment
- App + npyx run in conda env **`neuropygui`** (`C:\Users\bellone\.conda\envs\neuropygui\python.exe`, Py 3.9, numpy<2 pinned).
- `matplotlib 3.9.3` with the **QtAgg backend works** in this env -> we can embed *real* npyx
  matplotlib figures inside the Qt GUI via `FigureCanvasQTAgg`. This is the faithful path to "match npyx output".
- Headless screenshots: follow `tools/histology_verify.py` pattern (`QT_QPA_PLATFORM=offscreen`, `widget.grab().save(...)`).

### NeuroPyxels version (RESOLVED: already latest)
- Vendored copy at [npyx/](npyx/) is **4.2.0**, which **equals current GitHub master (4.2.0)** and is *ahead* of the
  PyPI release (4.1.3). npyx is NOT pip-installed in the env; the bridge imports the vendored copy via `sys.path`.
- File-level diff vendored vs upstream master (ignoring CRLF): **content-identical except 7 files with tiny local
  patches** that are deliberate and worth keeping: try/except around `IPython.display`/`IPython.core.debugger`/
  `cmcrameri` (so npyx runs headless / outside Jupyter), a `cupy` import guard with a better message, and an
  `fs=30000` fallback in `spk_t` so functions don't require a `.meta`. These patches are exactly what embedding
  npyx in a headless Qt app needs.
- **Conclusion:** we are already on the latest npyx; upstream master has nothing newer to pull, and overwriting
  would lose the headless patches for zero gain. Keep the vendored copy as-is.

### npyx figure embedding (verified)
- `npyx.plot.plot_acg(dp, unit, ..., ax=None, train=None, prettify=True, normalize='Hertz') -> matplotlib Figure`
  and `plot_ccg(dp, units, ..., ax=None, trains=None, style='line'/'bar') -> Figure`. Both accept `ax=` and
  **return a real matplotlib figure**, and accept externally-fed spike trains (`train=`/`trains=`, in samples) so
  they render from the loaded dataset without re-reading disk. `FigureCanvasQTAgg(fig)` embeds them in Qt.
- This is the faithful path: the app will show *actual npyx figures*, not pyqtgraph look-alikes.

### Test data
- **Spike sorting (KS4):** `B:\NPX\processedData\VTA_NPX\29237\2\spike_sorting\catgt_29537_2_trial1_g0\29537_2_trial1_g0_imec0\imec0_ks4`
  - 367 units, 3.66M spikes, **~1512 s** (25 min), fs = 30 kHz, 385 channels.
  - Curation present: phy `cluster_group.tsv` -> **77 good**, 290 noise; `cluster_KSLabel.tsv` -> 184 good / 183 mua.
  - No bombcell labels in this folder (Good-source "Auto" should fall back to phy/KSLabel).
  - AP binary present at `..\29537_2_trial1_g0_tcat.imec0.ap.bin` (+ `.ap.meta`) -> Raw Explorer is usable.
  - Good demo units (high FR): 8 (100 Hz), 166, 154, 175, 120, 213, 221.
- **Behavior:** `Z:\#SHARE\AndryA\PROJECTS\processedData\VTA_NPX\29537\2\dlc\behaviors_binary_threshold_v2.csv`
  - 45,302 rows @ **30 fps**, `time` spans -0.066 -> 1509.96 s -> **already in the spike time base** (no extra alignment needed).
  - 12 **binary** behavior columns (0/1 per frame): approach, withdrawal, nose_to_nose, anogenital_sniff,
    nose_to_body, follow, chase, escape, no_contact, freeze_resident, side_by_side, passive.
  - Real onset counts (0->1 transitions) = the actual PSTH "trials": approach 88, nose_to_nose 85,
    anogenital_sniff 93, follow 115, no_contact 64, escape 50, side_by_side 38, freeze_resident 28,
    withdrawal 22, chase 20, passive 22, nose_to_body 59.

---

## 3. Architecture (as found)

- [neuropyguin/tabs/postprocessing_tab.py](neuropyguin/tabs/postprocessing_tab.py) (~2600 lines): the Qt GUI. Six views
  (Unit Basics, Raw Explorer, Correlogram, Condition PSTH, Network, Advanced Corr), all drawn in **pyqtgraph**.
- [neuropyguin/postproc_engine.py](neuropyguin/postproc_engine.py): `NeuropixelsDataset` loader + **hand-rolled** numpy
  ACG/CCG/ISI/PSTH/synchrony. Core figures do NOT use npyx.
- [neuropyguin/npyx_corr_bridge.py](neuropyguin/npyx_corr_bridge.py): wraps ~50 `npyx.corr` functions for the
  "Advanced Corr" dropdown; returns "payload" dicts rendered by the tab.
- [neuropyguin/postproc_events.py](neuropyguin/postproc_events.py): event-CSV loader (time/label column detection).

---

## 4. Audit findings

### CRITICAL
1. **Behavior/event loader is broken for our data format.** `postproc_events.py` looks for a non-numeric
   "label" column; our behavior columns are all numeric 0/1, so `detect_event_label_column` returns `None`
   and `load_event_times` returns **every one of the 45,302 frame timestamps** as "events". Condition PSTH
   is meaningless on real data. Fix: detect binary behavior matrices, convert 0->1 transitions to onset times
   per selected behavior column; let the user pick behavior(s); support optional duration/offset.
2. **Core figures are not npyx and cannot match it.** ACG/CCG/waveform/raster/PSTH are custom numpy in pyqtgraph.
   npyx normalizes ACG/CCG in **Hertz** by default and has signature plotting styles; the app shows raw counts
   with different conventions. Fix: route computation through npyx (`acg`/`ccg`, `normalize='Hertz'`) and/or
   embed npyx's matplotlib `plot_acg`/`plot_ccg`/`plot_wvf`/`plot_raw` so output is literally npyx.

### HIGH
3. **Advanced Corr is a kitchen sink.** ~50 entries, many of which are internal helpers
   (`get_log_bins_samples`, `get_ccgstack_fullname`, `canUse_Nbins`, `make_phy_like_spikeClustersTimes`,
   `par_process`, `cisi_chunk`, `get_ustack_i`) or return only `str(type(out))` text. Overwhelming and
   untrustworthy. Fix: curate to ~10-12 real analyses with good defaults and proper plots; hide plumbing.

### Audit verdict (5-way parallel review, all claims source-verified)
Three compounding problems justify "not good at all":
1. **Normalization schism.** `postproc_engine.correlogram` returns raw `np.histogram` counts
   ([postproc_engine.py:297-332](neuropyguin/postproc_engine.py#L297)) used by Unit Basics ACG, the Correlogram
   tab, and the Network matrix, while npyx ACG/CCG are in **Hertz**. The same unit's ACG looks different per tab
   (counts on two tabs, spk/s on Advanced Corr), and the count scale tracks the bin spinbox, not the neuron.
2. **The npyx bridge is a raw namespace dump,** ~50 entries including private numba kernels (`cisi_numba`),
   joblib workers (`par_process`), cache-name builders (`get_ccgstack_fullname`), feasibility booleans
   (`canUse_Nbins`), and ~5 branches that emit `str()`/`type()` text. The renderer also discards npyx's real
   lag-ms / firing-rate-Hz axes (labels everything "x"/"value"/"row"/"col") and hardcodes `fs=30000` in 8 places.
3. **It does not work with the user's real data** (confirmed on screen: PSTH banner shows **45302/45302 trials**).

### npyx reference facts (for the implementation)
- `npyx.corr.acg/ccg`: `normalize in ['Counts','Hertz','Pearson','zscore']` (default Hertz). `trains=` kwarg is in
  **samples**. Hz formula: `counts/(Nspk_ref * bin_ms/1000)`. ACG = `ccg(dp,[u,u])[0,0]`.
- `npyx.behav.get_processed_ifr(times_s, events_s, b=10, window, zscore, zscoretype, convolve, gsd_ms,
  bsl_subtract, bsl_window) -> (x_ms, y_raw[trials x bins], y_mean, y_sem)` - the right primitive for the PSTH.
  `align_times(times_s, events_s, b, window) -> (aligned_t, aligned_tb counts)`.
- `npyx.plot.plot_acg(dp,unit,...,ax=,saveFig=False,normalize='Hertz') -> Figure` (accepts `ax=`).
  `plot_ccg(dp,units>=2,...,saveFig=False) -> Figure` (no `ax=`). `plot_wvf(dp,u,...,saveFig=False)` (no `ax=`,
  draws uV scale bar). `plot_cm(dp,units,...)` correlation matrix. **`plt_ccg` defaults `saveFig=True`** (writes a
  PDF) - always go through `plot_ccg(..., saveFig=False)`. Embed via `FigureCanvasQTAgg(fig)`.
- npyx reads this KS4 folder directly as its `dp` (the Advanced Corr tab already does), so npyx-native figures need
  no extra conversion; first call caches into `<dp>/.NeuroPyxels/`.

---

## 5. Plan

1. [done] Recon: architecture, env, npyx version, test-data ground truth.
2. [in progress] Deep audit (parallel workflow) of tab plotting, npyx plot/corr/behav API, bridge, app-drive recipe.
3. [todo] Pull/diff npyx vs upstream master; sync if needed.
4. [todo] Build `tools/postproc_verify.py` headless screenshot harness for all six views on the real data.
5. [todo] Fix behavior/event handling (binary-matrix aware) + Condition PSTH.
6. [todo] Make core figures npyx-faithful (embed npyx matplotlib figures and/or align computations + styling).
7. [todo] Curate + clean the Advanced Corr menu.
8. [todo] GUI friendliness pass.
9. [todo] Iterate on screenshots until figures match npyx and look publication-clean.

---

## 6. Progress log

- 2026-06-29: Recon complete. Confirmed env, npyx 4.2.0 == master, full test-data ground truth (section 2).
  Identified the broken event loader and the non-npyx core figures as the two critical issues. Launched a
  5-way parallel audit workflow (tab plotting / npyx plot API / npyx compute / bridge / app-drive recipe).
- 2026-06-29: Audit complete (verdict + npyx refs above). Built `tools/postproc_verify.py` headless harness;
  captured baseline screenshots of all 6 views on the real data (`tools/_postproc_shots/fonttest/`). Baseline
  confirms on screen: PSTH "45302/45302 trials" (every frame = an event), ACG/ISI in raw counts, IFR squished
  into the raster. Note: Qt `offscreen` renders text as tofu on Windows; harness now defaults to the `windows`
  platform for readable labels.
- 2026-06-29: Implemented and verified Stages 1-5 (see "Implemented" section). Critical behavior-PSTH bug fixed
  (88 onsets vs 45302 frames); Correlogram tab renders real npyx ACG/CCG grids; Advanced Corr curated to 14
  analyses with real axes/colorbars and the true sample rate; Network/PSTH labelled with colorbars. 14/14
  post-processing tests pass. Net new files: `neuropyguin/npyx_figures.py`, `tools/postproc_verify.py`; local
  npyx patch: `npyx/inout.py` (KS4 1D channel_map).

### Implemented (verified on the real dataset via tools/postproc_verify.py)
Screenshots in `tools/_postproc_shots/` (`fonttest` = before, `stage5` = after).

- **[DONE, critical] Binary behavior-matrix ingestion.** Rewrote [postproc_events.py](neuropyguin/postproc_events.py):
  detects a wide 0/1 matrix, exposes each behavior column as a selectable label, and derives event times from
  bout onsets (rising 0->1), offsets (falling), or bout midpoints, using the file's own time column (or a
  frame-rate fallback). Condition PSTH gained an alignment selector, a frame-rate spinbox, a baseline-subtract
  option, and an "Add all behaviors" button. **Result:** PSTH on "approach" now uses **88 real onsets** instead of
  45,302 frames; the line shows true peri-event structure and the heatmap carries a Rate-(Hz) colorbar.
- **[DONE, critical] npyx-native correlograms.** New [npyx_figures.py](neuropyguin/npyx_figures.py) embeds real
  `npyx.plot.plot_acg`/`plot_ccg` figures via `FigureCanvasQTAgg`. The Correlogram tab now renders the canonical
  npyx grid (Hz ACGs on the diagonal, z-scored CCGs off-diagonal, peak-channel labels like `8@328`) with a
  matplotlib toolbar and Normalize/Style controls. This is literally npyx output inside the app.
- **[DONE] npyx KS4 patch.** Patched vendored `npyx/inout.py chan_map` to accept 1D KS4 `channel_map.npy`
  (was an AxisError); this also unblocks peak-channel labels and any get_depthSort_peakChans use.
- **[DONE] Engine Hz normalization.** `postproc_engine.correlogram` now defaults to npyx's Hertz convention
  (`counts/(N_ref*bin_s)`), so the Unit Basics ACG (relabelled "Firing rate (Hz)") and the Network matrix agree
  with npyx and are bin-invariant.
- **[DONE] Advanced Corr curation + correctness.** Trimmed the bridge menu from ~50 raw-namespace entries to **14
  curated analyses** (ACG, CCG, correlation matrix, Pearson, correlation index, cross-ISI, synchrony, population
  coupling, STTC, significant CCG pairs, Stark-Abeles predictor, scaled ACG, 3D ACG/CCG); hid the numba/joblib/
  cache-name/feasibility plumbing. Threaded the dataset `sample_rate` through `run_method` (removed `fs=30000`
  hardcodes), made the renderer honor the payload's real axis labels + coordinates (was "x/value/row/col") and
  added colorbars to image results. Fixed `correlation_index` scalar-vs-matrix on unit count.
- **[DONE] Network labels.** Matrix retitled "Pairwise peak CCG rate (Hz)" with a colorbar and unit-ID ticks;
  synchrony plot axes labeled.
- **[DONE] Tests.** Added regression tests for behavior-matrix detection, onset/offset extraction, long-format
  fallback, and Hz bin-invariance. 14/14 post-processing tests pass.

- **[DONE] Layout overhaul (visuals-first, npyx aesthetic).** Restructured the tab from three roughly-equal
  columns (Units | Controls | Plots) into a **compact left sidebar + large figure canvas**: Units list (now with a
  scrollable, taller list) stacked above the Analysis Pages controls (each control page wrapped in a QScrollArea so
  tall pages like Condition PSTH never starve the unit list). The sidebar is width-capped (~310-560 px) so the
  **figures always get ~75% of the width**. Tightened margins, hid the boxed top/right axes, set a clean
  Segoe-UI tick/label font, gave the Unit Basics raster + ACG/waveform/ISI row much more height, and gave the
  Network matrix the dominant share over the synchrony trace. Also fixed a waveform-panel bug: it now always
  reflects the selected unit and shows "No template waveform for unit N" instead of a stale/blank panel.

- **[DONE] Unit Basics + Raw Explorer renewed as npyx-style figures (two specialist agents).**
  New module [neuropyguin/unit_figures.py](neuropyguin/unit_figures.py) builds publication-clean matplotlib
  figures embedded via `NpyxFigureView`:
  - **Unit Basics** is now a 5-panel npyx/phy "unit card" for the selected unit(s): mean waveform on real probe
    (x,y) geometry with peak channel highlighted + uV/ms scalebar; Hz ACG (via npyx `plot_acg`) with a 2 ms
    refractory band; log-x ISI with a refractory marker + violation %; per-spike amplitude over the whole session
    (drift); and session firing rate. Title carries unit id, peak ch, n spikes, mean Hz, refractory %. Fixes the
    old empty-1s-raster problem by favouring session-wide panels; degrades gracefully for noise/no-waveform units.
  - **Raw Explorer** is an npyx `plot_raw_units`-style stack of HP-filtered channels around the peak channel with
    per-unit spike overlays, channel/depth y-axis, and a uV/ms scalebar. Cleaner defaults (32 ch, 0.15 s).
- **[DONE] Tab reorganization (UI specialist).** Removed the confusing redundant second tab bar (the right view
  `QTabWidget` is now a hidden-tab stacked container driven by the single Analysis-Pages tab bar). The **Units
  list is now a prominent, always-visible, tall left column** (filter, Good-only, Good-source, ~18 visible units +
  a live count, metric table) instead of being squeezed under the controls. Figures get ~75% width. Removed the
  obsolete "ACG:ISI ratio" controls and the dead pyqtgraph Unit Basics/Raw widgets + helpers. Other tabs
  (Correlogram npyx grid, Condition PSTH, Network, Advanced Corr) preserved; 14/14 tests pass.

- **[DONE] Figure + layout polish round (two specialist agents + manual Correlogram fix).**
  - **Waveform mean ± SEM band:** the unit card now extracts up to 300 real spike snippets from the AP binary
    and draws each channel's mean with a shaded ±SEM band (cleanliness indicator). Real peak ~114 uV vs the
    template's ~14 uV, and noisy units show visibly wider bands. Falls back to the template line ("no SEM") if the
    AP binary is unreadable.
  - **Unit-card overlaps fixed:** removed the redundant subtitle, switched to `constrained_layout`; panel titles no
    longer collide with the axes above them.
  - **Sexy multi-unit PSTH:** new `condition_psth_figure(psth_results, mode, baseline, trial_from, trial_to, dark)`
    with a **Display mode** selector - "Average across units" (across-units mean ± SEM band + faint individual
    unit traces) or "Per-unit panels" (a grid of per-unit mean ± SEM, conditions overlaid) - both over a per-unit
    heatmap (conditions as labelled row-blocks) with a colorbar. Baseline -> diverging cmap + Δ Rate axis.
  - **Correlogram ACG grid fixed** ([npyx_figures.py](neuropyguin/npyx_figures.py)): each unit in a distinct
    Okabe-Ito colour with a colour-matched "unit N" title, top/right frames removed, single shared
    "Autocorrelation (spk/s)" / "Time (ms)" labels (no more misplaced/overlapping y-label).
  - **Thinner chrome:** controls strip capped ~200 px with compact forms; log is a thin collapsible strip
    (splitter 9:1); figures gain vertical space. Raw Explorer defaults to 32 ch / 0.15 s for readability.
  - 18/18 tests pass; harness renders all tabs cleanly; no dangling references to the removed pyqtgraph widgets.

- **[DONE] Collapsible right-side settings panel (UI specialist).** The per-analysis settings forms moved out of
  the top strip into a card-style **collapsible panel docked on the right** (`settings_stack`/`settings_panel`),
  toggled by one discrete auto-raise chevron (`›` open / `‹` collapsed, state persisted). Navigation was separated
  from settings: a slim top nav tab bar (`analysis_tabs`, empty pages, ~52 px, still the authoritative index) drives
  both the figure stack and the settings stack, so switching analyses works even while settings are hidden. When
  collapsed the figure canvas takes the full width. Detach/Attach repointed to the new figure row. Harness adds
  collapsed-state captures (`6_settings_collapsed.png`, `6b_collapsed_nav_switch.png`); 14/14 tests pass.

- **[DONE] CCG grid rebuilt, Network rethought, Advanced pruned, C4 cell-type panel added.**
  - **CCG grid** ([npyx_figures.py](neuropyguin/npyx_figures.py) `ccg_grid_figure`): replaced the unreadable
    npyx `as_grid` (64 overlapping panels for 8 units) with a clean custom NxN grid (cap 6) - diagonal ACGs in
    distinct Okabe-Ito colours, off-diagonal CCGs, unit-id row/col headers, shared outer labels, despined.
  - **Network** ([postproc_engine.py](neuropyguin/postproc_engine.py) `network_analysis` + `unit_figures.network_figure`):
    replaced the raw-count peak matrix + ad-hoc CV with a hierarchically-sorted spike-count (noise) correlation
    matrix, Okun population-coupling vs depth, and a putative monosynaptic-connection matrix (short-latency CCG
    z-deviation) with a significant-pair count.
  - **Advanced** (was "Advanced Corr"): pruned the bridge menu from ~50/14 to 6 unique, non-redundant analyses
    (3D ACG, 3D CCG, scaled ACG, Stark-Abeles monosynaptic significance, STTC, cross-ISI); correlation matrices /
    coupling moved to the Network panel, basic ACG/CCG to the Correlogram tab.
  - **C4 cell-type classifier** (new panel): runs the NeuroPyxels C4 Laplace-calibrated CNN ensemble (GoC/MLI/MFB/
    PkC_ss/PkC_cs) per unit. Because C4's `laplace` stack needs torch>=2.6 (incompatible with kilosort's CUDA torch
    2.5.1), it runs in an **isolated `npyx_c4` env via subprocess** ([c4_runner.py](neuropyguin/c4_runner.py)
    dispatcher -> [c4_bridge.py](neuropyguin/c4_bridge.py) worker), mirroring the phy/IBL isolation pattern.
    Verified end-to-end (5 units classified, 2020-model ensemble). `unit_figures.c4_figure` renders predictions +
    confidence + class-probability heatmap.

### npyx_c4 isolated environment (C4 classifier) - setup notes
- Env: `C:\Users\bellone\.conda\envs\npyx_c4` (python 3.10). Created with `conda create -p ... python=3.10` then
  `pip install "npyx[c4]"` (pins torch 1.13.1 CPU + laplace-torch 0.1a2 + backpack 1.6.0 - the C4-paper combo
  matching the cached `~/.npyx_c4_resources` models, ~2.7 GB).
- Two non-obvious fixes were needed after `npyx[c4]`: **`pip install "setuptools<80"`** (setuptools 81+ removed
  `pkg_resources`, which `backpack` needs) and **`pip install "scikit-learn<1.6"`** (1.6 removed `_safe_tags`,
  which `imbalanced-learn` imports via `npyx.ml`).
- The main `neuropygui` env is unchanged (torch 2.5.1+cu124, kilosort 4.0.37); C4 deps were removed from it after a
  laplace install briefly clobbered CUDA torch. Override the C4 interpreter with env var `NPYX_C4_PYTHON`.
- The bridge hardlinks the `.ap.bin` and copies the `.ap.meta` into the ks4 folder for the run (KS4 leaves them in
  the parent), then removes them.

- **[DONE] Export waveforms, second cell-type classifier (Bombcell), ACG cleanup.**
  - **Export waveforms** toolbar button: off-thread batch export of every good unit's compact
    waveform-on-geometry + Hz ACG card to per-unit PNGs and one `good_units_waveform_acg.pdf`
    (`unit_figures.unit_waveform_acg_figure` + a `_export_good_unit_figures` worker helper).
  - **Bombcell cell-type classifier** added alongside C4 in the renamed **Cell Types** panel: a Method
    selector (C4 / Bombcell) + Brain-region selector (Cortex / Striatum). [bombcell_classify.py](neuropyguin/bombcell_classify.py)
    computes bombcell ephys properties (cached per dp, ~1 min, runs in the MAIN env - threshold-based, no
    laplace) and classifies (cortex: Wide/Narrow-spiking; striatum: MSN/FSI/TAN/UIN). `unit_figures.bombcell_celltype_figure`
    renders a colored per-unit strip + the feature scatter with the threshold lines. Verified on real units.
  - **CSV + TSV output** for both classifiers: `cell_types_<method>.csv` (full fields) and a phy-compatible
    `cluster_<method>_cell_type.tsv` written into the dataset folder on each run.
  - **ACG decorations removed** (user preference): the shaded refractory band and the vertical dashed
    (0-lag / +/-2 ms) lines are gone from every autocorrelogram (unit card, export card, npyx ACG grid, CCG-grid
    diagonal); the off-diagonal CCG keeps its 0-lag reference line.
  - **Advanced** trimmed to the 4 analyses that actually work on this data (3D ACG, 3D CCG, Stark-Abeles
    monosynaptic significance, cross-ISI); `scaled_acg` and `STTC` are broken in vendored npyx 4.2.0 for this
    dataset (numpy/neo type bugs) and were removed from the menu.
  - **README** expanded with the Post Processing feature list + a "Cell-type classification (C4 and Bombcell)"
    section incl. the isolated `npyx_c4` env setup; real per-tab use-case screenshots regenerated via
    `tools/readme_screens.py`.

### Remaining optional polish (non-blocking)
- Unit Basics IFR overlay on its own right-hand Hz axis (currently scaled into the raster).
- Log-spaced ISI x-axis with a refractory marker; Raw Explorer uV scale bar.
- npyx-native waveform (`plot_wvf`) needs the `.ap.meta` staged into the ks4 folder (meta lives in the parent);
  the pyqtgraph template waveform is kept for now.
- Multi-unit PSTH: optional per-unit z-score via `npyx.behav.get_processed_ifr` / `summary_psth`.

### Prioritized implementation order (from the audit, most impactful first)
1. [critical] Binary behavior-matrix ingestion + behavior-column picker + frame-rate -> Condition PSTH works.
2. [critical] npyx-native figures: embed real `npyx.plot` ACG/CCG/waveform via FigureCanvasQTAgg (the gold
   standard for "match npyx"); unify engine correlograms on Hz for the remaining pyqtgraph/Network views.
3. [high] Advanced Corr renderer honors npyx axes; thread `dataset.sample_rate` through the bridge (kill 8x fs=30000).
4. [high] Curate bridge menu to ~12 real analyses; hide internal plumbing/text-only methods.
5. [high] Colorbars + value scales on all heatmaps (PSTH, Network, Advanced images).
6. [medium] Network npyx-faithful metrics (normalized matrix + `frac_pop_sync`, labeled axes).
7. [medium] Multi-unit PSTH z-score/baseline per unit (mirror `get_processed_ifr`); IFR own Hz axis.
8. [medium] ISI log axis + refractory marker; Raw Explorer uV scale bar.
