# Population and behavior workspace

The **Population + behavior** analysis page extends Post Processing with a
bounded, cancellable population workflow. Open **Analysis settings**, start with PCA, choose an interval
and unit scope, and press **Analyze population**. Click a geometry point to
inspect its exact time across linked activity panels. Time axes share pan/zoom. The initial 60-second window
keeps exploratory work quick; the selected interval remains explicit.

## Representations and controls

- **PCA** is the transparent variance baseline. Standardizing units gives each
  variable unit equal variance; disable this to preserve relative rate variance.
- **Factor analysis** estimates shared variability with per-unit noise.
- **NEMBA** exposes the actual `embedders_backbones.Embedder.fit_matrix` API for
  TCN, PCA, FA and GPFA. Its input is standardized explicitly, with three latent
  dimensions by default, zero post-embedding smoothing, CPU execution, seed 0,
  and NEMBA's `fast` preset. Requested dimensions are clamped to available
  variable units. The internal three-state fit is not displayed or interpreted.
- Choose **All loaded units** or **Selected units** explicitly. All loaded units
  can include noise units, so select curated units for scientific interpretation.
- Constant units remain visible in the rate matrix but are excluded from fitting.
- The dense matrix is capped at 5 million entries and 100,000 time bins.
  Increase bin size, reduce the interval, or select fewer units for larger data.
- Computation runs in a separate process. **Cancel** stops it immediately.

Complete bins are half-open intervals `[start, end)`. An incomplete trailing bin
is omitted, with the effective end recorded in the manifest. No source spike,
behavior, or metadata file is modified. The worker opens spike arrays read-only.

## Behavior and evaluation

Use a CSV with `time_s` (or `time`) and numeric continuous target columns:

```csv
time_s,speed_cm_s,position_cm
0.0,1.4,5.0
0.1,1.8,5.16
0.2,2.0,5.35
```

Timestamps must already be in seconds on the spike recording clock. Synchronize
acquisition systems before import. The workspace does not infer clock offsets,
drift, units or synchronization. Timestamps must be finite, unique and strictly
increasing. Linear interpolation never extrapolates, crosses a missing value, or
bridges a gap above the explicit maximum interpolation gap. Coverage is shown.
Use this page for continuous variables, not categorical states or wrapped angles.

The geometry is a descriptive embedding of the complete selected interval. It is
**not** used as input to the predictive evaluation. To avoid that leakage, the
behavior panel independently uses unsmoothed raw firing rates and ridge
regression (`alpha=1`) with scaling fitted inside each training fold:

1. Five contiguous time blocks form held-out test sets.
2. A configurable purge gap excludes neighboring training bins on both sides.
3. A training-mean baseline provides an explicit comparison.
4. Unique circular shifts of the complete target produce a null distribution.
   The default requests 99 shifts, so the minimum p-value is 0.01 if all 99
   offsets are available. The actual shifts and scores are exported.

Set the purge gap longer than relevant temporal dependencies. Circular shifts
assume approximate stationarity; drift and shared trends can violate that
assumption. A single recording's score is not evidence across animals. If
missing coverage makes a shifted evaluation impossible, the null p-value is
withheld instead of selectively dropping null draws. Too little coverage or a
constant target disables decoding with an explanation while preserving the
aligned descriptive view. Fold scores are descriptive, not independent samples
for a t-test. There is one selected target per run; account for multiple testing
when comparing many targets, windows, models or settings.

## NEMBA installation

Reference inspected and exercised: [Andrianarivelo/NEMBA](https://github.com/Andrianarivelo/NEMBA),
version 0.3.0 at commit `78ccb6b32b24e8c0ef0d17dcd6b839d20645e954`.

For an environment that already contains NEMBA's required NumPy, SciPy,
scikit-learn, Matplotlib, pandas, Pillow and Torch >=2 dependencies, install only
the package so the sorting environment's Torch build is preserved:

```bash
python -m pip install --no-deps "git+https://github.com/Andrianarivelo/NEMBA.git@78ccb6b32b24e8c0ef0d17dcd6b839d20645e954"
```

This exact dependency-preserving installation was used in the existing local
`neuropyguin` conda environment. For a separate existing NEMBA environment, set
**Analysis settings > Advanced settings > NEMBA Python** to its absolute interpreter path, or use
`NEMBA_PYTHON`. A source checkout can be supplied via **NEMBA source** or
`NEMBA_PATH`. Source overrides must contain `embedders_backbones/__init__.py`.
The optional environment needs NumPy, pandas and scikit-learn as well as NEMBA's
dependencies. No GUI or HDF5 dependency is required by the worker.

NEMBA is optional. PCA, factor analysis and continuous behavior decoding remain
available without it. Trained representations are exploratory; compare them
against the baseline and validate across independent sessions before biological
interpretation. The `fast` preset has a time budget, so training results can vary
with hardware even when the seed is fixed.

## Figure and data export

The embedded paper canvas scrolls vertically on small displays to keep all four
panels readable. **Detach plots** offers a larger analysis window. The compact
workspace summary keeps the current model, unit scope, interval and bin width
visible while controls are in **Analysis settings**.

**Export figure + data** writes white-paper PDF, SVG and 600-dpi PNG figures,
CSV tables for all binned rates, embedding coordinates, predictions, fold scores
and null draws, plus a JSON manifest. Export names are versioned rather than
silently overwritten. The manifest records selected units, interval, bin width,
settings, sampling rate, software versions, source file sizes and modification times, behavior
coverage, and NEMBA package installation provenance when available.

Dense scatter points and heatmaps are rasterized inside vector figures to keep
files manageable; text and axes remain editable. Heatmaps average contiguous
display blocks above 2,500 time bins; scatter displays at most 8,000 evenly spaced
samples. Exported CSV data retain every analyzed bin.

## Verification

`tests/test_population_analysis.py` covers interval endpoints, incomplete bins,
allocation limits, interpolation gaps, missing values, blocked evaluation,
fold-local baselines, known-signal decoding, constant units, safe serialization,
the NEMBA API contract, and figure/data export. All fixtures are explicitly
synthetic, not user experimental data.
