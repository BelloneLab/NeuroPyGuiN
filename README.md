# NeuroPyGuiN

GUI for Neuropixels preprocessing, curation, quality metrics, and post-processing.

![NeuroPyGuiN](./neuropyguin/assets/big.jpg)

## Features

### Preprocessing
- Add AP `.bin` files by drag/drop, file picker, or folder scan.
- Queue-based execution with live log/progress.
- Integrated `ecephys_spike_sorting` modules:
  - `catGT_helper` (with retry logic and failure gating)
  - `ks4_helper` / `kilosort_helper`
  - `kilosort_postprocessing`
  - `noise_templates`
  - `mean_waveforms`
  - `quality_metrics`
  - `tPrime_helper`
- Auto-generated input/output JSON per run.
- CatGT success is enforced before downstream ecephys spike sorting when CatGT is enabled.
- Built-in CatGT and TPrime string builders so users can generate common command fragments, including analog/digital and bit-field CatGT extractors, from structured inputs instead of typing raw flags.

### Curation
- Launch `phy template-gui params.py`.
- Bombcell thresholds editor with:
  - live preview of labels
  - histogram view of selected metric + threshold lines
  - `Apply settings` + optional live update
  - auto quality-metrics computation if `metrics.csv` is missing
  - `Run py_bombcell` to compute BombCell metrics + plots (`bombcell/templates._bc_qMetrics.csv`)

### Quality Metrics
- Run/re-run quality metrics.
- Optional `Run py_bombcell` backend for BombCell-native quality metrics and plots.
- Interactive table + filter + metric histogram.

### Post Processing
- Auto-compute and auto-plot workflow (no separate compute buttons).
- Unit list with filters and `Show good units only`.
- Export selected dataset units to nested HDF5 (`all units` or `good units only`).
- Unit quality side panel (labels + key metrics).
- Unit basics:
  - raster for selected units
  - optional instantaneous firing-rate overlay
  - mean waveform with `mean +/- SEM`
- Raw explorer with settings-driven refresh and channel/depth Y-axis modes.
- Correlogram with histogram + alternate line-style cross-correlogram view.
- Condition PSTH and Network views auto-refresh on relevant changes.

### Global plot settings
- App-level controls:
  - Light/Dark plot theme
  - Grid on/off
- Applied across curation, quality metrics, and post-processing plots.

## Dependencies

Core Python packages are listed in [`requirements.txt`](./requirements.txt):
- `PySide6`
- `pyqtgraph>=0.14.0,<0.15`
- `numpy`
- `pandas`
- `scipy`
- `matplotlib`
- `tqdm`
- `numba`
- `scikit-learn`
- `imbalanced-learn`
- `statsmodels`
- `networkx`
- `psutil`
- `joblib`
- `h5py`
- `seaborn`
- `cachecache`
- `upsetplot`
- `pyarrow`
- `ipython`
- `cmcrameri`
- `pillow`

GPU-backed filtering/whitening in `npyx` also needs CuPy. For NVIDIA systems with a CUDA 12 driver/runtime, install:
- `conda install -c conda-forge cupy cuda-version=12`

Or, in an existing environment:
- `python -m pip install cupy-cuda12x`

## Install

### 1) Create/activate environment

Example with conda:

```powershell
conda create -n neuropygui python=3.10 -y
conda activate neuropygui
```

Or from the bundled environment file:

```powershell
conda env create -f environment.yml
conda activate neuropygui
```

### 2) Install GUI requirements

From the project root (this folder):

```powershell
pip install -r requirements.txt
```

If you are installing into an older Python 3.9 environment such as `ks4_ece`, keep in mind that `pyqtgraph 0.14` requires Python 3.10+. The bundled `environment.yml` already targets Python 3.10 and is the recommended path for rebuilding the app or packaging a new `.exe`.

### 3) Install optional/feature dependencies

If you want full preprocessing/curation stack:

```powershell
pip install phy
```

`ecephys_spike_sorting` and `py_bombcell` are bundled inside this app folder.
No external source checkout is required.

External tool dependencies that must still be installed separately:
- `CatGT`
- `TPrime`
- `C_Waves-win`
- `Kilosort4` (Python package/runtime and compatible environment)

Configure these paths in the Preprocessing tab:
- `CatGT executable dir`
- `TPrime executable dir`
- `C_Waves executable dir`
- `KS4 repository dir`
- `Kilosort temp dir`

## Run

From the project root (this folder):

```powershell
python main.py
```

If you have multiple Python installations on PATH, prefer:

```powershell
conda run -n ks4_ece python main.py
```

## Notes

- Preprocessing expects SpikeGLX-style AP files (`*.imecX.ap.bin`).
- For curation previews, `metrics.csv` is loaded from the selected Kilosort folder.
- For post-processing quality columns and good-unit filtering, labels are read from:
  - `bombcell_labels.csv` (preferred)
  - `cluster_group.tsv` (fallback)



