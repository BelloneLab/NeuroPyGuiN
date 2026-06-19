<div align="center">

<img src="./neuropyguin/assets/big.jpg" alt="NeuroPyGuiN" width="720">

# NeuroPyGuiN

### Your Neuropixels pipeline, minus the terminal gymnastics.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![GUI](https://img.shields.io/badge/GUI-PySide6-41cd52)
![Spike%20sorting](https://img.shields.io/badge/Spike%20sorting-Kilosort4-orange)
![Curation](https://img.shields.io/badge/Curation-Bombcell%20%2B%20phy-8a2be2)

</div>

---

Neuropixels data is glorious. It is also a small mountain of binary files, config
flags, command-line tools, and "wait, which script do I run again?". NeuroPyGuiN
puts the whole journey behind a clean, clickable desktop app: drop in your
recordings, press buttons, watch progress bars, and end up with sorted, curated,
quality-checked, plotted units. No raw flag-typing required.

It is one window with three friendly tabs that follow the natural flow of a
project: **Preprocess and sort -> Curate -> Explore.**

## Take the tour

### 1. Preprocessing: build a queue, press go

Drag your SpikeGLX `.bin` files in (or scan a folder), pick your steps, and let
the queue run CatGT, Kilosort4, quality metrics, and friends while you get a
coffee. Recorded the same neurons across several sessions? Select them, hit
**Concatenate selected**, and sort them together so units keep the same identity
across days.

![Preprocessing tab](./neuropyguin/assets/screenshots/01_preprocessing.png)

### 2. Curation: judge your units, fast

Launch phy for manual curation, or let **Bombcell** do the heavy lifting:
tune thresholds with a live preview of how many units pass, eyeball the metric
histograms, and label good / noise / MUA in a couple of clicks. Sorted a
concatenated recording? One button splits the result back into per-session
spike trains, with each session's events attached.

![Curation tab](./neuropyguin/assets/screenshots/02_curation.png)

### 3. Post Processing: see your neurons do their thing

Load a curated dataset and the plots build themselves: rasters, firing rates,
mean waveforms, autocorrelograms, ISI histograms, condition PSTHs, and network
views. Filter to good units only, then export everything to a tidy HDF5 file.

![Post Processing tab](./neuropyguin/assets/screenshots/03_postprocessing.png)

### Prefer the dark side?

A single toggle flips the whole app (and every plot) between light and dark.

![Dark theme](./neuropyguin/assets/screenshots/04_curation_dark.png)

## Why you might like it

- **Zero terminal drama.** The whole pipeline is point-and-click, with live logs and progress so you always know what is happening.
- **Sort across sessions.** Fuse multiple recordings into one, sort them jointly, then split the spikes back per session. Same neuron, same ID, every day.
- **Curation that respects your time.** Bombcell metrics and labels, live threshold previews, and a one-click jump into phy.
- **Plots on tap.** Post-processing figures refresh automatically as you change settings. No "compute" button hunting.
- **Take it with you.** Export selected units (all, or good-only) to nested HDF5.
- **Looks good doing it.** Light and dark themes across every view.

## Get started in 3 steps

```powershell
# 1) Make an environment (Python 3.10+)
conda create -n neuropygui python=3.10 -y
conda activate neuropygui

# 2) Install the app
pip install -r requirements.txt
pip install phy            # optional, for manual curation

# 3) Launch it
python main.py
```

That is it. `ecephys_spike_sorting` and `py_bombcell` are bundled inside the app
folder, so there is nothing else to clone.

> On NVIDIA machines, GPU filtering/whitening uses CuPy. With a CUDA 12 driver:
> `conda install -c conda-forge cupy cuda-version=12` (or `pip install cupy-cuda12x`).

### Bring your own tools

A few external programs do the actual sorting heavy-lifting and are installed
separately, then pointed to from the Preprocessing tab:
`CatGT`, `TPrime`, `C_Waves`, and `Kilosort4`.

<details>
<summary>Full dependency list</summary>

Core Python packages live in [`requirements.txt`](./requirements.txt): `PySide6`,
`pyqtgraph`, `numpy`, `pandas`, `scipy`, `matplotlib`, `tqdm`, `numba`,
`scikit-learn`, `imbalanced-learn`, `statsmodels`, `networkx`, `psutil`,
`joblib`, `h5py`, `seaborn`, `cachecache`, `upsetplot`, `pyarrow`, `ipython`,
`cmcrameri`, `pillow`. The bundled `environment.yml` targets Python 3.10 and is
the recommended path for rebuilding or packaging the app.

</details>

## Good to know

- Preprocessing expects SpikeGLX-style AP files (`*.imecX.ap.bin`).
- Quality labels are read from `bombcell_labels.csv` first, then `cluster_group.tsv`.
- Settings, recents, and window layout are remembered between sessions.

## Standing on the shoulders of giants

NeuroPyGuiN is a friendly front-end. The real science is done by the tools
below. If you use this app in a publication, please cite the ones you used.

### ecephys_spike_sorting (Allen Institute)

The spike-sorting pipeline (CatGT, Kilosort, TPrime, quality metrics, etc.) is
built on `ecephys_spike_sorting`, developed by the Allen Institute for Brain
Science for the Allen Brain Observatory. Per the
[Allen Institute citation policy](https://alleninstitute.org/legal/citation-policy),
cite both the software and its primary publication:

- Allen Institute for Brain Science (2019). *ecephys_spike_sorting* [software]. Available from https://github.com/AllenInstitute/ecephys_spike_sorting
- Siegle, J. H., Jia, X., Durand, S., et al. (2021). Survey of spiking in the mouse visual system reveals functional hierarchy. *Nature*, 592, 86-92. https://doi.org/10.1038/s41586-020-03171-x

> © 2019 Allen Institute for Brain Science. Used under the Allen Institute Terms of Use.

The version bundled here is the SpikeGLX/CatGT/TPrime/Kilosort4 fork maintained
by Jennifer Colonell (https://github.com/jenniferColonell/ecephys_spike_sorting),
which adapts the Allen Institute pipeline; please acknowledge it as well.

### NeuroPyxels / npyx (M. Beau et al.)

Loading, processing, and plotting of Neuropixels data uses NeuroPyxels:

- Beau, M., D'Agostino, F., Lajko, A., Martínez, G., Häusser, M., & Kostadinov, D. (2021). *NeuroPyxels: loading, processing and plotting Neuropixels data in Python.* Zenodo. https://doi.org/10.5281/zenodo.5509733

Repository: https://github.com/m-beau/NeuroPyxels

### Bombcell (J. Fabre et al.)

Automated quality metrics and unit classification use Bombcell:

- Fabre, J. M. J., van Beest, E. H., Peters, A. J., Carandini, M., & Harris, K. D. (2023). *Bombcell: automated curation and cell classification of spike-sorted electrophysiology data.* Zenodo. https://doi.org/10.5281/zenodo.8172821

Repository: https://github.com/Julie-Fabre/bombcell

---

<div align="center">

Made with care in the Bellone Lab for the Neuropixels community.

</div>
