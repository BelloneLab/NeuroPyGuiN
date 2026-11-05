# Learning track: Neuropixels in freely moving mice, with Python

A step-by-step course built on your own data. Each lesson has a short reading
part (this file) and a script you run and modify yourself in Spyder.

| Lesson | Topic | Script | Status |
| --- | --- | --- | --- |
| 1 | Structure of an analysis, inventory of the dataset | `lesson01_inventory.py` | ready |
| 2 | Writing `load_session()`: one object, one clock | written together | next |
| 3 | Synchronisation: imec, NIDQ and camera clocks | | planned |
| 4 | Tracking cleanup: likelihood, jumps, speed, smoothing | | planned |
| 5 | Spikes vs position: occupancy and rate maps | | planned |
| 6 | Spikes vs behaviour: bout-aligned rates (PETH), statistics | | planned |

---

## Lesson 1: how to structure the analysis

### 1.1 The pipeline, from photons and volts to figures

```text
RAW (never modified)          PROCESSED (reproducible)         ANALYSIS (your code)
------------------------      ---------------------------      -------------------------
SpikeGLX .bin/.meta   --CatGT--> filtered .bin --Kilosort4--> spike_times, clusters
                                                --Phy--------> curated labels
NIDQ sync / TTL edges --TPrime--> event times on the probe clock
Video .mp4            --DLC/SLEAP--> x, y, likelihood per frame
Video / scoring       --(BORIS, keypoint-MoSeq, manual)--> behaviour bouts
                                          |
                                          v
                              load_session()  ->  ONE session object,
                                                  ALL times in seconds,
                                                  ON ONE master clock
                                          |
                                          v
                     tracking cleanup -> rate maps / PETHs / decoding -> stats -> figures
```

Three rules that this course enforces everywhere:

1. **Raw data is read-only.** Every output goes to a separate output folder.
   If a step is wrong you rerun it; you never "fix" a source file by hand.
2. **Everything is in seconds on one clock.** Spikes come in samples of the
   probe clock (~30 kHz). Positions come in video frames. Behaviour often
   comes in frames too. Converting all of them to seconds on the *same* clock
   is the single most important step, and the most common source of silent
   errors (see 1.3).
3. **Loading is separate from analysis.** One function, `load_session()`,
   reads files and checks them. Analysis code only ever receives the clean
   object. If the file format changes, you change one function.

### 1.2 Only three kinds of data exist

Every signal in your experiment fits one of three shapes. Learn to recognise
them, and most analyses become "combine shape A with shape B".

| Kind | Example | Python representation |
| --- | --- | --- |
| **Events** (points in time) | spikes, TTL pulses, rewards | 1D array of times in s (+ a label array, e.g. cluster id) |
| **Time series** (value sampled at times) | x, y, speed, LFP | array of timestamps + array of values (same length) |
| **Intervals** (start, stop) | grooming bout, social contact, sleep | table with `start_s`, `stop_s`, `label` |

Example combinations: spikes (events) during grooming (intervals) gives a
firing rate per behaviour; spikes (events) against position (time series)
gives a place-field map.

### 1.3 The freely moving trap: three clocks

The probe, the NIDQ board and the camera each have their own quartz. They
drift by tens of ppm, i.e. tens of milliseconds per hour, and cameras drop
frames. Therefore:

- `frame_time = frame_index / fps` is **wrong** for anything beyond a rough
  look. One dropped frame shifts every later position.
- The right approach: the camera emits a TTL per frame (or the acquisition
  system records a shared sync pulse). Those edges are recorded by SpikeGLX,
  and TPrime maps them onto the probe clock. Each frame then gets a measured
  timestamp. Lesson 3 is entirely about this; the inventory script asks you
  whether your dataset has such timestamps.

### 1.4 Code layout we will grow into

```text
tutorials/learning/
  lesson01_inventory.py   # look at the data (read-only)
  lesson02_load.py        # load_session() + checks          (lesson 2)
  ...
  myanalysis/             # later: reusable functions moved here
    io.py                 # loaders only
    sync.py               # clock conversion only
    tracking.py           # cleanup, speed, occupancy
    analysis.py           # rate maps, PETHs
    plotting.py           # figure functions only
```

Each script starts with a `USER SETTINGS` block (paths, parameters). Nothing
below it should contain a hard-coded path or magic number.

### 1.5 Alternatives you should know about (and why we start without them)

- **pynapple**: implements exactly the events / time series / intervals
  vocabulary above (`Ts`, `Tsd`, `IntervalSet`) with restrict/align
  operations. Excellent; we will likely adopt it in lesson 5. Starting with
  plain NumPy first means you understand what it does for you.
- **NWB (pynwb)**: a standard file format holding spikes, tracking and
  behaviour in one file. Great for sharing and archiving, heavy for learning.
- **SpikeInterface**: for the sorting side (preprocessing, Kilosort,
  quality metrics). Your NeuroPyGuiN pipeline already covers that part.

### 1.6 Spyder setup (once)

1. Spyder must use the environment that has your packages. In
   `Tools > Preferences > Python interpreter`, choose
   `C:\Users\bellone\anaconda3\envs\neuropyguin\python.exe`.
2. That environment does not yet contain `spyder-kernels`, which Spyder
   needs. Install it once from an Anaconda Prompt (match the version Spyder
   asks for in its error message if it complains):
   `conda install -n neuropyguin -c conda-forge spyder-kernels`
3. Optional, only if your positions are DeepLabCut `.h5` files that you want
   to read with pandas: `conda install -n neuropyguin -c conda-forge pytables`.
   (The inventory script uses `h5py`, which is already installed.)
4. Useful keys: `Ctrl+Enter` runs the current `# %%` cell, `F5` runs the
   file, `F9` runs the selected lines. Use the Variable Explorer constantly.

### 1.7 Your exercise

1. Set `DATASET_PATH` in `lesson01_inventory.py` and run it.
2. Answer the five questions at the bottom of the script.
3. Send me the printed report (or the saved CSV) and your answers. Lesson 2
   is writing `load_session()` together for exactly your files.
