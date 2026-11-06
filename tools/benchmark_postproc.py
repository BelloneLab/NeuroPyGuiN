"""Compare the audited engine with a Git baseline using deterministic synthetic spikes.

Run from the repository: python tools/benchmark_postproc.py
All configuration is below. Results are descriptive timings, not biological data
or inferential tests. PDF, SVG, PNG and CSV outputs share the same measurements.
"""
from __future__ import annotations
import csv
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

import numpy as np

BASELINE_REVISION = '01f7752'
SEED = 173
REPEATS = 3
SPIKES = 12_000
EVENTS = 400
DURATION_S = 600.
OUTPUT_DIR = Path(__file__).resolve().parents[1] / 'tools' / '_postproc_shots' / 'audit_benchmark'
FIGURE_SIZE = (6.8, 2.8)
COLORS = ('#94a3b8', '#087f8c')


def load_baseline(root: Path):
    """Load committed source without touching the working tree or raw data."""
    source = subprocess.check_output(['git', 'show', f'{BASELINE_REVISION}:neuropyguin/postproc_engine.py'], cwd=root, text=True)
    module = types.ModuleType('benchmark_baseline')
    sys.modules[module.__name__] = module
    exec(compile(source, 'baseline_postproc_engine.py', 'exec'), module.__dict__)
    return module.NeuropixelsDataset


def main() -> int:
    """Verify numerical equivalence before reporting paired timing distributions."""
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from neuropyguin.postproc_engine import NeuropixelsDataset
    baseline = load_baseline(root)
    rng = np.random.default_rng(SEED)
    times = np.sort(rng.uniform(0, DURATION_S * 1000, SPIKES))
    labels = rng.integers(1, 3, SPIKES)
    events = np.linspace(2, DURATION_S - 2, EVENTS)
    rows, checks = [], {}
    with tempfile.TemporaryDirectory() as folder:
        for name, constructor in [('Before', baseline), ('After', NeuropixelsDataset)]:
            for analysis in ('CCG', 'PSTH', 'Unit browsing'):
                for repeat in range(REPEATS):
                    ds = constructor(Path(folder), 1000., 1, 1., None, times, labels, np.array([1, 2]), None, None, None, None)
                    # Compare computation, excluding optional disk serialization.
                    ds._cache_get = lambda *a: None
                    ds._cache_set = lambda *a: None
                    start = time.perf_counter()
                    if analysis == 'CCG':
                        result = ds.correlogram(1, 2, 1, 50, normalize='Counts')[1]
                    elif analysis == 'PSTH':
                        result = ds.psth_trials(1, events, .5, .5, 10)[1]
                    else:
                        for _ in range(100):
                            result = ds.unit_spike_times_s(1)
                    elapsed = time.perf_counter() - start
                    rows.append(dict(analysis=analysis, version=name, repeat=repeat, seconds=elapsed, seed=SEED, spikes=SPIKES))
                    if name == 'Before':
                        checks[analysis] = result.copy()
                    else:
                        np.testing.assert_allclose(result, checks[analysis], rtol=1e-12, atol=1e-12)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / 'timings.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=FIGURE_SIZE, constrained_layout=True)
    for ax, analysis in zip(axes, ('CCG', 'PSTH', 'Unit browsing')):
        values = [np.array([r['seconds'] * 1000 for r in rows if r['analysis'] == analysis and r['version'] == version]) for version in ('Before', 'After')]
        for x, (array, color) in enumerate(zip(values, COLORS)):
            ax.bar(x, np.median(array), width=.42, color=color)
            ax.scatter(np.full(array.size, x), array, s=8, c='#243447', zorder=3)
        ratio = np.median(values[0]) / np.median(values[1])
        ax.set_title(f'{analysis}\n{ratio:.1f}× faster', fontsize=10)
        ax.set_xticks([0, 1], ['Before', 'After'], fontsize=8)
        ax.set_ylabel('Time (ms)', fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)
        print(f'{analysis}: before={np.median(values[0]):.3f}ms after={np.median(values[1]):.3f}ms ratio={ratio:.2f}x')
    fig.suptitle('Synthetic benchmark · median and individual runs (n=3)', fontsize=10)
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(OUTPUT_DIR / f'benchmark.{extension}', dpi=300)
    plt.close(fig)
    print(f'Numerically equivalent; artifacts: {OUTPUT_DIR}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
