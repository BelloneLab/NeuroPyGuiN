"""Isolated, cancellable population worker: python -m neuropyguin.population_runner."""
from __future__ import annotations

import json
from pathlib import Path
import sys


def main():
    """Read spike arrays without optional GUI/HDF5 imports, then analyze safely."""
    import numpy as np
    from .population_analysis import PopulationConfig, analyze_population, save_result
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))

    class SpikeSource:
        """Minimal read-only dataset interface shared with the main app engine."""
        def __init__(self):
            self.ks_folder = Path(request["dataset"])
            self.sample_rate = float(request["sample_rate"])
            self.source_metadata_files = request.get("metadata_files", [])
            if not np.isfinite(self.sample_rate) or self.sample_rate <= 0:
                raise ValueError("Sampling rate must be positive and finite.")
            self.spike_times = np.load(self.ks_folder / "spike_times.npy", mmap_mode="r", allow_pickle=False).reshape(-1)
            self.spike_clusters = np.load(self.ks_folder / "spike_clusters.npy", mmap_mode="r", allow_pickle=False).reshape(-1)
            if self.spike_times.shape != self.spike_clusters.shape:
                raise ValueError("Spike time and cluster arrays have different lengths.")
            self.units = np.unique(self.spike_clusters)

        def unit_spike_times_s(self, unit):
            """Return one unit in seconds; indexing leaves the source mmap untouched."""
            return self.spike_times[self.spike_clusters == unit].astype(float) / self.sample_rate

    print("Binning selected units and fitting the population representation...", flush=True)
    result = analyze_population(SpikeSource(), request["units"], PopulationConfig(**request["config"]))
    save_result(result, sys.argv[2])
    print("Population analysis complete.", flush=True)


if __name__ == "__main__":
    main()
