
Purpose:
+ Mean waveforms calculator.
Run messages are appended to C_Waves.log in the current working directory.

Install:
1) Copy C_Waves-linux to your machine, cd into folder.
2) If needed, > chmod +x install.sh
3) > ./install.sh
4) Read notes in runit.sh wrapper script (required).

Compatibility:
- Included libraries are from Ubuntu 16.04 (Xenial).
- Tested with Ubuntu 20.04 and 20.10.
- Tested with Scientific Linux 7.3.
- Tested with Oracle Linux Server 8.3.
- Let me know if it runs on other distributions.

Output:
+ 'mean_waveforms.npy': Array of mean waveforms (uV) with major-to-minor
+    dims {nClusters, nChannels, nSamples}. nChannels matches the channel
+    count of the spikeglx bin file. Waveforms for non-neural channels are zeroed.
+ 'median_peak_waveforms.npy': Median waveform (uV) at peak with major-to-minor
+    dims {nClusters, nSamples}. Waveforms for empty clusters are zeroed.
+ 'cluster_snr.npy': 2-col table of snr for the peak channels with cols
+    {snr, nSpikes_in_snr}. SNR for an empty cluster is set to -1.

Usage:
>runit.sh < required_parameters > [ options ]

# Enclosing whole parameter list in quotes is recommended, like this:
# >runit.sh '< required_parameters > [ options ]'

Required parameters:
-spikeglx_bin=<filepath>    ;spikeglx metafile required in same dir
-clus_table_npy=<filepath>  ;2-col table (uint32): {num_spike, pk-chan}
-clus_time_npy=<filepath>   ;1-col table (64-bit): {spike_time} (ASCENDING ORDER)
-clus_lbl_npy=<filepath>    ;1-col table (uint32): {clus_lbl per spike_time}
-dest=<path>                ;output dir (must exist)
-samples_per_spike=82       ;waveform timepoints
-pre_samples=30             ;subset of samples_per_spike to left of peak
-num_spikes=1000            ;max waveforms included in averages
-snr_radius=8               ;disk radius (rows/columns) about pk-chan, or zero

Options:
-snr_radius_um=140          ;disk radius (microns) about pk-chan, or zero
-startsecs=10.0             ;only include spikes >= (seconds)
-endsecs=100.0              ;only include spikes <  (seconds)
-chnexcl=0,3:5              ;exclude these acq chans from snr
-prefix=string              ;output files are named:
                            ;  'prefix_mean_waveforms.npy'
                            ;  'prefix_cluster_snr.npy'
-debug_npy                  ;log 1st 10 {table, time, lbl} entries then quit

Notes:
- clus_time_npy spike times must be sorted into ascending order before calling C_Waves.
- clus_lbl_npy and clus_time_npy must be in 1-1 correspondence.
- clus_lbl_npy entries are zero-based row indices into clus_table_npy, hence...
- clus_lbl_npy values must be in range [0,N-1], where N = clus_table_npy row count.

- SNR is calculated on a disk of given radius (see below) about pk-chan. If you set -snr_radius=0 (or -snr_radius_um=0) only the pk-chan is used for signal and noise estimation.
- SNR signal is defined as peak-to-peak maximum on the average waveform, where highest and lowest voltages may appear anywhere in footprint.
- SNR noise is calculated as standard deviation of residuals of member waveforms on first 15 samples (left tail). This sampling strives to be insensitive to cluster contamination. For this to work as intended, -pre_samples should be at least 20.
- SNR disk radius: Use option -snr_radius_um to set the radius in microns; this requires metadata item ~snsGeomMap. If -snr_radius_um or ~snsGeomMap are absent, we instead use -snr_radius (default 8) which specifies the radius as a count of rows/columns in the required ~snsShankMap metadata item.
- SNR Note: As of version 20230202, SpikeGLX (*.meta) files record ~snsGeomMap (but NOT ~snsShankMap).


Change Log
----------
Version 2.9
- Support NP1014, NP1033, NP2005, NP2006.

Version 2.8
- Support NP2021 quad-probes.
- Script default radius changed to 140um.
- Add -startsecs and -endsecs options.

Version 2.7
- Support NP1221 probes.
- Support NXT probes.
- Parse older three-value geomMap headers.
- Add median_peak_waveforms.npy.

Version 2.6
- Support NP2020 quad-probes.
- Handle int64 or uint64 spike times.

Version 2.5
- Handle phy output.

Version 2.4
- Sanity-check npy files.
- Update probe support.

Version 2.3
- Support probes {2003,2004,2013,2014}.

Version 2.2
- Add option -snr_radius_um, to specify SNR disk in microns.

Version 2.1
- Support latest probes.

Version 2.0
- Fix SNR channel exclusion.

Version 1.9
- Support UHD2, all current probes.

Version 1.8
- Improved calling scripts.

Version 1.7
- Working/calling dir can be different from installed dir.
- Log file written to working dir.

Version 1.6
- Improved SNR estimator.

Version 1.5
- Smoother mean waveforms.

Version 1.4
- Support NP1010 probe.

Version 1.3
- -debug_npy option.
- Handle fortran-order npy files.
- Output npy headers are x64 size.

Version 1.2
- Uses 3A imro classes.
- Support for UHD-1 and NHP.

Version 1.1
- Remove 4GB binary file size limit.
- Faster.

Version 1.0
- Initial release.


