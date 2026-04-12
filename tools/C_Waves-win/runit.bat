
:: You can call C_Waves three ways:
::
:: 1) > C_Waves cmd-line-parameters
:: 2) > runit.bat cmd-line-parameters
:: 3a) Edit parameters in runit.bat, then call it ...
:: 3b) > runit.bat
::
:: This script effectively says:
:: "If there are no parameters sent to runit.bat, call C_Waves
:: with the parameters hard coded here, else, pass all of the
:: parameters through to C_Waves."
::

@echo off
@setlocal enableextensions
@cd /d "%~dp0"

set LOCALARGS=-spikeglx_bin=\\dm11\apig\C_waves_test_data\SC024_092319_NP1.0_Midbrain_g0_tcat.imec0.ap.bin ^
-clus_table_npy=\\dm11\apig\C_waves_test_data\clus_Table.npy ^
-clus_time_npy=\\dm11\apig\C_waves_test_data\spike_times.npy ^
-clus_lbl_npy=\\dm11\apig\C_waves_test_data\spike_clusters.npy ^
-dest=\\dm11\apig\C_waves_test_data\out ^
-samples_per_spike=82 -pre_samples=20 -num_spikes=1000 -snr_radius=8

if [%1]==[] (set ARGS=%LOCALARGS%) else (set ARGS=%*)

%~dp0C_Waves %ARGS%

