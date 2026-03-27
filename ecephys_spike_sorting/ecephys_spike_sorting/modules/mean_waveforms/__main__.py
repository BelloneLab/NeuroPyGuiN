from argschema import ArgSchemaParser
import ast
import os
import pathlib
import re
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from scipy.io import loadmat

from ...common.utils import load_kilosort_data
from ...common.utils import getSortResults
from ...common.utils import getFileVersion

from .extract_waveforms import extract_waveforms, writeDataAsNpy
from .waveform_metrics import calculate_waveform_metrics
from .metrics_from_file import metrics_from_file


def _parse_phy_dat_path(params_path):
    if not os.path.exists(params_path):
        return None
    with open(params_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    match = re.search(r"^dat_path\s*=\s*(.+)$", text, flags=re.MULTILINE)
    if not match:
        return None
    raw = match.group(1).strip()
    try:
        return ast.literal_eval(raw)
    except Exception:
        return raw.strip().strip("'").strip('"')


def _iter_phy_dat_candidates(output_dir, input_file):
    params_value = _parse_phy_dat_path(os.path.join(output_dir, 'params.py'))
    if isinstance(params_value, (list, tuple)):
        values = [str(item).strip() for item in params_value if str(item).strip()]
    elif params_value is not None and str(params_value).strip():
        values = [str(params_value).strip()]
    else:
        values = []

    basenames = []
    for value in values:
        path = pathlib.Path(value).expanduser()
        if not path.is_absolute():
            path = (pathlib.Path(output_dir) / path).resolve()
        if path.exists():
            yield path.resolve()
        if path.name and path.name not in basenames:
            basenames.append(path.name)

    input_path = pathlib.Path(input_file).expanduser()
    if input_path.exists():
        yield input_path.resolve()
    if input_path.name and input_path.name not in basenames:
        basenames.append(input_path.name)

    probe_dir = pathlib.Path(output_dir).parent
    for name in basenames:
        candidate = probe_dir / name
        if candidate.exists():
            yield candidate.resolve()

    sibling_bins = sorted(probe_dir.glob('*.ap.bin'))
    if len(sibling_bins) == 1:
        yield sibling_bins[0].resolve()
    for candidate in sibling_bins:
        yield candidate.resolve()


def _resolve_spikeglx_bin(args):
    output_dir = args['directories']['kilosort_output_directory']
    input_file = args['ephys_params']['ap_band_file']
    seen = set()
    for candidate in _iter_phy_dat_candidates(output_dir, input_file):
        key = str(candidate).lower()
        if key in seen:
            continue
        seen.add(key)
        return str(candidate)
    return input_file


def _chanmap_candidate(path):
    p = pathlib.Path(path)
    name = p.name[:-4] if p.name.lower().endswith('.bin') else p.name
    return p.parent / (name + '_chanMap.mat')


def _resolve_chanmap_mat(output_dir, resolved_bin, input_file):
    ks_dir = pathlib.Path(output_dir)
    candidates = [
        ks_dir / _chanmap_candidate(resolved_bin).name,
        ks_dir / _chanmap_candidate(input_file).name,
        _chanmap_candidate(resolved_bin),
        _chanmap_candidate(input_file),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    ks_chanmaps = sorted(ks_dir.glob('*_chanMap.mat'))
    if len(ks_chanmaps) == 1:
        return str(ks_chanmaps[0])
    probe_chanmaps = sorted(ks_dir.parent.glob('*_chanMap.mat'))
    if len(probe_chanmaps) == 1:
        return str(probe_chanmaps[0])

    searched = "\n".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        "Could not resolve chanMap.mat for mean_waveforms. Searched:\n" + searched
    )


def _stream_subprocess(cmd):
    proc_cmd = ["cmd.exe", "/c"] + cmd if sys.platform.startswith('win') else cmd
    proc = subprocess.Popen(
        proc_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            print('[C_Waves] ' + line)
    return proc.wait()


def calculate_mean_waveforms(args):

    print('ecephys spike sorting: mean waveforms module')
    
    start = time.time()
    clu_version = 0
    wm_fullpath = args['waveform_metrics']['waveform_metrics_file']
    
    if args['mean_waveform_params']['use_C_Waves']:
        
        print('Calculating mean waveforms using C_waves.')
        output_dir = args['directories']['kilosort_output_directory']
        spikeglx_bin = _resolve_spikeglx_bin(args)
        print('Resolved AP source for C_Waves: ' + spikeglx_bin)
        # regenerate the clus_Table in case there has been manual curation of the data in phy
        
        # get version number for new clus_table file
        clu_path_orig = os.path.join(output_dir, 'clus_Table.npy' )
        clus_table_npy, clu_version = getFileVersion(clu_path_orig)
        
        #version = 0 if no clu_Table exists, file = clus_Table.npy
        #version = 1 or higher, new clus_Table = clus_Table_version.npy
        
        getSortResults(output_dir, clu_version)
        
        # build paths to cluster and times tables, which are generated by
        # kilosort_helper module
        clus_time_npy = os.path.join(output_dir, 'spike_times.npy' )
        clus_lbl_npy = os.path.join(output_dir, 'spike_clusters.npy' )
        dest, wavefile = os.path.split(args['mean_waveform_params']['mean_waveforms_file'])
        
        # on first call to mean_waveforms, output has no version indicator;
        # for later calls, will rename to _version.npy
        # need to rename the originals, because the output names from C_Waves are hard coded

        if clu_version == 1:           
            # if mean_waveforms files exists, rename
            old_mwf = os.path.join(dest,'mean_waveforms.npy')
            if os.path.exists(old_mwf):
                new_mwf = os.path.join(dest,'mean_waveforms_0.npy')
                os.rename(old_mwf, new_mwf)
            old_snr = os.path.join(dest,'cluster_snr.npy')
            if os.path.exists(old_snr):
                new_snr = os.path.join(dest,'cluster_snr_0.npy')
                os.rename(old_snr, new_snr)

        
        # kilosort saves the spike_clusters files as uint32. 
        # when phy re-saves after curation, it saves as int32 (!)
        # to ensure the correct datatype for C_Waves, load the spike_clusters
        # and convert if necessary
        sc = np.load(clus_lbl_npy)
        if sc.dtype != 'uint32':
            sc = sc.astype('uint32')
            np.save(clus_lbl_npy,sc)
            
        
        
        # path to the 'runit.bat' executable that calls C_Waves.
        # Essential in linux where C_Waves executable is only callable through runit
        if sys.platform.startswith('win'):
            exe_path = os.path.join(args['mean_waveform_params']['cWaves_path'], 'runit.bat')
        elif sys.platform.startswith('linux'):
            exe_path = os.path.join(args['mean_waveform_params']['cWaves_path'], 'runit.sh')
        else:
            print('unknown system, cannot run C_Waves')
        
        cwaves_cmd = [
            exe_path,
            '-spikeglx_bin=' + spikeglx_bin,
            '-clus_table_npy=' + clus_table_npy,
            '-clus_time_npy=' + clus_time_npy,
            '-clus_lbl_npy=' + clus_lbl_npy,
            '-dest=' + dest,
            '-samples_per_spike=' + repr(args['mean_waveform_params']['samples_per_spike']),
            '-pre_samples=' + repr(args['mean_waveform_params']['pre_samples']),
            '-num_spikes=' + repr(args['mean_waveform_params']['spikes_per_epoch']),
            '-snr_radius=' + repr(args['mean_waveform_params']['snr_radius']),
            '-snr_radius_um=' + repr(args['mean_waveform_params']['snr_radius_um']),
        ]

        print('Launching C_Waves...')
        print(' '.join(cwaves_cmd))
        
        # make the C_Waves call
        rc = _stream_subprocess(cwaves_cmd)
        if rc != 0:
            raise RuntimeError('C_Waves exited with code ' + repr(rc))
        print('C_Waves finished, loading outputs...')
        
        # for first version, retain original names
        if clu_version == 0:
            mean_waveform_fullpath = os.path.join(dest, 'mean_waveforms.npy')
            snr_fullpath = os.path.join(dest, 'cluster_snr.npy')
        else:
            # build names with version number and rename
            # version 0 files are not renamed to maintain compatiblity with
            mean_waveform_fullpath = os.path.join(dest, 'mean_waveforms_' + repr(clu_version) + '.npy')
            snr_fullpath = os.path.join(dest, 'cluster_snr_' + repr(clu_version) + '.npy')
            os.rename(os.path.join(dest, 'mean_waveforms.npy'), mean_waveform_fullpath)
            os.rename(os.path.join(dest, 'cluster_snr.npy'), snr_fullpath)
        if not os.path.exists(mean_waveform_fullpath) or not os.path.exists(snr_fullpath):
            raise FileNotFoundError(
                'C_Waves did not produce expected outputs: '
                + mean_waveform_fullpath
                + ' and '
                + snr_fullpath
            )
            
        
        # C_Waves writes out files of the waveforms and snr
        # call version of calculate_waveform_metrics that will use these files
        
        print('Loading kilosort outputs and whitening matrix...')
        # load in kilosort output needed for these calculations
        spike_times, spike_clusters, spike_templates, amplitudes, templates, channel_map, \
        channel_pos, clusterIDs, cluster_quality, cluster_amplitude = \
                load_kilosort_data(args['directories']['kilosort_output_directory'], \
                    args['ephys_params']['sample_rate'], \
                    convert_to_seconds = False)
                
        # read in inverse of whitening matrix
        w_inv = np.load((os.path.join(args['directories']['kilosort_output_directory'], 'whitening_mat_inv.npy')))
        
        # the channel_pos loaded from the phy output omits any sites excluded
        # as noise by the kilosort_helper module, or excluded fow low spike rete
        # by kilosort itself. The waveform metrics are calculated on ALL sites
        # based on the mean waveforms calculated for each unit; therefore
        # we need the site locations for all sites.
        # load the channel map associated with this kilosort run; in kilosort_helper
        # a copy is made next to the data file
        input_file = args['ephys_params']['ap_band_file']
        chanMapMat = _resolve_chanmap_mat(output_dir, spikeglx_bin, input_file)
        print('Using chanMap: ' + chanMapMat)
        site_x = np.squeeze(loadmat(chanMapMat)['xcoords'])
        site_y = np.squeeze(loadmat(chanMapMat)['ycoords'])
        
                
        print('Computing waveform metrics...')
        metrics = metrics_from_file(mean_waveform_fullpath, snr_fullpath, clus_table_npy, \
                    spike_times, \
                    spike_clusters, \
                    templates, \
                    channel_map, \
                    args['ephys_params']['bit_volts'], \
                    args['ephys_params']['sample_rate'], \
                    args['ephys_params']['vertical_site_spacing'], \
                    w_inv, \
                    site_x, site_y, \
                    args['mean_waveform_params'])
        
        wm_fullpath = (args['waveform_metrics']['waveform_metrics_file'])

        if clu_version > 0:
           # save new metrics as _version number
           wm_fullpath = os.path.join(pathlib.Path(wm_fullpath).parent, pathlib.Path(wm_fullpath).stem + '_' + repr(clu_version) + '.csv')
    
        print('Saving waveform metrics...')
        metrics.to_csv(wm_fullpath, index=False)
            
        
    else:
        
        print('Calculating mean waveforms using python.')
        print("Loading data...")
    
        rawData = np.memmap(args['ephys_params']['ap_band_file'], dtype='int16', mode='r')
        data = np.reshape(rawData, (int(rawData.size/args['ephys_params']['num_channels']), args['ephys_params']['num_channels']))
    
        spike_times, spike_clusters, spike_templates, amplitudes, templates, channel_map, \
        channel_pos, clusterIDs, cluster_quality, cluster_amplitude = \
                load_kilosort_data(args['directories']['kilosort_output_directory'], \
                    args['ephys_params']['sample_rate'], \
                    convert_to_seconds = False)
    
        print("Calculating mean waveforms...")
    
        waveforms, spike_counts, coords, labels, metrics = extract_waveforms(data, spike_times, \
                    spike_clusters,
                    templates,
                    channel_map,
                    args['ephys_params']['bit_volts'], \
                    args['ephys_params']['sample_rate'], \
                    args['ephys_params']['vertical_site_spacing'], \
                    args['mean_waveform_params'])
    
        writeDataAsNpy(waveforms, args['mean_waveform_params']['mean_waveforms_file'])
        metrics.to_csv(args['waveform_metrics']['waveform_metrics_file'], index=False)


    # if the cluster metrics have already been run, merge the waveform metrics into that file
    # build file path with current version
    metrics_args = args['cluster_metrics']['cluster_metrics_file']
    metrics_curr = os.path.join(pathlib.Path(metrics_args).parent, pathlib.Path(metrics_args).stem + '_' + repr(clu_version) + '.csv')

    if os.path.exists(metrics_curr):
        qmetrics = pd.read_csv(metrics_curr)
        qmetrics = qmetrics.drop(qmetrics.columns[0], axis='columns')
        qmetrics = qmetrics.merge(pd.read_csv(wm_fullpath, index_col=0),
                     on='cluster_id',
                     suffixes=('_quality_metrics','_waveform_metrics'))  
        print("Saving merged quality metrics ...")
        qmetrics.to_csv(metrics_curr, index=False)
        
    execution_time = time.time() - start

    print('total time: ' + str(np.around(execution_time,2)) + ' seconds')
    print()
    
    return {"execution_time" : execution_time} # output manifest


def main():

    from ._schemas import InputParameters, OutputParameters

    mod = ArgSchemaParser(schema_type=InputParameters,
                          output_schema_type=OutputParameters)

    output = calculate_mean_waveforms(mod.args)

    output.update({"input_parameters": mod.args})
    if "output_json" in mod.args:
        mod.output(output, indent=2)
    else:
        print(mod.get_output_json(output))


if __name__ == "__main__":
    main()



