import ast
import numpy as np
import os
import time
import shutil

from argschema import ArgSchemaParser

from pykilosort import run, Bunch
from pykilosort.params import KilosortParams

from ...common.SGLXMetaToCoords import readMeta, MetaToCoords

from pathlib import Path


def _string_as_list_param(dict, param, default_val):
    v = dict.get(param, default_val)
    if v is None:
        return None
    else:
        return ast.literal_eval(v if ',' in v else v.replace(' ', ',', 1))


def _get_ks_params(meta_file, params_dict):
    """
    Create kilosort parameters from the probe metadata and
    from the input JSON
    """
    probe_meta = readMeta(meta_file)
    params = KilosortParams()

    params.preprocessing_function = params_dict.get('preprocessing_function', False)
    params.seed = params_dict.get('seed', 42)
    params.ks2_mode = params_dict.get('ks2_mode', False)
    params.perform_drift_registration = params_dict.get('perform_drift_registration',
                                                        True)
    params.do_whitening = True
    params.car = params_dict.get('car', False)
    params.fs = probe_meta.get('imSampRate', 30000)  # sample rate
    params.n_channels = int(probe_meta.get('nSavedChans'))
    params.save_temp_files = params_dict.get('save_temp_files', False)
    if params_dict.get('doFilter'):
        params.fshigh = params_dict.get('fshigh')
        params.fslow = params_dict.get('fslow')
    else:
        # skip filtering
        params.fshigh = None
        params.fslow = None
    params.minfr_goodchannels = params_dict.get('minfr_goodchannels', 0.1)
    params.genericSpkTh = params_dict.get('ThPre', 8.0)
    params.nblocks = params_dict.get('nblocks', 5)
    params.overwrite = True if params_dict.get('copy_fproc') else False
    params.sig_datashift = params_dict.get('sig_datashift', 20.0)
    params.deterministic_mode = params_dict.get('deterministic_mode', True)
    params.datashift = params_dict.get('datashift')
    params.Th = _string_as_list_param(params_dict, 'Th', '[10, 4]')
    params.ThPre = params_dict.get('ThPre', 8)
    params.lam = params_dict.get('lam', 10)
    params.AUCsplit = params_dict.get('AUCsplit', 0.9)
    params.minFR = params_dict.get('minFR', 0.02)
    params.momentum = _string_as_list_param(params_dict,
                                            'momentum', '[20,400]')
    params.output_filename = params_dict.get('fproc')
    params.sigmaMask = params_dict.get('sigmaMask', 30)
    params.whiteningRange = params_dict.get('whiteningRange', 32)

    return dict(params)


def _fix_phy_params(output_dir, dat_path, dat_name, chan_phy_binary,
                    sample_rate):
    """
    Writes a new params.py file.
    dat_path will be set to a relative path from output_dir to
    dat_path/dat_name
    sample rate will be written out to sufficient digits to be used
    """
    shutil.copy(os.path.join(output_dir, 'params.py'),
                os.path.join(output_dir, 'old_params.py'))

    relPath = os.path.relpath(dat_path, output_dir)
    new_path = os.path.join(relPath, dat_name)
    new_path = new_path.replace('\\', '/')

    paramLines = list()

    with open(os.path.join(output_dir, 'old_params.py'), 'r') as f:
        currLine = f.readline()

        while currLine != '':  # The EOF char is an empty string
            if 'dat_path' in currLine:
                currLine = "dat_path = '" + new_path + "'\n"
            elif 'n_channels_dat' in currLine:
                currLine = "n_channels_dat = " + repr(chan_phy_binary) + "\n"
            elif 'sample_rate' in currLine:
                currLine = (f'sample_rate = {sample_rate:.12f}\n')
            paramLines.append(currLine)
            currLine = f.readline()

    with open(os.path.join(output_dir, 'params.py'), 'w') as fout:
        for line in paramLines:
            fout.write(line)


def run_kilosort(args):

    print('ecephys spike sorting: pyks2 helper module')


    input_file_name = args['ephys_params']['ap_band_file']
    input_file = Path(input_file_name)

    ks_output_dir_name = args['directories']['kilosort_output_directory']
    ks_output_dir = Path(ks_output_dir_name)
    ks_output_dir.mkdir(parents=True, exist_ok=True)

    ks_tmp_output_dir_name = args['directories']['kilosort_output_tmp']
    ks_tmp_output_dir = Path(ks_tmp_output_dir_name)
    ks_tmp_output_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    meta_file = input_file.with_suffix('.meta')
    meta_name = meta_file.stem
    chanmap_filename = meta_name + '_chanMap.mat'
    chanmap_file = os.path.join(ks_output_dir, chanmap_filename)
    # generate chanMap file
    MetaToCoords(metaFullPath=meta_file, outType=1,
                 destFullPath=str(chanmap_file))

    # make a copy of the chanMap file to the binary directory; pyks looks inside
    # output directory (matching nextflow pipeline); modules in this pipeline
    # in binary directory.
    shutil.copy(chanmap_file, os.path.join(ks_output_dir.parent, chanmap_filename))
    
    pyks_params = args['pykilosort_helper_params']
    ks_params = _get_ks_params(meta_file, pyks_params)
    print(repr(ks_params))
    run(input_file,
        output_dir=ks_output_dir,
        dir_path=ks_tmp_output_dir,
        probe_path=str(chanmap_file),
        **ks_params)

    if pyks_params.get('copy_fproc'):
        fproc_path_str = pyks_params['fproc']
        fproc_path = ks_tmp_output_dir / '.kilosort' / Path(input_file).stem/'proc.dat'
        # make a new name for the processed file based on the original
        # binary and metadata files
        fp_save_name = meta_name + '_ksproc.bin'
        shutil.copy(fproc_path, os.path.join(ks_output_dir.parent, fp_save_name))
        cm_path = os.path.join(ks_output_dir, 'channel_map.npy')
        cm = np.load(cm_path)
        chan_phy_binary = cm.size
        _fix_phy_params(ks_output_dir, ks_output_dir.parent, fp_save_name,
                        chan_phy_binary, args['ephys_params']['sample_rate'])
    else:
        chan_phy_binary = args['ephys_params']['num_channels']
        _fix_phy_params(ks_output_dir, input_file.parent, input_file.name,
                       chan_phy_binary, args['ephys_params']['sample_rate'])

    if ks_tmp_output_dir != ks_output_dir:
        try:
            print('Remove KS temporary dir %s' % ks_tmp_output_dir)
            shutil.rmtree(ks_tmp_output_dir)
        except OSError as e:
            print('Error: %s : %s' % (ks_tmp_output_dir, e.strerror))

    execution_time = time.time() - start

    print('kilsort run time: ' + str(np.around(execution_time, 2)) + ' seconds')
    print()

    return {
        'execution_time': execution_time,
    }  # output manifest


def main():

    from ._schemas import InputParameters, OutputParameters

    """Main entry point:"""
    mod = ArgSchemaParser(schema_type=InputParameters,
                          output_schema_type=OutputParameters)

    output = run_kilosort(mod.args)

    output.update({"input_parameters": mod.args})
    if "output_json" in mod.args:
        mod.output(output, indent=2)
    else:
        print(mod.get_output_json(output))


if __name__ == "__main__":
    main()



