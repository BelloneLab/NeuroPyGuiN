import ast
import numpy as np
import os
import time
import shutil


from argschema import ArgSchemaParser

from kilosort import run_kilosort, io
from kilosort.parameters import DEFAULT_SETTINGS

from ...common.SGLXMetaToCoords import readMeta, MetaToCoords

from pathlib import Path

def _string_as_list_param(dict, param, default_val):
    v = dict.get(param, default_val)
    if v is None:
        return None
    else:
        return ast.literal_eval(v if ',' in v else v.replace(' ', ',', 1))


def _get_ks_params(meta_file, settings_from_json, b_seed):
    """
    Create kilosort parameters from the probe metadata and
    from the input JSON
    """
    probe_meta = readMeta(meta_file)

    # in run_kilosort, the settings dictionary is merged with the dictionary
    # DEFAULT_SETTINGS. Here, only set settings passed from the pipeline params
    # and read from metadata.
    settings = DEFAULT_SETTINGS
    settings['n_chan_bin'] = int(probe_meta.get('nSavedChans'))
    settings['fs'] = float(probe_meta.get('imSampRate')) # sample rate
    # all other user setting coming from the json
    settings = {**settings, **settings_from_json}
    if settings['tmax'] < 0:
        settings['tmax'] = np.inf
    if not(b_seed):
        # remove seed settings from ks4 settings
        settings.pop('template_seed')
        settings.pop('cluster_seed')

    return dict(settings)


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

    # create a relative path if possible, otherwise use the full path to the data
    try:
        relPath = os.path.relpath(dat_path, output_dir)
        new_path = os.path.join(relPath, dat_name)
    except ValueError:
        new_path = os.path.join(dat_path, dat_name)

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

    # also, resave spike_times.npy as uint64
    # st = np.load(os.path.join(output_dir,'spike_times.npy'))
    #st_neg = np.squeeze(np.where(st<0))
    #st[st_neg] = 0
    #np.save(os.path.join(output_dir,'spike_times.npy'), st.astype(np.uint64))



def run_ks4(args):

    """
    Run full ecephys spike sorting pipeline on specified data.

     Set up call to kilosort 4 call run_kilosort.run_kilosort

     Comments on the parameters and call from teh ks4 code:

     run_kilosort(settings, probe=None, probe_name=None, filename=None,
                  data_dir=None, file_object=None, results_dir=None,
                  data_dtype=None, do_CAR=True, invert_sign=False, device=None,
                  progress_bar=None, save_extra_vars=False):


     Parameters
     ----------
     settings : dict
         Specifies a number of configurable parameters used throughout the
         ecephys spike sorting pipeline. See `kilosort/parameters.py` for a full list of
         available parameters.
         NOTE: `n_chan_bin` must be specified here, but all other settings are
               optional.
     probe : dict; optional.
         A Kilosort4 probe dictionary, as returned by `kilosort.io.load_probe`.
     probe_name : str; optional.
         Filename of probe to use, within the default `PROBE_DIR`. Only include
         the filename without any preceeding directories. Will ony be used if
         `probe is None`. Alternatively, the full filepath to a probe stored in
         any directory can be specified with `settings = {'probe_path': ...}`.
         See `kilosort.utils` for default `PROBE_DIR` definition.
     filename: str or Path; optional.
         Full path to binary data file. If specified, will also set
         `data_dir = filename.parent`.
     data_dir : str or Path; optional.
         Specifies directory where binary data file is stored. Kilosort will
         attempt to find the binary file. This works best if there is exactly one
         file in the directory with a .bin, .bat, .dat, or .raw extension.
         Only used if `filename is None`.
         Also see `kilosort.io.find_binary`.
     file_object : array-like file object; optional.
         Must have 'shape' and 'dtype' attributes and support array-like
         indexing (e.g. [:100,:], [5, 7:10], etc). For example, a numpy
         array or memmap. Must specify a valid `filename` as well, even though
         data will not be directly loaded from that file.
     results_dir : str or Path; optional.
         Directory where results will be stored. By default, will be set to
         `data_dir / 'kilosort4'`.
     data_dtype : str or type; optional.
         dtype of data in binary file, like `'int32'` or `np.uint16`. By default,
         dtype is assumed to be `'int16'`.
     do_CAR : bool; default=True.
         If True, apply common average reference during preprocessing
         (recommended).
     invert_sign : bool; default=False.
         If True, flip positive/negative values in data to conform to standard
         expected by Kilosort4.
     device : torch.device; optional.
         CPU or GPU device to use for PyTorch calculations. By default, PyTorch
         will use the first detected GPU. If no GPUs are detected, CPU will be
         used. To set this manually, specify `device = torch.device(<device_name>)`.
         See PyTorch documentation for full description.
     progress_bar : tqdm.std.tqdm or QtWidgets.QProgressBar; optional.
         Used by sorting steps and GUI to track sorting progress. Users should
         not need to specify this.
     save_extra_vars : bool; default=False.
         If True, save tF and Wall to disk after sorting.
    """

    start = time.time()
    print('ecephys spike sorting: ks4 helper module')

    input_file_name = args['ephys_params']['ap_band_file']
    input_file = Path(input_file_name)

    ks_output_dir_name = args['directories']['kilosort_output_directory']
    ks_output_dir = Path(ks_output_dir_name)
    ks_output_dir.mkdir(parents=True, exist_ok=True)

    meta_file = input_file.with_suffix('.meta')
    meta_name = meta_file.stem
    chanmap_filename = meta_name + '_chanMap.mat'
    chanmap_file = os.path.join(ks_output_dir, chanmap_filename)
    # generate chanMap file
    #check if chanMap file already exists
    # if os.path.exists(chanmap_file):
    # import pdb
    # pdb.set_trace()
    MetaToCoords(metaFullPath=meta_file, outType=1,
                 destFullPath=str(chanmap_file))
    # pdb.set_trace()
    # make a copy of the chanMap file to the binary directory; serves as a record
    # simplfies re-running the sorter outside of the pipeline.
    shutil.copy(chanmap_file, os.path.join(input_file.parent, chanmap_filename))
    ks4_prb = io.load_probe(os.path.join(input_file.parent, chanmap_filename))

    settings_from_json = args['ks4_helper_params']['ks4_params']

    # set b_seed=False to work with standard versions of ks4
    settings = _get_ks_params(meta_file, settings_from_json, b_seed = False)
    print('helloooooo',repr(settings))
#    print(repr(ks4_prb))

    run_kilosort(settings,
                 probe=ks4_prb,
                 filename=input_file,
                 results_dir=ks_output_dir,
                 data_dtype='int16',
                 do_CAR=args['ks4_helper_params']['do_CAR'],
                 save_extra_vars=args['ks4_helper_params']['save_extra_vars'],
                 clear_cache=True,
                 save_preprocessed_copy=args['ks4_helper_params']['save_preprocessed_copy'],
                 verbose_console=True)
#
    # make sure the params file for phy has the correct number of channels
    # to match the input binary. This is likely not necessary for KS4.
    chan_phy_binary = args['ephys_params']['num_channels']
    _fix_phy_params(ks_output_dir, input_file.parent, input_file.name,
                       chan_phy_binary, args['ephys_params']['sample_rate'])

    if args['ks4_helper_params']['ks_make_copy']:
        # get the kilsort output directory name
        phyName = ks_output_dir.stem
        # build a name for the copy
        copy_dir = os.path.join(ks_output_dir.parent, phyName + '_orig')
        # check for whether the directory is already there; if so, delete it
        if os.path.exists(copy_dir):
            shutil.rmtree(copy_dir)
        # make a copy of output_dir
        shutil.copytree(ks_output_dir, copy_dir)

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

    output = run_ks4(mod.args)

    output.update({"input_parameters": mod.args})
    if "output_json" in mod.args:
        mod.output(output, indent=2)
    else:
        print(mod.get_output_json(output))


if __name__ == "__main__":
    main()



