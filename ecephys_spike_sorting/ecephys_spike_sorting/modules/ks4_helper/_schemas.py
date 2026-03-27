from argschema import ArgSchema, ArgSchemaParser 
from argschema.schemas import DefaultSchema
from argschema.fields import Nested, InputDir, String, Float, Dict, Int, Bool, NumpyArray
from ...common.schemas import EphysParams, Directories, CommonFiles

class ks4_params(DefaultSchema):    
    n_chan_bin = Int(required=False, default=385, help='total channels in the binary file, including non-ephys channels')
    batch_size = Int(required=False, default=60000, help='number of samples per processing batch')
    Th_universal = Float(required=False, default=9, help='threshold for creating templates')
    Th_learned = Float(required=False, default=8, help='threshold for creating templates')
    Th_single_ch = Float(required=False, default=8, help='threshold crossings for pre-clustering (in PCA projection space)')
    duplicate_spike_ms = Float(required=False, default=0.25, help='Number of bins for which subsequent spikes from the same cluster are assumed to be artifacts. A value of 0 disables this step.')
    nblocks = Int(required=False, default=5, help='number of blocks used to segment the probe when tracking drift, 0 == do not track, 1 == rigid, > 1 == non-rigid')
    sig_interp = Float(required=True, default=20.0, help='sigma for the Gaussian interpolation in drift correction um)')
    whitening_range = Int(required=False, default=32, help='number of channels to use for whitening each channel')
    min_template_size = Float(required=False, default=10, help='Width in um of Gaussian envelope for template weight')
    template_sizes = Int(required=False, default=5, help='number of template sizes, multiples of min_template size')
    templates_from_data=Bool(required=False, default=True, help='set to True to extract templates from data')
    tmin = Float(required=False, default=0, help='time in sec to start processing')
    tmax = Float(required=False, default=-1, help='time in sec to end processing; if < 0, set to inf ')
    nt = Int(required=False, default=61, help='time samples used to represent spike waveforms')
    dmin = Float(required=False, allow_none=True, default=None, help='vertical spacing for universal templates (um)')
    dminx = Float(required=False, default=32, help='lateral spacing for universal templates (um)')
    x_centers = Int(required=False, default=4, help='number of x-position groups for template centers')
    nearest_chans = Int(required=False, default=10, help='Number of nearest channels to consider when finding local maxima during spike detection.')
    nearest_templates = Int(required=False, default=100, help='Number of nearest spike template locations to consider when finding local maxima during spike detection.')
    ccg_threshold = Float(required=False, default=0.25, help='Fraction of refractory period violations that are allowed in the CCG compared to baseline; used to perform splits and merges. ')
    acg_threshold = Float(required=False, default=0.20, help='Fraction of refractory period violations that are allowed in the ACG compared to baseline; used to assign "good" units. ') 
    template_seed = Int(required=False, default=0, help='seed to pick which batches are used for finding universal templates')
    cluster_seed = Int(required=False, default=0, help='start seed for clustering')
    
class KS4HelperParameters(DefaultSchema):
    do_CAR = Bool(required=False, default=True, help='set to True to perform common average referencing (median subtraction)')
    save_extra_vars = Bool(required=False, default=False, help='If true, save Wall and pc features in save_to_phy ')
    ks_make_copy = Bool(required=False, default=False, help='If true, make a copy of the original KS output')
    save_preprocessed_copy = Bool(required=False, default=False, help='If true, make a copy of the prprocessed data')
    doFilter = Int(required=False, default=0, help='filter if = 1, skip bp filtering if = 0')
    fproc = String(required=False, default='D:\kilosort_datatemp\temp_wh.dat', help='Processed data file on a fast ssd')
    fshigh = Float(required=False, allow_none=True, default=300, help='high pass filter frequency')
    fslow = Float(required=False, allow_none=True, default=10000, help='low pass filter frequency')
    ks4_params = Nested(ks4_params, required=True, help='Parameters used to auto-generate a Kilosort config file')


class InputParameters(ArgSchema):
    ks4_helper_params = Nested(KS4HelperParameters)
    directories = Nested(Directories)
    ephys_params = Nested(EphysParams)
    common_files = Nested(CommonFiles)
    

class OutputSchema(DefaultSchema): 
    input_parameters = Nested(InputParameters, 
                              description=("Input parameters the module " 
                                           "was run with"), 
                              required=True) 
 
 
class OutputParameters(OutputSchema): 
    message = String()
    execution_time = Float()
