from argschema import ArgSchema, ArgSchemaParser 
from argschema.schemas import DefaultSchema
from argschema.fields import Nested, InputDir, String, Float, Dict, Int, Bool, NumpyArray
from ...common.schemas import EphysParams, Directories, CommonFiles


class PyKilosortHelperParameters(DefaultSchema):
    preprocessing_function = String(required=True, default='kilosort2', help='Preprocessing function. Valid values: {"kilosort2", "destriping"} ')
    alf_location = String(required=False, default='', help='ALF location under the results directory')
    copy_fproc = Int(required=False, default=1, help='Copy processed binary to output directory')
    fproc = String(required=False, default='D:\kilosort_datatemp\temp_wh.dat',
                    help='Processed data file on a fast ssd')
    seed = Int(required=False, default=42, help="seed for deterministic output")
    ks2_mode = Bool(required=False, default=False, help='Use ClusterSingleBatches and reorder')
    perform_drift_registration = Bool(required=False, default=True, help='Estimate electrode drift and apply registration')
    car = Bool(required=False, default=True, help='set to True to perform common average referencing (median subtraction)')
    Th = String(required=False, default='[10 4]', help='threshold last pass can be lower')
    ThPre = Float(required=False, default=8, help='threshold crossings for pre-clustering (in PCA projection space)')
    lam = Float(required=False, default=10, help='how important is the amplitude penalty (like in Kilosort1, 0 means not used,10 is average, 50 is a lot)')
    AUCsplit = Float(required=False, default=0.9, help='splitting a cluster at the end requires at least this much isolation for each sub-cluster (max=1)')
    minFR = Float(required=False, default=1.0/50, help='minimum spike rate (Hz), if a cluster falls below this for too long it gets removed')
    momentum = String(required=False, default='[20 400]', help='number of samples to average over (annealed from first to second value)')
    sig_datashift = Float(required=True, default=20.0, help='sigma for the Gaussian process smoothing')
    sigmaMask = Float(required=False, default=30, help='spatial constant in um for computing residual variance of spike')
    fshigh = Float(required=False, allow_none=True, default=300, help='high pass filter frequency')
    fslow = Float(required=False, allow_none=True, help='low pass filter frequency')
    minfr_goodchannels = Float(required=False, default=0.1, help='minimum firing rate on a "good" channel (0 to skip)')
    whiteningRange = Int(required=False, default=32, help='number of channels to use for whitening each channel')
    save_temp_files: bool = Bool(required=False, default=True, help='keep temporary files created while running')
    deterministic_mode =Bool(required=False, default=True, help='make output deterministic by sorting spikes before applying kernels')
    output_filename = String(required=False, allow_none=True, help='optionally save registered data to a new binary file')
    nblocks = Int(required=False, default=5, help='number of blocks used to segment the probe when tracking drift, 0 == do not track, 1 == rigid, > 1 == non-rigid')
    doFilter = Int(required=False, default=0, help='filter if = 1, skip bp filtering if = 0')


class InputParameters(ArgSchema):
    pykilosort_helper_params = Nested(PyKilosortHelperParameters)
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
