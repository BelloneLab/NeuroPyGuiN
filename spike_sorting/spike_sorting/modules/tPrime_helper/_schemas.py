from argschema import ArgSchema, ArgSchemaParser 
from argschema.schemas import DefaultSchema
from argschema.fields import Nested, InputDir, String, Float, Dict, Int, List, Boolean
from ...common.schemas import EphysParams, Directories
from ..catGT_helper._schemas import CatGTParams


class tPrimeParams(DefaultSchema):
    tPrime_path = InputDir(help='directory containing the TPrime executable.')
    sync_period = Float(default=1.0, help='Period of sync waveform (sec).')
    toStream_sync_params = String(required=False, default='imec0', help='stream identifier for tostream, imec<index>, ni, or obx<index>')
    ni_sync_params = String(required=False, default='', help='deprecated, now read from fyi file')
    ni_ex_list = String(required=False, default='', help='deprecated, now read from fyi file')
    im_ex_list = String(required=False, default='', help='deprecated, now read from fyi file')
    tPrime_3A = Boolean(required=False, default=False, help='is this 3A data?')
    toStream_path_3A = String(required=False, help='full path to toStream edges file')
    fromStream_list_3A = List(String, required=False, help='list of full paths to fromStream edges files')
    psth_ex_str = String(required=False, help='extract string for events.csv for phy psth')
    sort_out_tag = String(required=False, help='tag for sort output (phy) folder')

class InputParameters(ArgSchema):
    tPrime_helper_params = Nested(tPrimeParams)
    catGT_helper_params = Nested(CatGTParams)
    directories = Nested(Directories)
    ephys_params = Nested(EphysParams)

class OutputSchema(DefaultSchema): 
    input_parameters = Nested(InputParameters, 
                              description=("Input parameters the module " 
                                           "was run with"), 
                              required=True) 
 
class OutputParameters(OutputSchema): 

    execution_time = Float()
    