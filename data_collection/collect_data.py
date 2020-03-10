import os
import sys
from ..utils.flags import read_collect_data_flags
from ..utils.utils import write_file
from ..utils.utils import check_config
from .gen_params import Gen_Params
from .exe_params import Exe_Params
from .pro_params import Pro_Params
# from parameters import ParamsConv, ParamsDense, ParamsPooling

def main():
    flags = read_collect_data_flags()

    if flags.gen_params:
        print('gen_random_params')
        gen_params = Gen_Params(flags.predition_layertype, flags.num, flags.shuffle)
        gen_result = gen_params.generate()
        if not flags.output_params_file:
            flags.output_params_file = flags.predition_layertype + '_parameters.csv'
        write_file(gen_result, flags.output_params_path, flags.output_params_file)
        
    if flags.exe_params:
        print('exe_params')
        if not flags.input_params_file_path:
            flags.input_params_file_path = os.path.join(flags.output_params_path, flags.output_params_file)
        if not flags.output_exe_file:
            flags.output_exe_file = flags.predition_layertype + '_' + flags.device + '.csv'
        check_config(flags)
        exe_params = Exe_Params(flags.predition_layertype, flags.input_params_file_path,
            flags.output_exe_path, flags.output_exe_file, flags.iter_warmup, flags.iter_benchmark)
        exe_params.execute()

    if flags.profile_params:
        print('profile_params')
        if not flags.input_params_file_path:
            flags.input_params_file_path = os.path.join(flags.output_params_path, flags.output_params_file)
        check_config(flags)
        profile_params = Pro_Params(flags.predition_layertype, flags.device, flags.input_params_file_path,
            flags.output_timeline_profile_path, flags.iter_warmup)
        profile_params.execute()


if __name__ == '__main__':
    main()