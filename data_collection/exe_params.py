import os
import time
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import stats
from datetime import datetime
from termcolor import colored
from tensorflow.python.client import timeline
from ..utils.utils import get_support_devices, get_colnames, get_hash_colnames, get_time_colnames
from ..utils.parameters import ParamsConv, ParamsDense, ParamsPooling

class Exe_Params(object):
    """ "Store Data infos """
    def __init__(self, predition_layertype, input_params_file_path, output_exe_path,
        output_exe_file, iter_warmup, iter_benchmark):

        self.predition_layertype = predition_layertype
        self.input_params_file_path = input_params_file_path
        self.output_exe_path = output_exe_path
        self.output_exe_file = output_exe_file
        self.iter_warmup = iter_warmup
        self.iter_benchmark = iter_benchmark
        self.output_exe_file_path = os.path.join(output_exe_path, output_exe_file)

    def execute(self):
        print(self.input_params_file_path)
        df_  = pd.read_csv(self.input_params_file_path)
        print(df_)

        colnames = get_colnames(self.predition_layertype)
    
        if self.predition_layertype == 'convolution':
            params = ParamsConv(df_.shape[0], self.predition_layertype)
        elif self.predition_layertype == 'dense':
            params = ParamsDense(df_.shape[0], self.predition_layertype)
        elif self.predition_layertype == 'pooling':
            params = ParamsPooling(df_.shape[0], self.predition_layertype)
        else:
            print("This type of layer is not support!")
            return
        
        params.set_data(df_)
        params.set_colnames(colnames)
        params.auto_generate_elements()
        params.generate_hashkey()
        # print(params.data)
        
        run_metadata = tf.RunMetadata() ### metadata tags is here !!!
        ### Generate an Run each opearation in dataframe
        for index in range(params.data.shape[0]):
            print('========== ', params.typename, index , '==========')
            tf.reset_default_graph()
            op = params.get_tensor_from_index(index)
            sess = tf.Session()

            # session init 
            if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
                init = tf.initialize_all_variables()
            else:
                init = tf.global_variables_initializer()
            sess.run(init)

            try:
                # Do WarmUp               
                for _ in range(flags.iter_warmup):
                    sess.run(op)
            except:
                continue
            # Do Benchmark
            hash_colname = get_hash_colnames()[0]
            if flags.profile:
                profile_json_name = str(params.data.loc[index, hash_colname]) + '.json'
                filename = os.path.join(flags.timeline_path, profile_json_name)
                sess.run(op, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                    run_metadata=run_metadata)
                tl = timeline.Timeline(run_metadata.step_stats)
                ctf = tl.generate_chrome_trace_format()
                with open(filename, 'w') as f:
                    f.write(ctf) 
                time.sleep(0.005)
            else:
                time_list = []
                for _ in range(flags.iter_benchmark):
                    start_time = time.time()
                    sess.run(op)
                    time_list.append(((time.time()-start_time) * 1000))
                
                time_list = np.array(time_list)
                time_data_ele = {
                    'hashkey':        str(params.data.loc[index, hash_colname]),
                    'time_max':       np.amax(time_list),
                    'time_min':       np.amin(time_list),
                    'time_median':    np.median(time_list),
                    'time_mean':      np.mean(time_list), 
                    'time_trim_mean': stats.trim_mean(time_list, 0.1),
                }
                 
                df_ele = pd.DataFrame(data = time_data_ele, index=[0])
                print("time_mean: {} ms".format(time_data_ele['time_mean']))
                #return
                if index==0: 
                    df_ele.to_csv(flags.output_filename, index=False)
                else:
                    df_ele.to_csv(flags.output_filename, index=False, mode='a', header=False)



    # if not os.path.isdir(flags.output_path) and not flags.profile:
    #     os.makedirs(flags.output_path)

    # if (not os.path.isdir(flags.timeline_path)) and flags.profile:
    #     os.makedirs(flags.timeline_path)

    # if not os.path.isdir(flags.backup_path):
    #     os.makedirs(flags.backup_path)
   
    # if not flags.output_filename:
    #     tmp_str = flags.predition_layertype + '_' + flags.device + '.csv'
    #     flags.output_filename = os.path.join(flags.output_path, tmp_str)

    # if os.path.isfile(flags.output_filename) and not flags.profile:
    #     ### Backup the Output CSV file
    #     base_name = os.path.basename(flags.output_filename)
    #     split_basname = os.path.splitext(base_name)
    #     bk_filename = split_basname[0] + '_' + datetime.now().strftime('%m%d-%H%M%S') + split_basname[1]
    #     print(warn_tag + 'Ouput CSV: ' + flags.output_filename + ' is existed, backup as ' + bk_filename)
    #     os.rename(flags.output_filename, os.path.join(flags.backup_path, bk_filename))

# def main():


# if __name__ == '__main__':
#     main()
