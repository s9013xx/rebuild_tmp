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
from ..utils.utils import write_file
from ..utils.utils import append_file

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

    def execute(self):
        df_  = pd.read_csv(self.input_params_file_path)
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
        print(params.data)

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
                for _ in range(self.iter_warmup):
                    sess.run(op)
            except:
                continue
            # Do Benchmark
            hash_colname = get_hash_colnames()[0]
            
            time_list = []
            for _ in range(self.iter_benchmark):
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
                write_file(df_ele, self.output_exe_path, self.output_exe_file)
            else:
                append_file(df_ele, self.output_exe_path, self.output_exe_file)
