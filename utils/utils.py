import os
import sys
from termcolor import colored
from datetime import datetime
import tensorflow as tf

def get_support_devices():
    dict_ = {
        '1080ti': 'gpu',
    }
    return dict_

def get_support_layers():
    return ['convolution', 'pooling', 'dense']

def get_colnames(typename):
    if typename == 'convolution':
        return get_cov_colnames()
    elif typename == 'dense':
        return get_dense_colnames()
    elif typename == 'pooling':
        return get_pool_colnames()
    else:
        print("This type of layer is not support!")
        return

def get_hash_colnames():
    return ['hashkey']

def get_cov_colnames():
    return ['batchsize', 'matsize', 'kernelsize', 'channels_in', 'channels_out', 'strides', 'padding', 'activation_fct', 'use_bias', 'elements_matrix', 'elements_kernel']

def get_dense_colnames():
    return ['batchsize', 'dim_input', 'dim_output', 'activation_fct']

def get_pool_colnames():
    return ['batchsize', 'matsize', 'channels_in', 'poolsize', 'strides', 'padding', 'elements_matrix']

def get_time_colnames():
    return ['time_max', 'time_min', 'time_median', 'time_mean', 'time_trim_mean']

def get_profile_colnames():
    return ['preprocess_time', 'execution_time', 'memcpy_time', 'retval_time', 'retval_half_time', 'memcpy_retval', 'memcpy_retval_half', 'sess_time']#, 'elements_matrix', 'elements_kernel']

def get_colnames_from_dict():
    conv_colnames  = get_cov_colnames()
    dense_colnames = get_dense_colnames()
    pool_colnames  = get_pool_colnames()
    time_colnames  = get_time_colnames()
    profile_colnames = get_profile_colnames()
    cols_dict = {
        'convolution': conv_colnames,
        'dense': dense_colnames,
        'pooling': pool_colnames,
        'profile': profile_colnames,
        'time': time_colnames,
        'hash': get_hash_colnames()
    }
    return cols_dict

def check_config(flags):
    warn_tag = colored('[Warn] ', 'red', attrs=['blink']) 
    success_tag = colored('[Success] ', 'green')
    device_dict = get_support_devices()
    if flags.device in device_dict.keys():
        foolproof_device = device_dict[flags.device]
        if foolproof_device.lower() == 'cpu':
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        print(success_tag + 'foolproof: Use ' + foolproof_device + ' to computate')
    if flags.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        print(warn_tag + 'Force to use cpu to compuate')
    
    print(success_tag + 'GPU is found') if tf.test.gpu_device_name() else print(warn_tag + 'GPU is Not found')

def backup_file(file_path):
    ### Backup the Output CSV file
    warn_tag = colored('[Warn] ', 'red', attrs=['blink']) 
    base_name = os.path.basename(file_path)
    path = os.path.dirname(file_path)
    split_basname = os.path.splitext(base_name)
    bk_filename = split_basname[0] + '_' + datetime.now().strftime('%m%d-%H%M%S') + split_basname[1]
    print(warn_tag + 'Ouput CSV: ' + file_path + ' is existed, backup as ' + bk_filename)
    os.rename(file_path, os.path.join(path, bk_filename))

def write_file(data, path, file):
    print('path', path)
    print('file', file)
    file_path = os.path.join(path, file)
    warn_tag = colored('[Warn] ', 'red', attrs=['blink']) 
    if not os.path.isdir(path):
        os.makedirs(path)
    else:
        if os.path.isfile(file_path):
            backup_file(file_path)

    print(warn_tag + 'Auto create file: ' + file_path)
    data.to_csv(file_path, index=False)

def append_file(data, path, file):
    file_path = os.path.join(path, file)
    data.to_csv(file_path, index=False, mode='a', header=False)
