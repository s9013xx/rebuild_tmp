import re
import os
import sys
import json
import time
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import stats
from datetime import datetime
from termcolor import colored
# from tensorflow.python.client import timeline
# from ..utils.utils import get_support_devices, get_colnames, get_hash_colnames, get_time_colnames
# from ..utils.parameters import ParamsConv, ParamsDense, ParamsPooling
from ..utils.utils import write_file
from ..utils.utils import append_file

class Recorders(object):
    """ "Store Data infos """
    def __init__(self, json_filename = None, json_data = None,
                str_replica_cpu = '(replica:0)*(CPU:0)+ (Compute)+', 
                str_replica_gpu    = '(replica:0)*(GPU:0)+ (Compute)+', 
                str_all_compute    = '(GPU:0)*(all Compute)',
                str_transpose_in   = 'TransposeNHWCToNCHW',
                str_transpose_out  = 'TransposeNCHWToNHWC',
                str_memcpyD2H      = 'memcpy',
                str_retval         = 'retval'):

        self.json_filename = json_filename
        self.json_data     = json_data
        self.init_json_data()

        self.eztags = EasyTags(self.json_data, str_replica_cpu, str_replica_gpu, str_all_compute, 
                    str_transpose_in, str_transpose_out, str_memcpyD2H, str_retval)
        self.init_time = self.eztags.init_time
        self.replica_transpose_in = RecorderBase("replica_transpose_in", self.init_time,
            self.eztags.replica_gpu.pid, self.eztags.transpose_in.search_pattern, self.json_data)
        
        self.compute_transpose_in = Recorder_CmdInHelp("compute_transpose_in", self.init_time,
            self.eztags.all_compute.pid, self.eztags.transpose_in.search_pattern, self.json_data, self.replica_transpose_in.existed)
        
        self.replica_transpose_out = RecorderBase("replica_transpose_out", self.init_time,
            self.eztags.replica_gpu.pid, self.eztags.transpose_out.search_pattern, self.json_data)
        
        self.compute_transpose_out = RecorderBase("compute_transpose_out", self.init_time,
            self.eztags.all_compute.pid, self.eztags.transpose_out.search_pattern, self.json_data)
        
        self.memcpyD2H = Recorder_MemcpyD2H("memcpyD2H", self.init_time,
            self.eztags.memcpyD2H.pid, self.eztags.memcpyD2H.search_pattern, self.json_data)

        self.retval = RecorderBase("retval", self.init_time,
            self.eztags.retval.pid, self.eztags.retval.search_pattern, self.json_data)
        
        self.first_gpu = Recorder_Frist("frist_compute", self.init_time,
            self.eztags.all_compute.pid, None, self.json_data)
        
        self.last_gpu = Recorder_Last("last_compute", self.init_time,
            self.eztags.all_compute.pid, None, self.json_data, self.memcpyD2H)
        self.result_time()
    
    def init_json_data(self):
        if not self.json_data:
            if not self.json_filename:
                raise FileNotFoundError("Json Data Not Found!!")
            with open(self.json_filename, 'r') as f:
                self.json_data = json.load(f)
    
    def result_time(self):
        self.transOut_time   = self.compute_transpose_out.wall_time
        self.last_gpu_time   = self.last_gpu.start_time + self.last_gpu.wall_time 
        self.memcpyD2H_time  = self.memcpyD2H.start_time + self.memcpyD2H.wall_time
        self.preprocess_time = self.first_gpu.start_time + self.compute_transpose_in.wall_time
        self.execution_time  = self.last_gpu_time - self.preprocess_time - self.transOut_time
        self.memcpy_time     = self.memcpyD2H_time - self.last_gpu_time + self.transOut_time
        self.retval_time     = self.retval.start_time + self.retval.wall_time - self.memcpyD2H_time
        self.retval_half_time = self.retval_time / 2 
        if self.retval.start_time:
            self.sess_time = self.retval.start_time + self.retval.wall_time
        elif self.memcpyD2H_time:
            self.sess_time = self.memcpyD2H_time
        else:
            self.sess_time = self.last_gpu_time


    def __str__(self):
        tmp_str  = "name:{}, s: {}, w:{}\n".format(self.replica_transpose_in.name, self.replica_transpose_in.start_time, self.replica_transpose_in.wall_time)
        tmp_str += "name:{}, s: {}, w:{}\n".format(self.compute_transpose_in.name, self.compute_transpose_in.start_time, self.compute_transpose_in.wall_time)
        tmp_str += "name:{}, s: {}, w:{}\n".format(self.replica_transpose_out.name, self.replica_transpose_out.start_time, self.replica_transpose_out.wall_time)
        tmp_str += "name:{}, s: {}, w:{}\n".format(self.compute_transpose_out.name, self.compute_transpose_out.start_time, self.compute_transpose_out.wall_time)
        tmp_str += "name:{}, s: {}, w:{}\n".format(self.memcpyD2H.name, self.memcpyD2H.start_time, self.memcpyD2H.wall_time)
        tmp_str += "name:{}, s: {}, w:{}\n".format(self.retval.name, self.retval.start_time, self.retval.wall_time)
        tmp_str += "name:{}, s: {}, w:{}\n".format(self.first_gpu.name, self.first_gpu.start_time, self.first_gpu.wall_time)
        tmp_str += "name:{}, s: {}, w:{}\n".format(self.last_gpu.name, self.last_gpu.start_time, self.last_gpu.wall_time)
        tmp_str += "*" * 40 + "\n" + "[Result]\n"
        tmp_str += "preprocess:       {} ms\n".format(self.preprocess_time/1000)
        tmp_str += "execution_time:   {} ms\n".format(self.execution_time/1000)
        tmp_str += "memcpy_time:      {} ms\n".format(self.memcpy_time/1000)
        tmp_str += "retval_time:      {} ms\n".format(self.retval_time/1000)
        tmp_str += "retval_half_time: {} ms\n".format(self.retval_half_time/1000)
        tmp_str += "session time      {} ms".format(self.sess_time/1000)
        return tmp_str

class RecorderBase(object):
    """Store Data info"""
    def __init__(self, name, init_time, pid, pattern, json_data):
        self._name       = name
        self._init_time  = init_time
        self._pid        = pid
        self._pattern    = pattern
        self._existed    = False
        self._wall_time  = 0
        self._start_time = 0
        self.set_time(json_data)
    
    def set_time(self, json_data):
        if self._existed:
            return
        for item in json_data['traceEvents']:
            if self._existed:
                break
            if 'pid' in item and item['pid'] == self._pid:
                if 'args' in item and re.search(self._pattern, item['args']['name'], re.M|re.I):
                    if 'ts' in item and 'dur' in item:
                        self._existed = True 
                        #print(item, self._pattern)
                        self._start_time = float(item['ts']) - self.init_time
                        self._wall_time  = item['dur']
    
    @property
    def init_time(self):
        return int(self._init_time)

    @property
    def name(self):
        return self._name

    @property
    def start_time(self):
        return int(self._start_time)
    
    @property
    def wall_time(self):
        return int(self._wall_time)
    
    @property
    def existed(self):
        return self._existed
    
class Recorder_CmdInHelp(RecorderBase):
    """Store Data info for transposeIn (Maybe not found name in pid)"""
    def __init__(self, name, init_time, pid, pattern, json_data, cmd_existed=False):
        self.cmd_existed = cmd_existed
        super().__init__(name, init_time, pid, pattern, json_data)
        
    
    def set_time(self, json_data):
        if self._existed:
            return
        if self.cmd_existed: #Frist is transpose in 
            first_exe = None 
            first_time = sys.maxsize
            for item in json_data['traceEvents']:
                if 'pid' in item and item['pid'] == self._pid:
                    if 'ts' in item and 'dur' in item and item['ts'] < first_time:
                        first_time = item['ts']
                        first_exe  = item
            if first_exe:
                self._existed = True 
                self._start_time = float(first_exe['ts']) - self.init_time
                self._wall_time  = first_exe['dur']
        else:
            for item in json_data['traceEvents']:
                if 'pid' in item and item['pid'] == self._pid:
                    if 'args' in item and re.search(self._pattern, item['args']['name'], re.M|re.I):
                        if 'ts' in item and 'dur' in item:
                            self._existed = True 
                            self._start_time = float(item['ts']) - self.init_time
                            self._wall_time  = item['dur']

class Recorder_Frist(RecorderBase):
    """Store Data info for first data"""
    def __init__(self, name, init_time, pid, pattern, json_data, cmd_existed=False):
        self.cmd_existed = cmd_existed
        super().__init__(name, init_time, pid, pattern, json_data)
        
    def set_time(self, json_data):
        if self._existed:
            return
        first_exe = None 
        first_time = sys.maxsize
        for item in json_data['traceEvents']:
            if 'pid' in item and item['pid'] == self._pid:
                if 'ts' in item and 'dur' in item and item['ts'] < first_time:
                    first_time = item['ts']
                    first_exe  = item
        if first_exe:
            self._existed = True 
            self._start_time = float(first_exe['ts']) - self.init_time
            self._wall_time  = first_exe['dur']

class Recorder_Last(RecorderBase):
    """Store Data info for Last data"""
    def __init__(self, name, init_time, pid, pattern, json_data, memcpyD2H):
        self.memcpyD2H = memcpyD2H
        super().__init__(name, init_time, pid, pattern, json_data)
    def set_time(self, json_data):
        if self._existed:
            return
        last_exe = None 
        last_time = self.init_time
        for item in json_data['traceEvents']:
            if 'pid' in item and item['pid'] == self._pid:
                if 'ts' in item and 'dur' in item and item['ts'] >= last_time:
                    if self.memcpyD2H.existed and (self.memcpyD2H.start_time + self.init_time) == item['ts']:
                        continue
                    else:
                        last_time = item['ts']
                        last_exe  = item
        if last_exe:
            self._existed = True 
            self._start_time = float(last_exe['ts']) - self.init_time
            self._wall_time  = last_exe['dur']
                        
class Recorder_MemcpyD2H(RecorderBase):
    """Store Data info for memcpyD2H"""
    def __init__(self, name, init_time, pid, pattern, json_data):
        super().__init__(name, init_time, pid, pattern, json_data)
    def set_time(self, json_data):
        if self._existed:
            return
        for item in json_data['traceEvents']:
            if self._existed:
                break
            if 'pid' in item and item['pid'] == self._pid:
                if 'args' in item and re.search(self._pattern, item['args']['name'], re.M|re.I):
                    if 'ts' in item and 'dur' in item:
                        self._existed = True 
                        self._start_time = float(item['ts']) - self.init_time
                        self._wall_time  = item['dur']
                elif 'args' in item and re.search(self._pattern, item['args']['op'], re.M|re.I):
                    if 'ts' in item and 'dur' in item:
                        self._existed = True 
                        self._start_time = float(item['ts']) - self.init_time
                        self._wall_time  = item['dur']

class EasyTags(object):
    """"Tags of all important process name"""
    def __init__(self, json_data, str_replica_cpu = '(replica:0)*(CPU:0)+ (Compute)+', 
                    str_replica_gpu    = '(replica:0)*(GPU:0)+ (Compute)+', 
                    str_all_compute    = '(GPU:0)*(all Compute)',
                    str_transpose_in   = 'TransposeNHWCToNCHW',
                    str_transpose_out  = 'TransposeNCHWToNHWC',
                    str_memcpyD2H      = 'memcpy',
                    str_retval         = 'retval'):
        self.json_data     = json_data
        self.str_replica_cpu   = str_replica_cpu
        self.str_replica_gpu   = str_replica_gpu
        self.str_all_compute   = str_all_compute
        self.str_transpose_in  = str_transpose_in
        self.str_transpose_out = str_transpose_out
        self.str_memcpyD2H = str_memcpyD2H
        self.str_retval    = str_retval
        self.init_time     = sys.maxsize
        self.replica_cpu   = EasyTag('replica_cpu', self.str_replica_cpu)
        self.replica_gpu   = EasyTag('replica_gpu', self.str_replica_gpu)
        self.all_compute   = EasyTag('all_compute', self.str_all_compute)
        self.transpose_in  = EasyTag('transpose_in', self.str_transpose_in)
        self.transpose_out = EasyTag('transpose_out', self.str_transpose_out)
        self.memcpyD2H     = EasyTag('memcpyD2H', self.str_memcpyD2H)
        self.retval        = EasyTag('retval', self.str_retval)
        self.reset()
    
    def reset(self):
        for item in self.json_data['traceEvents']:
            if 'ts' in item and item['ts'] < self.init_time:
                self.init_time = item['ts']
            if 'name' in item and item['name'] == 'process_name':
                if re.search(self.str_all_compute, item['args']['name'], re.M|re.I):
                    self.all_compute.set_pid(item['pid'])
                if re.search(self.str_replica_gpu, item['args']['name'], re.M|re.I):
                    self.replica_gpu.set_pid(item['pid'])
                if re.search(self.str_replica_cpu, item['args']['name'], re.M|re.I):
                    self.replica_cpu.set_pid(item['pid'])
                if re.search(self.str_memcpyD2H, item['args']['name'], re.M|re.I):
                    self.memcpyD2H.set_pid(item['pid'])
                
        # Second Round
        for item in self.json_data['traceEvents']:
            if self.transpose_in.existed and self.transpose_out.existed and self.retval.existed:
                break
            if self.replica_gpu.existed and 'args' in item:
                if 'pid' in item and self.replica_gpu.pid == item['pid']:
                    if re.search(self.str_transpose_in, item['args']['name'], re.M|re.I):
                        self.transpose_in.set_pid(self.all_compute.pid)
                    if re.search(self.str_transpose_in, item['args']['name'], re.M|re.I):
                        self.transpose_out.set_pid(self.all_compute.pid)
            if self.replica_cpu.existed and 'args' in item:
                if 'pid' in item and self.replica_cpu.pid == item['pid']:
                    if re.search(self.str_retval, item['args']['name'], re.M|re.I):
                        self.retval.set_pid(self.replica_cpu.pid)
        
    def __str__(self):
        tmp_str  = "[Init time] {}\n".format(self.init_time)
        tmp_str += "[{}] existed: {}, pid is {}\n".format(self.replica_cpu.name, self.replica_cpu.existed, self.replica_cpu.pid)
        tmp_str += "[{}] existed: {}, pid is {}\n".format(self.replica_gpu.name, self.replica_gpu.existed, self.replica_gpu.pid)
        tmp_str += "[{}] existed: {}, pid is {}\n".format(self.all_compute.name, self.all_compute.existed, self.all_compute.pid)
        tmp_str += "[{}] existed: {}, pid is {}\n".format(self.transpose_in.name, self.transpose_in.existed, self.transpose_in.pid)
        tmp_str += "[{}] existed: {}, pid is {}\n".format(self.transpose_out.name, self.transpose_out.existed, self.transpose_out.pid)
        tmp_str += "[{}] existed: {}, pid is {}\n".format(self.memcpyD2H.name, self.memcpyD2H.existed, self.memcpyD2H.pid)
        tmp_str += "[{}] existed: {}, pid is {}\n".format(self.retval.name, self.retval.existed, self.retval.pid)
        return tmp_str

class EasyTag(object):
    "The Tags of the important process name"
    def __init__(self, name, pattern):
        self._name = name
        self._existed = False
        self._pid  = None
        self._pattern = pattern

    def set_pid(self, pid):
        if not self._existed:
            self._pid = pid
            self._existed = True        

    def set_pattern(self, pattern):
        self._pattern = pattern

    @property
    def pid(self):
        return self._pid
    
    @property
    def existed(self):
        return self._existed
    
    @property
    def name(self):
        return self._name
    
    @property
    def search_pattern(self):
        return self._pattern


class Parse_Timeline(object):
    """ "Store Data infos """
    def __init__(self, predition_layertype, device, input_timeline_profile_path, output_parser_path,
        output_parser_file, replica_cpu, replica_gpu, all_compute, transpose_in, transpose_out,
        memcpyD2H, retval):

        self.predition_layertype = predition_layertype
        self.device = device
        self.input_timeline_profile_path = input_timeline_profile_path
        self.output_parser_path = output_parser_path
        self.output_parser_file = output_parser_file

        self.replica_cpu = replica_cpu
        self.replica_gpu = replica_gpu
        self.all_compute = all_compute
        self.transpose_in = transpose_in
        self.transpose_out = transpose_out
        self.memcpyD2H = memcpyD2H
        self.retval = retval

    def execute(self):
        # flags = read_collect_timeline_data_flags()
        # warn_tag = colored('[Warn] ', 'red', attrs=['blink']) 
        # success_tag = colored('[Success] ', 'green')
        # if not os.path.isdir(flags.output_path):
        #     os.makedirs(flags.output_path)

        # if not flags.output_filename:
        #     tmp_str = flags.predition_layertype + '_' + flags.device + '.csv'
        #     flags.output_filename = os.path.join(flags.output_path, tmp_str)
        #     print(warn_tag + 'Auto create file: ' + flags.output_filename)

        all_files = os.listdir(self.input_timeline_profile_path)
        print(all_files)
        index = 0
        for filename in all_files:
            full_filename = os.path.join(self.input_timeline_profile_path, filename)
            recorders = Recorders(full_filename, None, self.replica_cpu,
                self.replica_gpu, self.all_compute, self.transpose_in,
                self.transpose_out, self.memcpyD2H, self.retval)
            date_ele = {
                'hashkey':            os.path.splitext(filename)[0],
                'preprocess_time':    recorders.preprocess_time,
                'execution_time':     recorders.execution_time,
                'memcpy_time':        recorders.memcpy_time, 
                'retval_time':        recorders.retval_time,
                'retval_half_time':   recorders.retval_half_time,
                'memcpy_retval':      recorders.memcpy_time + recorders.retval_time,
                'memcpy_retval_half': recorders.memcpy_time + recorders.retval_half_time,
                'sess_time':          recorders.sess_time
            }
            for key, value in date_ele.items():
                if key == 'hashkey':
                    continue
                date_ele[key] = value / 1000
            df_ele = pd.DataFrame(data = date_ele, index=[0])
            if index==0:
                # print('1.', flags.output_filename)
                # df_ele.to_csv(flags.output_filename, index=False)
                write_file(df_ele, self.output_parser_path, self.output_parser_file)
                index = 1
            else:
                # print('2.', flags.output_filename)
                # df_ele.to_csv(flags.output_filename, index=False, mode='a', header=False)
                append_file(df_ele, self.output_parser_path, self.output_parser_file)
        success_tag = colored('[Success] ', 'green')
        print(success_tag + 'Timeline collection is end!!')