import os
import sys
import argparse
from .utils import get_support_layers

def read_collect_data_flags():
    parser = argparse.ArgumentParser('Collect Data Paremeters Parser')
    # Benchmarks parameters
    
    # General Parameters
    parser.add_argument('--predition_layertype', '-pl', default='convolution', type=str, choices=get_support_layers(), help='Layer types of neural networks')
    parser.add_argument('--gen_params', '-gp', action="store_true", default=False, help='generate random parameters')
    parser.add_argument('--exe_params', '-ep', action="store_true", default=False, help='execution parameters on specific device')
    parser.add_argument('--profile_params', '-pp', action="store_true", default=False, help='Use tensorflow profiler')

    #Generate Parameters
    parser.add_argument('--num', '-num', type=int, default=110000, help='Number of results to compute')
    parser.add_argument('--shuffle', '-shuffle', type=int, default=1, help='shuffle the data')
    parser.add_argument('--output_params_path', '-opp', type=str, default=os.path.join(os.getcwd(),'golden_struct_values'), help='The path of the output csv filename')
    parser.add_argument('--output_params_file', '-opf', type=str, default='', help='The output csv file name')
    
    #Execution Parameters
    parser.add_argument('--input_params_file_path', '-ipfp', type=str, default='', help='The input params csv file path')
    parser.add_argument('--output_exe_path', '-op', type=str, default=os.path.join(os.getcwd(), 'golden_time_values'), help='The path of the output csv filename')
    parser.add_argument('--output_exe_file', '-of', type=str, default='', help='The output csv filename')
    parser.add_argument('--device', '-d', type=str, default='1080ti', help='Device name as appearing in logfile')
    # parser.add_argument('--log_level', '-ll', default='3', type=str, choices=['0', '1', '2', '3'], help='log level of tensorflow')
    parser.add_argument('--iter_warmup', type=int, default=5, help='Number of iterations for warm-up')
    parser.add_argument('--iter_benchmark', type=int, default=10, help='Number of iterations for benchmark')
    parser.add_argument('--cpu', '-cpu', action="store_true", default=False, help='Force to use CPU to computate')
    # parser.add_argument('--backup_path', type=str, default=os.path.join(os.getcwd(), 'backup'), help='backup path')
    
    #Timeline Profile Parameters
    parser.add_argument('--output_timeline_path', '-otp', type=str, default=os.path.join(os.getcwd(), 'timeline'), help='timeline path')

    args = parser.parse_args()
    return args


# def read_collect_data_flags():
#     parser = argparse.ArgumentParser('Collect Time from Input Data(csv)')
#     # LayerType parameters
#     parser.add_argument('--predition_layertype', '-pl', default='convolution', 
#                         type=str, choices=get_support_layers(), help='Layer types of nn') ### TBD， maybe not need to use this tag, but tmp it 
#     # General parameters
#     parser.add_argument('--input_filename', '-if', type=str, default='', help='The input csv filename')
#     parser.add_argument('--output_path', '-op', type=str, default=os.path.join(os.getcwd(), 'golden_time_values'), help='The path of the output csv filename')
#     parser.add_argument('--output_filename', '-of', type=str, default='', help='The output csv filename')
#     parser.add_argument('--device', '-d', type=str, default='1080ti', help='Device name as appearing in logfile')
#     parser.add_argument('--log_level', '-ll', default='3', type=str, choices=['0', '1', '2', '3'], help='log level of tensorflow')
#     parser.add_argument('--iter_warmup', type=int, default=5, help='Number of iterations for warm-up')
#     parser.add_argument('--iter_benchmark', type=int, default=10, help='Number of iterations for benchmark')
#     parser.add_argument('--cpu', '-cpu', action="store_true", default=False, help='Force to use CPU to computate')
#     parser.add_argument('--profile', '-p', action="store_true", default=False, help='Use tensorflow profiler')
#     parser.add_argument('--timeline_path', type=str, default=os.path.join(os.getcwd(), 'timeline'), help='timeline path')
#     parser.add_argument('--backup_path', type=str, default=os.path.join(os.getcwd(), 'backup'), help='backup path')
#     args = parser.parse_args()
#     return args

def read_collect_timeline_data_flags():
    parser = argparse.ArgumentParser('Collect All Timeline Data from Timeline Path')
    # LayerType parameters
    parser.add_argument('--predition_layertype', '-pl', default='convolution', 
                        type=str, choices=get_support_layers(), help='Layer types of nn') ### TBD， maybe not need to use this tag, but tmp it 
    # Benchmarks parameters
    parser.add_argument('--all_compute', type=str, default='(GPU:0)*(all Compute)', help='search tag - all_compute')
    parser.add_argument('--replica_gpu', type=str, default='(replica:0)*(GPU:0)+ (Compute)+', help='search tag - replica_gpu')
    parser.add_argument('--replica_cpu', type=str, default='(replica:0)*(CPU:0)+ (Compute)+', help='search tag - replica_cpu')
    parser.add_argument('--memcpyD2H', type=str, default='memcpy', help='search tag - memcpy')
    parser.add_argument('--transpose_in', type=str, default='TransposeNHWCToNCHW', help='search tag - transpose_in')
    parser.add_argument('--transpose_out', type=str, default='TransposeNCHWToNHWC', help='search tag - transpose_out')
    parser.add_argument('--retval', type=str, default='retval', help='search tag - retval')
    # General parameters
    parser.add_argument('--timeline_path', type=str, default=os.path.join(os.getcwd(), 'timeline'), help='timeline path')
    parser.add_argument('--device', '-d', type=str, default='1080ti', help='Device name as appearing in logfile')
    parser.add_argument('--output_path', '-op', type=str, default=os.path.join(os.getcwd(), 'golden_timeline_values'), help='The path of the output csv filename')
    parser.add_argument('--output_filename', '-of', type=str, default='', help='The output csv filename')
    args = parser.parse_args()
    return args

def read_timeline_flags():
    parser = argparse.ArgumentParser('Parser for timeline data')
    # Benchmarks parameters
    parser.add_argument('--filename', '-f', type=str, default=os.path.join(os.getcwd(), 'timeline', 'lenet_1_to_1_bs1.json'), help='input jon file')
    parser.add_argument('--all_compute', type=str, default='(GPU:0)*(all Compute)', help='search tag - all_compute')
    parser.add_argument('--replica_gpu', type=str, default='(replica:0)*(GPU:0)+ (Compute)+', help='search tag - replica_gpu')
    parser.add_argument('--replica_cpu', type=str, default='(replica:0)*(CPU:0)+ (Compute)+', help='search tag - replica_cpu')
    parser.add_argument('--memcpyD2H', type=str, default='memcpy', help='search tag - memcpy')
    parser.add_argument('--transpose_in', type=str, default='TransposeNHWCToNCHW', help='search tag - trans_in')
    parser.add_argument('--transpose_out', type=str, default='TransposeNCHWToNHWC', help='search tag - trans_out')
    parser.add_argument('--retval', type=str, default='retval', help='search tag - retval')
    args = parser.parse_args()
    return args

def read_combine_csv_flags():
    parser = argparse.ArgumentParser('Combine CSV data')
    # LayerType parameters
    parser.add_argument('--predition_layertype', '-pl', default='convolution', 
                        type=str, choices=get_support_layers(), help='Layer types of nn') ### TBD， maybe not need to use this tag, but tmp it 
    # General parameters
    parser.add_argument('--device', '-d', type=str, default='1080ti', help='Device name as appearing in logfile')
    parser.add_argument('--input_time_csv', '-itc', type=str, default='', help='The input time csv filename')
    parser.add_argument('--input_timeline_csv', '-itlc', type=str, default='', help='The input timeline csv filename')
    parser.add_argument('--input_time_path', '-itp', type=str, default=os.path.join(os.getcwd(),'golden_time_values'), help='The path of input time csv filename')
    parser.add_argument('--input_timeline_path', '-itlp', type=str, default=os.path.join(os.getcwd(),'golden_timeline_values'), help='The path of input time csv filename')
    
    parser.add_argument('--input_struct_csv', '-isc', type=str, default='', help='The path of input struct csv filename')
    parser.add_argument('--input_struct_path', '-isp', type=str, default=os.path.join(os.getcwd(),'golden_struct_values'), help='The path of input struct csv filename')

    parser.add_argument('--output_path', '-op', type=str, default=os.path.join(os.getcwd(), 'golden_values'), help='The path of the output csv filename')
    parser.add_argument('--output_filename', '-of', type=str, default='', help='The output csv filename')

    args = parser.parse_args()
    return args

def read_data_filter_flags():
    parser = argparse.ArgumentParser('Data Filter Parser')
    # LayerType parameters
    parser.add_argument('--predition_layertype', '-pl', default='convolution', 
                        type=str, choices=get_support_layers(), help='Layer types of nn') ### TBD， maybe not need to use this tag, but tmp it 
    # General parameters
    parser.add_argument('--device', '-d', type=str, default='1080ti', help='Device name as appearing in logfile')
    
    parser.add_argument('--input_filename', '-if', type=str, default='', help='The input csv filename')

    parser.add_argument('--output_path', '-op', type=str, default=os.path.join(os.getcwd(), 'filter_data'), help='The path of the output csv filename')
    parser.add_argument('--output_sub_path', '-osp', type=str, default=os.path.join(os.getcwd(), 'filter_data', 'convolution'), help='The sub path of the output csv filename')
    parser.add_argument('--output_filename', '-of', type=str, default='', help='The output csv filename')
    
    args = parser.parse_args()
    return args

def read_check_timeline_flags():
    parser = argparse.ArgumentParser('Collect Actual Data Parser')
    # LayerType parameters
    parser.add_argument('--predition_layertype', '-pl', default='convolution', 
                        type=str, choices=get_support_layers(), help='Layer types of nn') ### TBD， maybe not need to use this tag, but tmp it 
    # Benchmarks parameters
    parser.add_argument('--filename', '-f', type=str, default=os.path.join(os.getcwd(), 'golden_struct_values','dense_parameters.csv'), help='csv file')
    parser.add_argument('--start_from_csv', '-sfc', type=int, default=1, help='layer start')
    parser.add_argument('--numbers_of_csv', '-nc', type=int, default=1, help='count of layers')
    parser.add_argument('--one_layer', '-ol', action="store_true", default=False, help='just use one layer')

    # General parameters
    parser.add_argument('--iter_warmup', type=int, default=5, help='Number of iterations for warm-up')    
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='Batch_size of data')
    parser.add_argument('--iter_benchmark', type=int, default=10, help='Number of iterations for benchmark')    

    parser.add_argument('--cpu', '-cpu', action="store_true", default=False, help='Benchmark using CPU')
    parser.add_argument('--profile', '-p', action="store_true", default=False, help='profiling')
    parser.add_argument('--timeline_name', type=str, default='', help='output timeline name')
    parser.add_argument('--timeline_path', type=str, default=os.path.join(os.getcwd(), 'check_timeline'), help='timeline path')

    args = parser.parse_args()
    return args