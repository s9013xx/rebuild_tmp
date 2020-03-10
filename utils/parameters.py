import os
import numpy as np
import pandas as pd
import tensorflow as tf
from abc import abstractmethod, ABCMeta
from sklearn.utils import shuffle
from .utils import get_colnames, get_hash_colnames

class ParamsBase(metaclass=ABCMeta):
    """Basic paramter of various layer tpye"""
    def __init__(self, nums, typename = '', output_name = '', precision = 32, optimizer = None):
        self._nums            = nums
        self._typename        = typename
        self._output_name      = output_name
        self._data            = None
        self._batchsize       = None
        self._activation_list = ['None', 'tf.nn.relu']
        self._colnames        = get_colnames(self.typename)
        self._hash_colnames   = None
        self._hashkey      = None
        #self.generate_output_name()

        ### TBD ###
        self._precision       = precision
        self._optimizer       = optimizer

    #def generate_output_name(self):
        #if not self._output_name:
        #    self._output_name = os.path.join(os.getcwd(), self._typename + '_parameters.csv')
        
    def generate_params_with_hashkey(self):
        self.generate_params()
        self.generate_hashkey()

    def generate_hashkey(self):

        #def get_all_data(row_data, colnames):
        #    return [row_data[col] for col in colnames]#list(row_data[colnames].values)

        if self._data is None:
            print("DataFrame is not found!!")
            return
        
        if list(self.data.columns) != self.colnames:
            print("colname order is not correct, auto finetune it!")
            self._data = self._data[self.colnames]

        if not self.hash_colnames:
            self._hash_colnames = get_hash_colnames()
        
        if self.hash_colnames[0] in list(self.data.columns):
            print("Already has hashkey, Do not genreate it again!")
            return

        print("generate the key, please wait...")
        #self._hashkey  = self.data.apply(get_all_data, colnames=self.colnames, axis = 1) ### old version, but too slow
        self._hashkey  = '[' + self.data.astype(str).apply(','.join, axis = 1) + ']'
        self._data[self.hash_colnames[0]] = self.hashkey
    
    def set_data(self, df_):
        self._data = df_
    
    def set_colnames(self, colnames):
        self._colnames = colnames
    
    def auto_generate_elements(self):
        matsize = 'matsize'
        kersize = 'kernelsize'
        ele_mat = 'elements_matrix'
        ele_ker = 'elements_kernel'
        if matsize in self.data.columns:
            if ele_mat not in self.data.columns:
                self._data[ele_mat] =  np.square(self._data[matsize])
        if kersize in self.data.columns:
            if ele_ker not in self.data.columns:
                self._data[ele_ker] =  np.square(self._data[kersize])
        
    def get_shuffle_data(self):
        if self._data is not None:
            return shuffle(self.data).reset_index(drop=True)
        return None

    @abstractmethod
    def generate_params(self):
        '''please Implement it in subclass'''
    
    @abstractmethod
    def get_tensor_from_index(self, index):
        '''please Implement it in subclass'''

    @property
    def nums(self):
        return self._nums
    
    @property
    def typename(self):
        return self._typename

    @property
    def output_name(self):
        return self._output_name
    
    @property
    def data(self):
        return self._data

    @property
    def batchsize(self):
        return self._batchsize

    @property
    def activation_list(self):
        return self._activation_list
    
    @property
    def colnames(self):
        return self._colnames
    
    @property
    def hash_colnames(self):
        return self._hash_colnames

    @property
    def hashkey(self):
        return self._hashkey 

    @property
    def precision(self):
        return self._precision
    
    @property
    def optimizer(self):
        return self._optimizer

class ParamsConv(ParamsBase):
    def __init__(self, nums, typename = 'convolution', output_name = '', precision = 32, optimizer = None):
        self._matsize        = None
        self._kernelsize     = None
        self._channels_in    = None
        self._channels_out   = None
        self._strides        = None
        self._padding        = None
        self._activation_fct = None
        self._use_bias       = None
        self._elements_matrix = None
        self._elements_kernel = None
        super().__init__(nums, typename, output_name, precision, optimizer)
    
    def generate_params(self):
        self._batchsize      = np.random.randint(1,  65, self.nums)
        self._matsize        = np.random.randint(1, 513, self.nums)
        self._kernelsize   = np.zeros(self.nums, dtype=np.int32)
        self._channels_in  = np.zeros(self.nums, dtype=np.int32)
        self._channels_out = np.zeros(self.nums, dtype=np.int32)
        self._strides        = np.random.randint(1,   5, self.nums)
        self._padding        = np.random.randint(0,   2, self.nums)
        self._activation_fct = np.random.randint(0, len(self.activation_list), self.nums)
        self._use_bias       = np.random.choice([True, False], self.nums)

        for i in range(self.nums):
            self._kernelsize[i]   = np.random.randint(1, min(7, self.matsize[i])+1)
            self._channels_in[i]  = np.random.randint(1, 10000/self.matsize[i])
            self._channels_out[i] = np.random.randint(1, 10000/self.matsize[i])

        self._elements_matrix = np.square(self.matsize)
        self._elements_kernel = np.square(self.kernelsize)

        self._data = pd.DataFrame(np.unique(np.array([self.batchsize, self.matsize, self.kernelsize, self.channels_in, self.channels_out, 
            self.strides, self.padding, self.activation_fct, self.use_bias, 
            self.elements_matrix, self.elements_kernel]).transpose(), axis=0), columns=self.colnames)
    
    def get_tensor_from_index(self, index):
        layer = self.data.loc[index, :]
        op = tf.Variable(tf.random_normal([layer['batchsize'].astype(int), 
            layer['matsize'].astype(int), layer['matsize'].astype(int), layer['channels_in'].astype(int)]))
        
        op = tf.layers.conv2d(op, filters=layer['channels_out'].astype(int), 
            kernel_size=[layer['kernelsize'].astype(int), layer['kernelsize'].astype(int)], 
            strides=(layer['strides'].astype(int), layer['strides'].astype(int)), 
            padding=('SAME' if layer['padding'].astype(int) ==1 else 'VALID'),
            activation=eval(self.activation_list[layer['activation_fct'].astype(int)]), 
            use_bias=layer['use_bias'].astype(int),
            name=self.typename)
        return op

    @property
    def matsize(self):
        return self._matsize
    
    @property
    def kernelsize(self):
        return self._kernelsize
    
    @property
    def channels_in(self):
        return self._channels_in

    @property
    def channels_out(self):
        return self._channels_out

    @property
    def strides(self):
        return self._strides
    
    @property
    def padding(self):
        return self._padding

    @property
    def activation_fct(self):
        return self._activation_fct

    @property
    def use_bias(self):
        return self._use_bias
    
    @property
    def elements_matrix(self):
        return self._elements_matrix

    @property
    def elements_kernel(self):
        return self._elements_kernel

class ParamsDense(ParamsBase):
    def __init__(self, nums, typename = 'dense', output_name = '', precision = 32, optimizer = None):
        self._dim_input      = None
        self._dim_output     = None
        self._activation_fct = None
        self._use_bias       = None ### TBD ### Daniel forgot to add this params 
        super().__init__(nums, typename, output_name, precision, optimizer)
    
    def generate_params(self):
        self._batchsize      = np.random.randint(1,   65, self.nums)
        self._dim_input      = np.random.randint(1, 4096, self.nums)
        self._dim_output     = np.random.randint(1, 4096, self.nums)

        self._activation_fct = np.random.randint(0, len(self.activation_list), self.nums)        

        self._data = pd.DataFrame(np.unique(np.array([self.batchsize, self.dim_input, self.dim_output, 
            self.activation_fct]).transpose(), axis=0), columns=self.colnames)
    
    def get_tensor_from_index(self, index):
        layer = self.data.loc[index, :]
        op = tf.Variable(tf.random_normal([layer['batchsize'].astype(int), layer['dim_input'].astype(int)]))
        op = tf.layers.dense(inputs=op, units=layer['dim_output'].astype(int),
            kernel_initializer=tf.ones_initializer(), 
            activation=eval(self.activation_list[layer['activation_fct'].astype(int)]), 
            name = self.typename)
        return op

    @property
    def dim_input(self):
        return self._dim_input
    
    @property
    def dim_output(self):
        return self._dim_output

    @property
    def activation_fct(self):
        return self._activation_fct
    
    @property
    def use_bias(self):
        return self._use_bias

class ParamsPooling(ParamsBase):
    def __init__(self, nums, typename = 'pooling', output_name = '', precision = 32, optimizer = None):
        self._matsize        = None
        self._channels_in    = None
        self._poolsize       = None
        self._strides        = None
        self._padding        = None
        self._activation_fct = None
        self._elements_matrix = None
        super().__init__(nums, typename, output_name, precision, optimizer)

    def generate_params(self):
        self._batchsize = np.random.randint(1,  65, self.nums)
        self._matsize   = np.random.randint(1, 513, self.nums)
        self._channels_in = np.zeros(self.nums, dtype=np.int32)
        self._poolsize    = np.zeros(self.nums, dtype=np.int32)
        self._strides   = np.random.randint(1,   5, self.nums)
        self._padding   = np.random.randint(0,   2, self.nums)
        
        self._activation_fct = np.random.randint(0,len(self.activation_list), self.nums)
        
        for i in range(self.nums):
            self._channels_in[i] = np.random.randint(1, 10000/self.matsize[i])
            self._poolsize[i]    = np.random.randint(1, min(7, self.matsize[i])+1)
        
        self._elements_matrix = np.square(self.matsize)
        
        self._data = pd.DataFrame(np.unique(np.array([self.batchsize, self.matsize, self.channels_in,
            self.poolsize, self.strides, self.padding,
            self.elements_matrix]).transpose(), axis=0), columns=self.colnames)            
    
    def get_tensor_from_index(self, index):
        layer = self.data.loc[index, :]
        op = tf.Variable(tf.random_normal([layer['batchsize'].astype(int), 
            layer['matsize'].astype(int), layer['matsize'].astype(int), layer['channels_in'].astype(int)]))
        op = tf.layers.max_pooling2d(op, pool_size=(layer['poolsize'].astype(int), layer['poolsize'].astype(int)), 
            strides=(layer['strides'].astype(int), layer['strides'].astype(int)), 
            padding=('SAME' if layer['padding'].astype(int)==1 else 'VALID'), 
            name = self.typename)
        # conv=tf.layers.conv2d(
        #       inputs=x,
        #       filters=32,
        #       kernel_size=[5, 5],
        #       padding="same",
        #       activation=tf.nn.relu)
        # op=tf.layers.max_pooling2d(inputs=op, pool_size=[2, 2], strides=2)
        return op

    @property
    def matsize(self):
        return self._matsize
    
    @property
    def channels_in(self):
        return self._channels_in
    
    @property
    def poolsize(self):
        return self._poolsize
    
    @property
    def strides(self):
        return self._strides 

    @property
    def padding(self):
        return self._padding

    @property
    def elements_matrix(self):
        return self._elements_matrix

