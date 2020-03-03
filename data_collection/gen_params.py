import os
from termcolor import colored
# from flags import read_random_generate_paramters_flags
# from ..utils.flags import read_collect_data_flags
from ..utils.parameters import ParamsConv, ParamsDense, ParamsPooling


class Gen_Params(object):
    """ "Store Data infos """
    def __init__(self, predition_layertype, num, shuffle):
        self.predition_layertype = predition_layertype
        self.num = num
        self.shuffle = shuffle

    def generate(self):

        if self.predition_layertype == 'convolution':
            Params = ParamsConv(self.num, self.predition_layertype)
        elif self.predition_layertype == 'pooling':
            Params = ParamsPooling(self.num, self.predition_layertype)
        elif self.predition_layertype == 'dense':
            Params = ParamsDense(self.num, self.predition_layertype)
        else:
            print("This type of layer is not support!")
            return
        
        Params.generate_params_with_hashkey()
        df_data = Params.get_shuffle_data() if self.shuffle else Params.data

        return df_data