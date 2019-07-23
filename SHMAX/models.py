import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from numpy.lib.stride_tricks import as_strided
from sklearn.decomposition import DictionaryLearning,MiniBatchDictionaryLearning,SparseCoder

def to_windows(input, kernel_size, step_size=1):
    out_shape = input.shape[:2]
    out_shape += ((input.shape[2] - kernel_size)//step_size +1, (input.shape[3] - kernel_size)//step_size +1) + (kernel_size,kernel_size)
    input_srides = input.strides 
    strides = input_srides[:2] + tuple(np.array(input.strides[-2:])*step_size) + input_srides[-2:]
    return as_strided(input, shape = out_shape, strides = strides,writeable=False)


class SparseLayer(nn.Module): # Convolution-like
    def __init__(self,input_channels,output_channels,filter_size,batch_size=500):
        super(SparseLayer,self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.dictionary_size = output_channels
        self.filter_size = filter_size
        self.pad = self.filter_size // 2
        self.n_features = self.filter_size**2 * self.input_channels
        self.batch_size = batch_size
        # if isinstance(self.filter_size,int): 
        # else:
        #     self.n_features = 1
        #     for k in self.filter_size:
        #         self.n_features *= k
        
        self.dictionary = DictionaryLearning(n_components = self.dictionary_size)
    # Actually a reconstruction
    def forward(self,input):
        # code : (batch_size, dictionary_size)
        coder = SparseCoder(dictionary = self.dictionary.components_)
        padded = np.pad(input,((0,0),(0,0),(self.pad,self.pad),(self.pad,self.pad)),mode='constant')
        
        windowed = to_windows(padded,kernel_size=self.filter_size)
        del padded

        reshaped = windowed.reshape(-1,self.n_features)
        del windowed

        code = coder.transform(reshaped)
        # code : (Batch_size * height * width, output channels)
        return torch.as_tensor(np.transpose(code.reshape((input.shape[0],input.shape[2],input.shape[3],self.output_channels)),(0,3,1,2)))


    def forward_conv(self,input):
        weights = torch.as_tensor(self.dictionary.components_.reshape(self.output_channels,self.input_channels,self.filter_size,self.filter_size))
        
        padded = np.pad(input,((0,0),(0,0),(self.pad,self.pad),(self.pad,self.pad)),mode='constant')

        return F.conv2d(torch.as_tensor(padded),weights)

    def train_dictionary(self,input):
        # input : (batch_size, channels, height, width)
        patches = self.extract_patches(input)
        self.dictionary.fit(patches)
        # self.dictionary_trainer.partial_fit(reshaped)

    def extract_patches(self,input):
        padded = np.pad(input,((0,0),(0,0),(self.pad,self.pad),(self.pad,self.pad)),mode='constant')
        
        windowed = to_windows(padded,kernel_size=self.filter_size)
        del padded

        reshaped = windowed.reshape(-1,self.n_features)
        del windowed
        
        indx = np.random.randint(reshaped.shape[0],size=self.batch_size)
        return reshaped[indx]
