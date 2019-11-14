import torch
import torch.nn as nn

from layers import SLayer
import utils

BATCH_SIZE = 'batch_size'
POOL_SIZE = 'pool_size'
POOL_STRIDES = 'pool_strides'
FILTER_SIZE = 'filter_size'
N_LAYERS = 'n_layers'
CHANNELS = 'channels'
SPAMS_PARAM = 'spams_param'
N_SUBITERATIONS = 'n_subiterations'
INCREMENT_RATE = 'increment_rate'

class SHMAX:
    def __init__(self,n_layers,filter_size,pool_size,pool_strides,channels,batch_size,increment_rate,spams_param):
        self.n_layers = n_layers
        self.pool_strides = pool_strides
        self.pool_size = pool_size
        self.s_layers = []
        self.iterations = 0
        self.increment_rate = increment_rate
        for i in range(self.n_layers):
            self.s_layers.append(SLayer(input_channels=channels[i],output_channels=channels[i+1],filter_size=filter_size,batch_size=batch_size,param=spams_param.copy()))

    @classmethod
    def from_config(cls,config):
        n_layers = config[N_LAYERS]
        pool_strides = config[POOL_STRIDES]
        pool_size = config[POOL_SIZE]
        filter_size = config[FILTER_SIZE]
        channels = config[CHANNELS]
        spams_param = config[SPAMS_PARAM]
        batch_size = config[BATCH_SIZE]
        n_subiterations = config.get(N_SUBITERATIONS,1)
        increment_rate = config.get(INCREMENT_RATE,20*n_subiterations)
        return cls(n_layers,filter_size,pool_size,pool_strides,channels,batch_size,increment_rate,spams_param)

    def forward_train(self,input):
        tensors = [[] for layer in range(self.n_layers*2+1)]
        tensors[0] = input
        for layer in range(self.n_layers):
            if self.iterations < layer*self.increment_rate:
                continue
    #               ------- S_layer -------------------------------------
            print(f'training layer {layer}')
            self.s_layers[layer].train(tensors[2*layer].numpy()[:-1]) # Only train on 4/5
            
            tensors[2*layer+1] = self.s_layers[layer].forward_conv(tensors[2*layer])
            reconstruction = self.s_layers[layer].backward_conv(tensors[2*layer+1])

            errors = utils.evaluate_reconstruction(tensors[2*layer],reconstruction)


    #               ---------- C_layer ------------------------------------
            tensors[2*layer+2] = nn.MaxPool2d(kernel_size=self.pool_size[layer],stride=(self.pool_strides[layer],self.pool_strides[layer]))(tensors[2*layer+1])
            # DONE customize the stride 
                
            print(f'Layer {layer} Reconstruction L2 total error : {errors[0]} L2 test error : {errors[1]}')
        self.iterations += 1

    def forward(self,input):
        tensors = [[] for i in range(self.n_layers*2+1)]
        tensors[0] = input

        for layer in range(self.n_layers):
        #               ------- S_layer -------------------------------------
            print(f'Forwarding through layer {layer}')                
            tensors[2*layer+1] = self.s_layers[layer].forward_conv(tensors[2*layer])
        #               ---------- C_layer ------------------------------------
            tensors[2*layer+2] = nn.MaxPool2d(kernel_size=self.pool_size[layer],stride=(self.pool_strides[layer],self.pool_strides[layer]))(tensors[2*layer+1])
        return tensors