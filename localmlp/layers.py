import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import spams
import pickle
from multiprocessing import Pool

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from sparse_music.common import constants as const
from sparse_music.common import whitening


class BandBlock(nn.Module):
    def __init__(self, spams_param, filters_shape, dict_size, channels_in=1, device='cpu'):
        super(BandBlock, self).__init__()
        self.dict = None
        self.dict_tensor = None
        self.dict_size = dict_size
        self.filters_shape = filters_shape  # H,W
        self.spams_param = spams_param
        self.device = device
        self.spams_param["K"] = self.dict_size
        self.channels_out = dict_size
        self.channels_in = channels_in
        self.savable = ["dict", "dict_tensor", "dict_size",
                        "filters_shape", "spams_param", "device", "channels_out",
                        "channels_in","mean","eigen_vals","eigen_vecs","n_components"]
        self.mean = None
        self.eigen_vals = None
        self.eigen_vecs = None
        self.n_components = None

        
    def whiten_train(self,patches,n_components=False,standardize=False):
        # call spams train memory
        if len(patches.shape) > 2:
            patches = patches.reshape(patches.shape[0], -1)
        if not n_components:
            n_components = patches.shape[-1]//4
        patches,self.eigen_vecs,self.eigen_vals,self.mean = whitening.whiten_fit_transform(patches.T, n_components = n_components)
        self.n_components = patches.shape[0]
        patches = patches.T
        patches = np.asfortranarray(patches)
        self.dict = learn_dictionary(
            patches, self.spams_param, standardize_input=standardize)
        self.dict_tensor = torch.tensor(
            self.dict.T.reshape(self.channels_out, 1,self.n_components,1)).type(torch.FloatTensor).to(self.device)
     
        
    def train(self, patches, standardize=False):
        # call spams train memory
        if len(patches.shape) > 2:
            patches = patches.reshape(patches.shape[0], -1)
        patches = np.asfortranarray(patches)
        self.dict = learn_dictionary(
            patches, self.spams_param, standardize_input=standardize)
        self.dict_tensor = torch.tensor(
            self.dict.T.reshape((self.channels_out, self.channels_in)+tuple(self.filters_shape))).to(self.device)

    def forward(self, input):
        # do conv
        return F.conv2d(input=input, weight=self.dict_tensor)
    
    def whiten_forward(self,input):
        if (self.eigen_vecs is None) or (self.eigen_vals is None):
            print('no whitening parameters')
            return
        # First whiten
        input = input.numpy()
        out = []
        for j in range(input.shape[-1]-self.filters_shape[-1]):
            patches = input[:,:,:,j:j+self.filters_shape[-1]].reshape(input.shape[0],-1)
#             shape: [n_samples,n_features]
            patches = whitening.whiten_transform(patches.T,self.eigen_vecs,self.eigen_vals,self.mean,self.n_components)
            patches = patches.T
            out.append(patches)
        out = np.stack(out,axis = -1)
#         shape: [n_samples,n_components,t_samples]
        out = torch.FloatTensor(out)
        out = out.unsqueeze(1)
        out = F.conv2d(input=out, weight=self.dict_tensor)
        return out
        

    def save(self, filename):
        data = {}
        for attr in self.__dict__:
            if attr in self.savable:
                data[attr] = self.__dict__[attr]
        with open(filename, "wb") as f:
            pickle.dump(data, f)

    def extract_patches(self, input):
        if len(input.shape) <4:
            input = torch.unsqueeze(input, 1)
        for i in range(len(self.band_blocks)):
            # self.patches is a list of per band patches, for each band, it contains a list of batches of patches to be concatenated at training time
            self.patches[i].append(patch(input, patch_batch=self.patch_batch, filter_height=self.filters_shape[0],
                                         filter_width=self.filters_shape[1], height_interval=[self.band_indices[i], self.band_indices[i]+1]))



class LocalSHMAX(nn.Module):
    def __init__(self, filters_shape, dict_size, channels_in=1, patch_batch=50, band_indices=None, band_blocks=None, n_bands=None, spams_param=None, device='cpu'):
        super(LocalSHMAX, self).__init__()
        self.dict_size = dict_size
        self.filters_shape = filters_shape
        self.channels_in = channels_in
        self.patch_batch = patch_batch
        self.band_blocks = None
        self.spams_param = spams_param
        # Allow for self specified band locations, to overlap bands if wanted
        if band_indices:
            self.band_indices = band_indices
        elif band_blocks:
            n_bands = len(band_blocks)
            self.band_blocks = nn.ModuleList([list(band_blocks)])
            self.band_indices = [i*filters_shape[0] for i in range(n_bands)]
        elif n_bands:
            self.band_indices = [i*filters_shape[0] for i in range(n_bands)]

        self.n_bands = len(band_indices)
        self.device = device
        if not self.band_blocks:
            self.band_blocks = nn.ModuleList(
                [BandBlock(spams_param.copy(), filters_shape, dict_size, channels_in, device) for i in range(self.n_bands)])

        self.patches = [[] for i in range(self.n_bands)]
        self.savable = ["dict_size" ,"filters_shape","spams_param" ,"patch_batch" ,"band_indices" ,"n_bands" ,"device"]

    @classmethod
    def from_config(cls, config):
        filters_shape = config.get(const.FILTER_SHAPE, None)
        channels_in = config.get(const.CHANNELS_IN, 1)
        dict_size = config.get(const.DICT_SIZE, None)
        patch_batch = config.get(const.PATCH_BATCH, None)
        band_indices = config.get(const.BAND_INDICES, None)
        n_bands = config.get(const.N_BANDS, None)
        spams_param = config.get(const.SPAMS_PARAM, None)
        device = config.get(const.DEVICE, None)
        return cls(filters_shape, dict_size, channels_in=channels_in, patch_batch=patch_batch, band_indices=band_indices, n_bands=n_bands, spams_param=spams_param, device=device)


    @classmethod
    def from_saved(cls,directory):
        if directory.endswith('.p'):
            directory = directory[:-2]
        with open(f'{directory}model.p','rb') as f:
            data = pickle.load(f)
        filters_shape = data.get(const.FILTER_SHAPE)
        dict_size = data.get(const.DICT_SIZE)
        channels_in = data.get(const.CHANNELS_IN)
        patch_batch = data.get(const.PATCH_BATCH)
        band_indices = data.get(const.BAND_INDICES)
        spams_param = data.get(const.SPAMS_PARAM)
        device = data.get(const.DEVICE)
        model = cls(filters_shape, dict_size, channels_in=channels_in, patch_batch=patch_batch, band_indices=band_indices, spams_param=spams_param, device=device)
        for b in range(model.n_bands):
            with open(f'{directory}{b}.p','rb') as f:
                data = pickle.load(f)
                model.band_blocks[b].dict = data['dict']
                model.band_blocks[b].dict_tensor = data['dict_tensor'].type(torch.FloatTensor)
                model.band_blocks[b].mean = data['mean']
                model.band_blocks[b].eigen_vals = data['eigen_vals']
                model.band_blocks[b].eigen_vecs = data['eigen_vecs']
                model.band_blocks[b].n_components = data['n_components']
        return model

    def extract_patches(self, input):
        if len(input.shape) <4:
            input = torch.unsqueeze(input, 1)
        for i in range(len(self.band_blocks)):
            # self.patches is a list of per band patches, for each band, it contains a list of batches of patches to be concatenated at training time
            self.patches[i].append(patch(input, patch_batch=self.patch_batch, filter_height=self.filters_shape[0],
                                         filter_width=self.filters_shape[1], height_interval=[self.band_indices[i], self.band_indices[i]+1]))


    def extract_patches_multiprocessing(self,input):
        if len(input.shape) <4:
            input = torch.unsqueeze(input, 1)
        def patch_band(i):
            print(f'patching for band{i}/{self.n_bands}')
            self.patches[i].append(patch(input, patch_batch=self.patch_batch, filter_height=self.filters_shape[0],
                                    filter_width=self.filters_shape[1], height_interval=[self.band_indices[i], self.band_indices[i]+1]))
        p = Pool(4)
        p.map(patch_band,range(len(self.patches)))


    def train(self):
        # Patch here?
        if isinstance(self.patches[0],list):
            self.patches = [np.concatenate(p, axis=0) for p in self.patches]
        for i, band in enumerate(self.band_blocks):
            # Need to make sure we're using patched data
            print(f'Training band {i}/{self.n_bands}')
            band.train(self.patches[i])
        del self.patches
    
    def whiten_train(self,n_components=None):
        if isinstance(self.patches[0],list):
            self.patches = [np.concatenate(p, axis=0) for p in self.patches]
        for i, band in enumerate(self.band_blocks):
            # Need to make sure we're using patched data
            print(f'Training band {i}/{self.n_bands}')
            band.whiten_train(self.patches[i],n_components)
        del self.patches

    def train_multiprocessing(self):
        if isinstance(self.patches[0],list):
            self.patches = [np.concatenate(p, axis=0) for p in self.patches]
        def train_band(i):
            print(f'Training band {i}/{self.n_bands}')
            self.band_blocks[i].train(self.patches(i))
        p = Pool(4)
        p.map(train_band,range(len(self.patches)))
        del self.patches



    def forward(self, input):
        output = []
        if len(input.shape) <4:
            input = torch.unsqueeze(input, 1)
        for i, band in enumerate(self.band_blocks):
            output.append(band.forward(
                input[:, :, self.band_indices[i]:self.band_indices[i]+self.filters_shape[0], :].type(torch.FloatTensor)))
        # conncatenate along the freq dimension
        output = torch.cat(output, dim=-2)
        return output

    def whiten_forward(self,input):
        output = []
        if len(input.shape) <4:
            input = torch.unsqueeze(input, 1)
        for i, band in enumerate(self.band_blocks):
            output.append(band.whiten_forward(
                input[:, :, self.band_indices[i]:self.band_indices[i]+self.filters_shape[0], :].type(torch.FloatTensor)))
        # conncatenate along the freq dimension
        output = torch.cat(output, dim=-2)
        return output
        
            
    def save(self, directory):
        # remove extension if present
        if directory.endswith('.p'):
            directory = directory[:-2]
        
        # create directory if not existing
        if not os.path.exists(os.path.dirname(directory)):
            os.mkdir(os.path.dirname(directory))
        
        # save model's attributes
        data = {}
        for attr in self.__dict__:
            if attr in self.savable:
                data[attr] = self.__dict__[attr]
        with open(f'{directory}model.p', "wb") as f:
            pickle.dump(data, f)
        
        # save composing band's dictionaries
        for b in range(self.n_bands):
            self.band_blocks[b].save(f'{directory}{b}.p')
        data = {}


def patch(input, patch_batch, filter_height, filter_width, width_interval=None, height_interval=None):
    if not height_interval:
        height_interval = (0, input.shape[-2] - filter_height + 1)
    if not width_interval:
        width_interval = (0, input.shape[-1] - filter_width + 1)

    height_index = np.random.randint(
        low=height_interval[0], high=height_interval[1], size=patch_batch)
    width_index = np.random.randint(
        low=width_interval[0], high=width_interval[1], size=patch_batch)
    batch_index = np.random.randint(input.shape[0], size=patch_batch)
    patches = np.zeros(
        (patch_batch, input.shape[1], filter_height, filter_width))
    for i in range(patch_batch):
        b = batch_index[i]
        h = height_index[i]
        w = width_index[i]
        fh = filter_height
        fw = filter_width
        patches[i] = input[b, :, h:h+fh, w:w+fw]

    return patches


def learn_dictionary(patches, spams_param, model=None, standardize_input=True):
    X = patches.T
    if standardize_input:
        X = standardize(X)
    X = np.asfortranarray(X, dtype=float)
    D = spams.trainDL(X, model=model, **spams_param)
    spams_param['D'] = D
    return D


def standardize(X):
    X = X - np.tile(np.mean(X, 0), (X.shape[0], 1))
    X = X / np.tile(np.sqrt((X * X).sum(axis=0)), (X.shape[0], 1))
    return X
