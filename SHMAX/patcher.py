
import numpy as np

class Patcher:
    def __init__(self,batch_size,filter_size,width_interval=None,height_interval=None):
        self.batch_size = batch_size
        self.width_interval = width_interval
        self.height_interval = height_interval
        self.filter_size = filter_size

    def extract_patches(self,input):
        # Patches will be selected from index as upper left corner
        if not self.height_interval:
            self.height_interval = (0,input.shape[-2] - self.filter_size + 1)
        if not self.width_interval:
            self.width_interval = (0,input.shape[-1] - self.filter_size + 1)
        height_index = np.random.randint(low=self.height_interval[0],high=self.height_interval[1], size=self.batch_size)
        width_index = np.random.randint(low=self.width_interval[0],high=self.width_interval[1], size=self.batch_size)
        batch_index = np.random.randint(input.shape[0],size=self.batch_size)

        patches = np.zeros((self.batch_size,input.shape[1],self.filter_size,self.filter_size))
        for i in range(self.batch_size):
            b = batch_index[i]
            h = height_index[i]
            w = width_index[i]
            f = self.filter_size
            patches[i] = input[b,:,h:h+f,w:w+f]

        return patches.reshape(self.batch_size,-1)