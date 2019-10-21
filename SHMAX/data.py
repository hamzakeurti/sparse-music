import torch

import musicnet



# CONFIG KEYS------
BATCH_SIZE = 'batch_size'
DIRECTORY = 'directory'
DATASET = 'dataset'

# MUSICNET----------
MUSICNET_DIRECTORY = '/data/valentin/music-learning/musicnet'
DOWNLOAD = 'download'
JITTER = 'jitter'
PITCH_SHIFT = 'pitch_shift', 
WINDOW = 'window'
SR = 'sampling_rate'
#--------------------

class DataIterator:
    def __init__(self,dataset,batch_size,directory='',**kwargs):
        if dataset == 'musicnet':
            if not directory:
                directory = MUSICNET_DIRECTORY
            self.dataset = musicnet.MusicNet(root=directory, train=True, window=kwargs[WINDOW], download=kwargs.get(DOWNLOAD,True),pitch_shift=kwargs.get(PITCH_SHIFT,0),jitter=kwargs.get(JITTER,0))
            self.loader = torch.utils.data.DataLoader(dataset=self.dataset,batch_size=batch_size)
        else:
            raise NameError(f'dataset {dataset} not handled')


    @classmethod
    def from_config(cls,config):
        config = config.copy()
        dataset = config.pop(DATASET,'')
        batch_size = config.pop(BATCH_SIZE,20)
        directory = config.pop(DIRECTORY,'')
        return cls(dataset,batch_size,directory = directory,**config)

    