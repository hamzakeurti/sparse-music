import torch
from torch.utils.data import Dataset, DataLoader
import os

import musicnet
import pickling


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
        elif dataset == 'cochleagram':
            self.dataset = CochleagramsDataset(root=directory)
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

    
class CochleagramsDataset(Dataset):

    def __init__(self,root):
        if not os.path.exists(root):
            raise FileNotFoundError
        self.root = root
        self.files_list = os.listdir(self.root) 
        if len(self.files_list) == 0:
            print(f'Dataset root folder {root} is empty')
            raise Exception
        self.extension = os.listdir(self.root)[0].split('.')[-1]
        

    def __len__(self):
        return len(os.listdir(self.root))
    
    def __getitem__(self,idx):
        idx  = idx % len(self.files_list)
        if torch.is_tensor(idx):
            idx = idx.tolist()
        f = self.files_list[idx]
        try:
            if self.extension == 'p':
                return pickling.load_tensor(os.path.join(self.root,f))
            if self.extension == 'pt':
                return torch.load(os.path.join(self.root,f)).numpy()
        except EOFError:
            os.remove(os.path.join(self.root,f))
            del self.files_list[idx]
            return self.__getitem__(idx)

    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass
