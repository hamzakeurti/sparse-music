import torch
from torch.utils.data import Dataset, DataLoader
import os

from sparse_music.common import musicnet, pickling

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
# --------------------

# MUSICNET COCHLEAGRAMS-----
MUSICNET_COCH_DIRECTORY = '/data/hamza/datasets/musicnet/cochleagrams/folder1'

# SPEECH COCHLEAGRAMS-------
SPEECH_DIRECTORY = '/home/valentin/data/twopaths/pt/fma_large_cgrams'


class DataIterator:
    def __init__(self, dataset, batch_size, directory='', shuffle=True, **kwargs):
        if dataset == 'musicnet':
            if not directory:
                directory = MUSICNET_DIRECTORY
            self.dataset = musicnet.MusicNet(root=directory, train=True, window=kwargs[WINDOW], download=kwargs.get(
                DOWNLOAD, True), pitch_shift=kwargs.get(PITCH_SHIFT, 0), jitter=kwargs.get(JITTER, 0))
        elif dataset == 'cochleagram':
            transform = None
            if kwargs.get('normalize', 0):
                transform = Normalize()
            self.dataset = CochleagramsDataset(
                root=directory, transform=transform)
        elif dataset == 'frequencies':
            transform = None
            if kwargs.get('normalize', 0):
                transform = Normalize()
            self.dataset = FrequencyCochleagramsDataset(
                root=directory, transform=transform)
        else:
            raise NameError(f'dataset {dataset} not handled')
        self.loader = torch.utils.data.DataLoader(
            dataset=self.dataset, batch_size=batch_size, shuffle=shuffle)

    @classmethod
    def from_config(cls, config):
        config = config.copy()
        dataset = config.pop(DATASET, '')
        batch_size = config.pop(BATCH_SIZE, 20)
        directory = config.pop(DIRECTORY, '')
        return cls(dataset, batch_size, directory=directory, **config)


class CochleagramsDataset(Dataset):

    def __init__(self, root, transform=None):
        if not os.path.exists(root):
            raise FileNotFoundError
        self.root = root
        self.files_list = os.listdir(self.root)
        if len(self.files_list) == 0:
            print(f'Dataset root folder {root} is empty')
            raise Exception
        self.extension = os.listdir(self.root)[0].split('.')[-1]
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.root))

    def __getitem__(self, idx):
        idx = idx % len(self.files_list)
        if torch.is_tensor(idx):
            idx = idx.tolist()
        f = self.files_list[idx]
        try:
            if self.extension == 'p':
                sample = pickling.load_tensor(os.path.join(self.root, f))
            if self.extension == 'pt':
                sample = torch.load(os.path.join(self.root, f)).numpy()
            if self.transform:
                sample = self.transform(sample)
            return sample
        except EOFError:
            os.remove(os.path.join(self.root, f))
            del self.files_list[idx]
            return self.__getitem__(idx)

    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass


class FrequencyCochleagramsDataset(Dataset):

    def __init__(self, root, transform=None):
        if not os.path.exists(root):
            raise FileNotFoundError
        self.root = root
        self.extension = os.listdir(self.root)[0].split('.')[-1]
        self.files_list = os.listdir(self.root)
        for f in self.files_list:
            if not f.endswith(f'.{self.extension}'):
                self.files_list.remove(f)

        self.files_list = sorted(
            self.files_list, key=lambda filename: int(filename[:-2]))
        if len(self.files_list) == 0:
            print(f'Dataset root folder {root} is empty')
            raise Exception
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.root))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        f = self.files_list[idx]
        freq = int(f.split('.')[0])
        try:
            if self.extension == 'p':
                sample = pickling.load_tensor(os.path.join(self.root, f))
            if self.extension == 'pt':
                sample = torch.load(os.path.join(self.root, f)).numpy()
            if self.transform:
                sample = self.transform(sample)
            return sample, freq
        except EOFError:
            print(f'file {f} is empty')
            os.remove(os.path.join(self.root, f))
            del self.files_list[idx]
            print(f'file {f} deleted')

            return self.__getitem__(idx)

    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass


class Normalize(object):
    """Normalize the image in a sample between 0 and 1."""

    def __init__(self):
        pass

    def __call__(self, image):
        min, max = image.min(), image.max()
        image = (image - min) / (max-min)
        return image
