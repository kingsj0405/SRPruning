import torch.utils.data as data
import torch
import h5py
import numpy as np
import random
from pathlib import Path
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHdf5, self).__init__()
        hf = h5py.File(file_path)
        self.data = hf.get('data')
        self.target = hf.get('label')

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index,:,:,:]).float(), torch.from_numpy(self.target[index,:,:,:]).float()
        
    def __len__(self):
        return self.data.shape[0]

class DatasetFromDIV2K(data.Dataset):
    def __init__(self, train_dirpath, label_dirpath, train_transform=None, label_transform=None):
        super(DatasetFromDIV2K, self).__init__()
        self.train_list = sorted(list(Path(train_dirpath).glob('*.png')))
        self.label_list = sorted(list(Path(label_dirpath).glob('*.png')))
        self.train_transform = train_transform
        self.label_transform = label_transform
        self.len = len(self.train_list)
        assert(self.len == len(self.label_list))
    
    def __getitem__(self, index):
        # Get random seed
        seed = np.random.randint(2147483647) # make a seed with numpy generator 
        # Transform train
        train_img = Image.open(self.train_list[index])
        if self.train_transform is not None:
            random.seed(seed) # apply this seed to img tranfsorms
            train_img = self.train_transform(train_img)
        # Transform test
        label_img = Image.open(self.label_list[index])
        if self.label_transform is not None:
            random.seed(seed) # apply this seed to target tranfsorms
            label_img = self.label_transform(label_img)
        # Return
        # print(f"Read {self.train_list[index]} and {self.label_list[index]}")
        return train_img, label_img
    
    def __len__(self):
        return self.len