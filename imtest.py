
import numpy as np
import pandas as pd
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os

df = pd.read_csv("brix_labels_07_ott_2021-Foglio1.csv", sep=",")
#print(df.head())
raw_path = "./dataset_examples/wb/IMG_20211007_105733020_wb.tiff"
rgb = np.array(imageio.imread(raw_path))
assert rgb.dtype == np.uint16, f'type {rgb.dtype} of file {raw_path} is not np.uint16'
#print(rgb.shape)


class MyDataset(Dataset):


    def __init__(self, csv_file, root_dir, transform=None):

        self.brix= pd.read_csv(csv_file,sep=",")
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.brix)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        l=self.brix.iloc[idx,5].split(".")
        name=l[0]+"_wb.tiff"
        img_name = os.path.join(self.root_dir,
                                name)


        image = imageio.imread(img_name)
        measures = self.brix.iloc[idx, 1:4]
        measures = np.array([measures])
        avg=np.mean(measures)
        label=0
        if avg>16:
            label=1
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

brix_dataset = MyDataset(csv_file='brix_labels_07_ott_2021-Foglio1.csv',
                                    root_dir='dataset_examples_2/wb')




sample = brix_dataset[2]

print(sample['image'].shape, sample['label'])
print(sample)




