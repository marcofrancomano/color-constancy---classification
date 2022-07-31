import torch
from torchvision.transforms.transforms import Normalize
import numpy as np
import pandas as pd
import imageio.v2 as imageio
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os



class MyDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform,train,dataset="tiff_wb"):
        split = "train" if train else "val"
        self.brix= pd.read_csv(csv_file,sep=",")
        self.root_dir = os.path.join(root_dir, dataset, split)
        self.transform = transform

        self.brix = self.brix[self.brix["split"] == split]
        self.brix = self.brix.reset_index()
        #self.img_labels = csv[["file_id", "brix", "label"]]

    def __len__(self):
        return len(self.brix)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
  
        l=self.brix.iloc[idx,2].split(".")
        name=l[0]+".tiff"
        img_name = os.path.join(self.root_dir, name)

        image = imageio.imread(img_name)
        # print("img", np.max(image))
        image = np.asarray(image, dtype=np.float32)  # convert imageio to ndarray
        #image = image / (2**16-1)  # normalize between 0 and 1

        
        image = torch.as_tensor(image)
        image = image.permute((2, 0, 1))  # permute the image from HWC to CHW format
        #print(image.dtype)
        # print(image.shape)
        label=np.float32(self.brix.iloc[idx,4])
        sample = {'image': image, 'label': label}
        # print("img", np.max(sample['image']))

        return sample


train_dataset = MyDataset(csv_file=open('/Users/Marco/Desktop/Canopies-AI_Project/brix_labels.csv', 'r'),
                                     root_dir='/Users/Marco/Desktop/Canopies-AI_Project',transform=None,train=True)
val_dataset = MyDataset(csv_file=open('/Users/Marco/Desktop/Canopies-AI_Project/brix_labels.csv', 'r'),
                                     root_dir='/Users/Marco/Desktop/Canopies-AI_Project',transform=None,train=False)
                           


dataset_size=MyDataset.__len__(train_dataset)
avg=np.zeros(3)
for batch_idx in range(dataset_size):
    sample=train_dataset[batch_idx]
    data=sample['image']
    for i in range(2000):
          avg[0]+=data[0][i].sum()
          avg[1]+=data[1][i].sum()
          avg[2]+=data[2][i].sum()
avg[0]=avg[0]/(dataset_size*(2000*1500))
avg[1]=avg[1]/(dataset_size*(2000*1500))
avg[2]=avg[2]/(dataset_size*(2000*1500))
print(avg)
stdev=np.zeros(3)
avgr=np.zeros(1500)
avgg=np.zeros(1500)
avgb=np.zeros(1500)
for i in range(1500):
    avgr[i]=avg[0]
    avgg[i]=avg[1]
    avgb[i]=avg[2]
    
for batch_idx in range(dataset_size):
    sample=train_dataset[batch_idx]
    data=sample['image']
    #print(data[0][0].shape)
    for i in range(2000):
        stdev[0]+=(np.power(data[0][i]-avgr, 2)).sum()
        stdev[1]+=(np.power(data[1][i]-avgg, 2)).sum()
        stdev[2]+=(np.power(data[2][i]-avgb, 2)).sum()
stdev[0]=np.sqrt(stdev[0]/(dataset_size*(2000*1500)))
stdev[1]=np.sqrt(stdev[1]/(dataset_size*(2000*1500)))
stdev[2]=np.sqrt(stdev[2]/(dataset_size*(2000*1500)))

print(stdev)
            
            
        
    

