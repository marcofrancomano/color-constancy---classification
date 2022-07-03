# -*- coding: utf-8 -*-
"""ColorCostancyNet-V2

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1QsdSSU3RGA1oMy3yuVmV5kSqEizNiqsh
"""

import torch
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
model.eval()

from google.colab import drive
drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard
import tensorflow as tf
import datetime

# Clear any logs from previous runs
!rm -rf ./logs/

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

import torch.optim as optim

num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 1)
t=torch.ones([1], dtype=torch.float64)
t[0]=0.4853
t= t.to('cuda')
unb_criterion = torch.nn.BCEWithLogitsLoss(weight =t)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

from torchvision.transforms.transforms import Normalize
import numpy as np
import pandas as pd
import imageio 
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
import albumentations as A
import cv2
!pip install albumentations==0.4.6
from albumentations.pytorch.transforms import ToTensorV2


preprocess_train = A.Compose([
    A.Normalize(max_pixel_value=1.0),
    A.RandomCropNearBBox(0.15),
    A.HorizontalFlip(),
    ToTensorV2(),


])
preprocess_val = A.Compose([
    A.Normalize(max_pixel_value=1.0),
    A.RandomCropNearBBox(0),
    ToTensorV2(),


])



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
        image = image / (2**16-1)  # normalize between 0 and 1

        # Cropping box
        box_path = os.path.join('/content/drive/MyDrive/canopies-utilities', 'bboxes', f'{l[0]}.txt')
        bbox = np.loadtxt(box_path, delimiter=" ", dtype=np.float32)
        bbox = bbox[1:]
        # Convert to pascal format xmin ymin xmax ymax
        x_min = int((bbox[0] - bbox[2] / 2.0) * 1500)
        x_max = int((bbox[0] + bbox[2] / 2.0) * 1500)
        y_min = int((bbox[1] - bbox[3] / 2.0) * 2000)
        y_max = int((bbox[1] + bbox[3] / 2.0) * 2000)
        
        if self.transform is not None:
          image = self.transform(image=image, cropping_bbox=[x_min, y_min, x_max, y_max])["image"]

        else:
        #image = torch.as_tensor(image['image'])  # convert to tensor
          image = torch.as_tensor(image)
          image = image.permute((2, 0, 1))  # permute the image from HWC to CHW format
        image=transforms.Resize((512,512))(image)
        # print(image.dtype)
        # print(image.shape)
        label=np.float32(self.brix.iloc[idx,4])
        sample = {'image': image, 'label': label}
        # print("img", np.max(sample['image']))

        return sample

#brix_dataset = MyDataset(csv_file=open('/content/drive/MyDrive/datasets/brix_labels.csv', 'r'),
                                     #root_dir='drive/MyDrive/datasets',transform=None,train=True)
train_dataset = MyDataset(csv_file=open('/content/drive/MyDrive/datasets/brix_labels.csv', 'r'),
                                     root_dir='drive/MyDrive/datasets',transform=preprocess_train,train=True)
val_dataset = MyDataset(csv_file=open('/content/drive/MyDrive/datasets/brix_labels.csv', 'r'),
                                     root_dir='drive/MyDrive/datasets',transform=preprocess_val,train=False)
                           
sample = train_dataset[2]
input_image=sample['image']
print(sample['label'])
input_tensor=input_image
input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)
    print(output)
    print(input_batch.shape)
probabilities = torch.sigmoid(output[0])
print(input_batch)
print("Prob:", probabilities)

from torch.utils.data import DataLoader
from sklearn.utils import resample

#training_data,test_data=torch.utils.data.random_split(brix_dataset,[120,30])
#train_dataloader = DataLoader(training_data, batch_size=6, shuffle=True)
#test_dataloader = DataLoader(test_data, batch_size=6, shuffle=True)


train_dataloader = DataLoader(train_dataset, batch_size=6, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=1)

import sklearn.metrics

def evaluate(model,dataloader,criterion,device):
  model.eval()
  with torch.no_grad():
    val_loss=0.0
    preds_all=[]
    labels_all=[]
    for data in dataloader:
      data_= data['image']
      target_=data['label']
      #target_=torch.tensor(target_).unsqueeze(0)
      if torch.cuda.is_available():
        data_, target_ = data_.to(device), target_.to(device)
    
      outputs=model(data_)
      outputs=outputs.reshape(data_.size(0))
      pred=1 if torch.sigmoid(outputs).item()>0.5 else 0
      preds_all.append(pred)
      labels_all.append(int(target_.item()))
      loss=criterion(outputs,target_)
      val_loss+=loss.item()
 
    f1_score=sklearn.metrics.f1_score(labels_all,preds_all)
    print(f'Validation loss: {val_loss/len(val_dataset):.4f}')
    print(f'F1 score: {f1_score:.4f}')
    #print(f'Predictions: {preds_all}')
    #print(f'Labels: {labels_all}')
    with test_summary_writer.as_default():
      tf.summary.scalar('loss', val_loss/len(val_dataset), step=epoch)
      tf.summary.scalar('f1_score', f1_score, step=epoch)
  model.train()
  return val_loss

n_epochs = 15

train_loss = 0
dataset_size=len(train_dataset)
preds_all=[]
labels_all=[]

model.train()

for epoch in range(1, n_epochs+1):
  running_loss = 0.0
  correct = 0
  total=0
  print(f'Epoch {epoch}\n')
  
  for data in train_dataloader:
    data_= data['image']
    target_=data['label']
    #data_ = data_.unsqueeze(0) 
    #target_=torch.tensor(target_).unsqueeze(0)
    if torch.cuda.is_available():
      data_, target_ = data_.to('cuda'), target_.to('cuda')
    optimizer.zero_grad()
    
    target_=target_.reshape([6])
    #print(data_.shape)     
    outputs = model(data_)
    outputs=outputs.reshape(data_.size(0))
    probabilities=torch.sigmoid(outputs)
    pred=torch.zeros([outputs.shape[0]])
    #pred=pred.cuda()
    for i in range(probabilities.shape[0]):
      if probabilities[i]>0.5:
        pred[i]=1

    preds_all.extend(pred)
    labels_all.extend(target_.cpu())
    
    loss = unb_criterion(outputs, target_)
    loss.backward()
    optimizer.step()
    train_loss += loss.item()

  print(f'\ntrain-loss: {train_loss/dataset_size:.4f}')
  f1_score=sklearn.metrics.f1_score(labels_all,preds_all)
  print(f'F1 score: {f1_score:.4f}')

  with train_summary_writer.as_default():
    tf.summary.scalar('loss', train_loss/dataset_size, step=epoch)
    tf.summary.scalar('f1_score', f1_score, step=epoch)
  train_loss = 0
  # Evaluate model after each epoch
  evaluate(model,val_dataloader,criterion,'cuda')

# Commented out IPython magic to ensure Python compatibility.
#plot results
!kill 1081
# %tensorboard --logdir logs/gradient_tape

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
model.eval()
with torch.no_grad():
    for data in val_dataloader:
        images= data['image']
        labels=data['label']
        images, labels = images.cuda(), labels.cuda()
        
        
        # calculate outputs by running images through the network
        outputs = model(images)
        probabilities = torch.sigmoid(outputs)
        pred=torch.zeros([probabilities.shape[0]])
        print(probabilities)
        pred=pred.cuda()
        for i in range(probabilities.shape[0]):
          if(probabilities[i]>0.5):
            pred[i]=1

       
        total += labels.size(0)
        correct += (pred == labels).sum().item()
        print(correct)

print(f'Accuracy of the network on the 10 test images: {100 * correct // total} %')