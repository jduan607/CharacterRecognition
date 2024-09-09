# Multilayer Perceptron using 45-class and 52-class

import os
#os.environ['CUDA_LAUNCH_BLOCKING']="1"
import time
import torch
import base64
import pandas as pd
import numpy as np
import tensorflow as tf
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from IPython.display import HTML
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torchvision import models
import matplotlib.pyplot as plt
from PIL import Image
%matplotlib inline

####################
# Settings
####################
# Device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyperparameters
RANDOM_SEED = 479
LEARNING_RATE = 0.1
NUM_EPOCHS = 30
BATCH_SIZE = 256

# Architecture
NUM_FEATURES = 64*64
NUM_CLASSES = 45 # 45 or 52

####################
# I. Custom dataset class
####################
# Grayscale images
class MyDataset_Grayscale(Dataset):

    def __init__(self, csv_path, img_dir, transform=None):
        df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.img_names = df['File Name']
        self.y = df['Class Label']
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir, self.img_names[index]))
          
        if self.transform is not None:
            img = self.transform(img)
        
        label = self.y[index]
        return img, label

    def __len__(self):
        return self.y.shape[0]
    
# Binary images
class MyDataset_Binary(Dataset):

    def __init__(self, csv_path, img_dir, transform=None):
    
        df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.img_names = df['File Name']
        self.y = df['Class Label']
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir,
                                      self.img_names[index]))
        
        img=img.convert("L")
        threshold = 128
        # if pixel value smaller than threshold, return 0 . Otherwise return 1.
        filter_func = lambda x: 0 if x < threshold else 1
        img=img.point(filter_func, "1")
          
        if self.transform is not None:
            img = self.transform(img)
        
        label = self.y[index]
        return img, label

    def __len__(self):
        return self.y.shape[0]
    
# RGB images
class MyDataset_RGB(Dataset):

    def __init__(self, csv_path, img_dir, transform=None):
        df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.img_names = df['File Name']
        self.y = df['Class Label']
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir, self.img_names[index]))
        
        # make 1 color channel to 3 
        ary = np.array(img)
        if len(ary.shape) < 3:
          ary = np.stack((ary,)*3,axis=-1)
          img = Image.fromarray(ary, 'RGB')
                    
        if self.transform is not None:
            img = self.transform(img)
        
        label = self.y[index]
        return img, label

    def __len__(self):
        return self.y.shape[0]
    
####################
# II. Data Augmentation
####################      
# Grayscale images
#Without Cropping
train_transforms_Grayscale = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomAffine(degrees=(-20, 20), translate=(0.15, 0.15),
                                        resample=Image.BILINEAR),
    transforms.Resize(size=(64, 64)),
    transforms.ToTensor()
])
valid_transforms_Grayscale = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(size=(64, 64)),
    transforms.ToTensor()
])
test_transforms_Grayscale = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(size=(64, 64)),
    transforms.ToTensor()
])


#With Cropping
train_transforms_GrayscaleCrop = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomAffine(degrees=(-20, 20), translate=(0.15, 0.15),
                                        resample=Image.BILINEAR),
    transforms.Resize(size=(76, 76)),
    transforms.RandomCrop(size=64),
    transforms.ToTensor()
])
valid_transforms_GrayscaleCrop = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(size=(76, 76)),
    transforms.CenterCrop(size=64),
    transforms.ToTensor()
])
test_transforms_GrayscaleCrop = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(size=(76, 76)),
    transforms.CenterCrop(size=64),
    transforms.ToTensor()
])

# Binary images
#Without cropping
train_transforms_Binary = transforms.Compose([
    transforms.RandomAffine(degrees=(-20, 20), translate=(0.15, 0.15),
                                      resample=Image.BILINEAR),
    transforms.Resize(size=(64, 64)),
    transforms.ToTensor()
])
valid_transforms_Binary = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.ToTensor()
])
test_transforms_Binary = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.ToTensor()
])

#With cropping
train_transforms_BinaryCrop = transforms.Compose([
    transforms.RandomAffine(degrees=(-20, 20), translate=(0.15, 0.15),
                                      resample=Image.BILINEAR),
    transforms.Resize(size=(76, 76)),
    transforms.RandomCrop(size=64),
    transforms.ToTensor()
])
valid_transforms_BinaryCrop = transforms.Compose([
    transforms.Resize(size=(76, 76)),
    transforms.CenterCrop(size=64),
    transforms.ToTensor()
])
test_transforms_BinaryCrop = transforms.Compose([
    transforms.Resize(size=(76, 76)),
    transforms.CenterCrop(size=64),
    transforms.ToTensor()
])

# RGB images
#Without cropping 
train_transforms_RGB = transforms.Compose([
    transforms.RandomAffine(degrees=(-20, 20), translate=(0.15, 0.15),
                                      resample=Image.BILINEAR),
    transforms.Resize(size=(64, 64)),
    transforms.ToTensor()
])

valid_transforms_RGB = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.ToTensor()
])

test_transforms_RGB = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.ToTensor()
])

#With cropping
train_transforms_RGBCrop = transforms.Compose([
    transforms.RandomAffine(degrees=(-20, 20), translate=(0.15, 0.15),
                                      resample=Image.BILINEAR),
    transforms.Resize(size=(76, 76)),
    transforms.RandomCrop(size=64),
    transforms.ToTensor()
])

valid_transforms_RGBCrop = transforms.Compose([
    transforms.Resize(size=(76, 76)),
    transforms.CenterCrop(size=64),
    transforms.ToTensor()
])

test_transforms_RGBCrop = transforms.Compose([
    transforms.Resize(size=(76, 76)),
    transforms.CenterCrop(size=64),
    transforms.ToTensor()
])

####################
# III. Custom Data Loader
####################  
# Grayscale images
# No cropping
train_dataset_Grayscale = MyDataset_Grayscale(csv_path='../input/stat479csv/TrainingCorrect_combinedULcases.csv',
                                              img_dir='../input/stat479project/data',
                                              transform=train_transforms_Grayscale)
train_loader_Grayscale = DataLoader(dataset=train_dataset_Grayscale,
                                    batch_size=BATCH_SIZE,
                                    shuffle=True, num_workers=4)

valid_dataset_Grayscale = MyDataset_Grayscale(csv_path='../input/stat479csv/ValidationCorrect_combinedULcases.csv',
                                              img_dir='../input/stat479project/data',
                                              transform=valid_transforms_Grayscale)
valid_loader_Grayscale = DataLoader(dataset=valid_dataset_Grayscale,
                                    batch_size=BATCH_SIZE,
                                    shuffle=False, num_workers=4)

test_dataset_Grayscale = MyDataset_Grayscale(csv_path='../input/stat479csv/TestingCorrect_combinedULcases.csv',
                                             img_dir='../input/stat479project/data',
                                             transform=test_transforms_Grayscale)
test_loader_Grayscale = DataLoader(dataset=test_dataset_Grayscale,
                                   batch_size=BATCH_SIZE,
                                   shuffle=False, num_workers=4)

#Cropping
train_dataset_GrayscaleCrop = MyDataset_Grayscale(csv_path='../input/stat479csv/TrainingCorrect_combinedULcases.csv',
                                              img_dir='../input/stat479project/data',
                                              transform=train_transforms_GrayscaleCrop)
train_loader_GrayscaleCrop = DataLoader(dataset=train_dataset_GrayscaleCrop,
                                    batch_size=BATCH_SIZE,
                                    shuffle=True, num_workers=4)

valid_dataset_GrayscaleCrop = MyDataset_Grayscale(csv_path='../input/stat479csv/ValidationCorrect_combinedULcases.csv',
                                              img_dir='../input/stat479project/data',
                                              transform=valid_transforms_GrayscaleCrop)
valid_loader_GrayscaleCrop = DataLoader(dataset=valid_dataset_GrayscaleCrop,
                                    batch_size=BATCH_SIZE,
                                    shuffle=False, num_workers=4)

test_dataset_GrayscaleCrop = MyDataset_Grayscale(csv_path='../input/stat479csv/TestingCorrect_combinedULcases.csv',
                                             img_dir='../input/stat479project/data',
                                             transform=test_transforms_GrayscaleCrop)
test_loader_GrayscaleCrop = DataLoader(dataset=test_dataset_GrayscaleCrop,
                                   batch_size=BATCH_SIZE,
                                   shuffle=False, num_workers=4)

# Binary images
# No cropping
train_dataset_Binary = MyDataset_Binary(csv_path='../input/stat479csv/TrainingCorrect_combinedULcases.csv',
                          img_dir='../input/stat479project/data',
                          transform=train_transforms_Binary)
train_loader_Binary = DataLoader(dataset=train_dataset_Binary,
                          batch_size=256,
                          shuffle=True, # want to shuffle the dataset
                          num_workers=4) # number processes/CPUs to use

valid_dataset_Binary = MyDataset_Binary(csv_path='../input/stat479csv/ValidationCorrect_combinedULcases.csv',
                          img_dir='../input/stat479project/data',
                          transform=valid_transforms_Binary)
valid_loader_Binary = DataLoader(dataset=valid_dataset_Binary,
                          batch_size=256,
                          shuffle=False, 
                          num_workers=4) # number processes/CPUs to use

test_dataset_Binary = MyDataset_Binary(csv_path='../input/stat479csv/TestingCorrect_combinedULcases.csv',
                          img_dir='../input/stat479project/data',
                          transform=test_transforms_Binary)
test_loader_Binary = DataLoader(dataset=test_dataset_Binary,
                          batch_size=256,
                          shuffle=False, 
                          num_workers=4) # number processes/CPUs to use

# Cropping
train_dataset_BinaryCrop = MyDataset_Binary(csv_path='../input/stat479csv/TrainingCorrect_combinedULcases.csv',
                          img_dir='../input/stat479project/data',
                          transform=train_transforms_BinaryCrop)
train_loader_BinaryCrop = DataLoader(dataset=train_dataset_BinaryCrop,
                          batch_size=256,
                          shuffle=True, # want to shuffle the dataset
                          num_workers=4) # number processes/CPUs to use

valid_dataset_BinaryCrop = MyDataset_Binary(csv_path='../input/stat479csv/ValidationCorrect_combinedULcases.csv',
                          img_dir='../input/stat479project/data',
                          transform=valid_transforms_BinaryCrop)
valid_loader_BinaryCrop = DataLoader(dataset=valid_dataset_BinaryCrop,
                          batch_size=256,
                          shuffle=False, 
                          num_workers=4) # number processes/CPUs to use

test_dataset_BinaryCrop = MyDataset_Binary(csv_path='../input/stat479csv/TestingCorrect_combinedULcases.csv',
                          img_dir='../input/stat479project/data',
                          transform=test_transforms_BinaryCrop)
test_loader_BinaryCrop = DataLoader(dataset=test_dataset_BinaryCrop,
                          batch_size=256,
                          shuffle=False, 
                          num_workers=4) # number processes/CPUs to use

# RGB images
# No cropping
train_dataset_RGB = MyDataset_RGB(csv_path='../input/stat479csv/TrainingCorrect_combinedULcases.csv',
                                  img_dir='../input/stat479project/data',
                                  transform=train_transforms_RGB)
train_loader_RGB = DataLoader(dataset=train_dataset_RGB,
                              batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=4)

valid_dataset_RGB = MyDataset_RGB(csv_path='../input/stat479csv/ValidationCorrect_combinedULcases.csv',
                                  img_dir='../input/stat479project/data',
                                  transform=valid_transforms_RGB)
valid_loader_RGB = DataLoader(dataset=valid_dataset_RGB,
                              batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=4)

test_dataset_RGB = MyDataset_RGB(csv_path='../input/stat479csv/TestingCorrect_combinedULcases.csv',
                                 img_dir='../input/stat479project/data',
                                 transform=test_transforms_RGB)
test_loader_RGB = DataLoader(dataset=test_dataset_RGB,
                              batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=4)

#Cropping
train_dataset_RGBCrop = MyDataset_RGB(csv_path='../input/stat479csv/TrainingCorrect_combinedULcases.csv',
                                  img_dir='../input/stat479project/data',
                                  transform=train_transforms_RGBCrop)
train_loader_RGBCrop = DataLoader(dataset=train_dataset_RGBCrop,
                              batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=4)

valid_dataset_RGBCrop = MyDataset_RGB(csv_path='../input/stat479csv/ValidationCorrect_combinedULcases.csv',
                                  img_dir='../input/stat479project/data',
                                  transform=valid_transforms_RGBCrop)
valid_loader_RGBCrop = DataLoader(dataset=valid_dataset_RGBCrop,
                              batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=4)

test_dataset_RGBCrop = MyDataset_RGB(csv_path='../input/stat479csv/TestingCorrect_combinedULcases.csv',
                                 img_dir='../input/stat479project/data',
                                 transform=test_transforms_RGBCrop)
test_loader_RGBCrop = DataLoader(dataset=test_dataset_RGBCrop,
                              batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=4)

####################
# IV. Model
#################### 
def compute_epoch_loss_MLP(model, data_loader):
    curr_loss, num_examples = 0., 0
    
    with torch.no_grad():
        for features, targets in data_loader:
            if GRAYSCALE:
                features = features.view(-1,NUM_FEATURES).to(DEVICE)
            else:
                features = features.view(-1,NUM_FEATURES*3).to(DEVICE)
            targets = targets.to(DEVICE)
            logits, probas = model.forward(features)
            loss = F.cross_entropy(logits, targets, reduction='sum')
            num_examples += targets.size(0)
            curr_loss += loss

        curr_loss = curr_loss / num_examples
        return curr_loss
    
    
def compute_accuracy_MLP(model, data_loader, detail):
    correct_pred, num_examples = 0, 0
    labels, predictions, result = [], [], []
    
    with torch.no_grad():
        for features, targets in data_loader:
            if GRAYSCALE:
                features = features.view(-1,NUM_FEATURES).to(DEVICE)
            else:
                features = features.view(-1,NUM_FEATURES*3).to(DEVICE)
            targets = targets.to(DEVICE)
            logits, probas = model.forward(features)
            predicted_labels = torch.argmax(probas, 1)
            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum()
            if detail:
                targets = targets.cpu().numpy()
                predicted_labels = predicted_labels.cpu().numpy()
                labels = np.concatenate((labels, targets), axis=0)
                predictions = np.concatenate((predictions, predicted_labels), axis=0)
        if detail:
            return correct_pred.float()/num_examples * 100, labels, predictions
        else:
            return correct_pred.float()/num_examples * 100
        
def train_MLP(model, train_loader, valid_loader, test_loader):
    minibatch_cost, epoch_cost = [], []
    start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        model.train()
    
        for batch_idx, (features, targets) in enumerate(train_loader):
            if GRAYSCALE:
                features = features.view(-1,NUM_FEATURES).to(DEVICE)
            else:
                features = features.view(-1,NUM_FEATURES*3).to(DEVICE)
            targets = targets.to(DEVICE)
      
            logits, probas = model.forward(features)
            cost = F.cross_entropy(logits, targets)
            optimizer.zero_grad()
      
            cost.backward()
            minibatch_cost.append(cost)
            optimizer.step()
       
            if not batch_idx % 50:
                print('Epoch: %03d/%03d | Batch %03d/%03d | Cost: %.4f'
                      %(epoch+1, NUM_EPOCHS, batch_idx, len(train_loader), cost))
        model.eval()
        with torch.set_grad_enabled(False):
            if not (epoch+1) % 10:
                cost = compute_epoch_loss_MLP(model, train_loader)
                epoch_cost.append(cost)
                train_accuracy = compute_accuracy_MLP(model, train_loader, detail=False)
                valid_accuracy = compute_accuracy_MLP(model, valid_loader, detail=False)
                print('Epoch: %03d/%03d | Train Cost: %.4f' % (epoch+1, NUM_EPOCHS, cost))
                print('Train Accuracy: %.3f%% | Validation Accuracy: %.3f%%' % (train_accuracy, valid_accuracy))
   
        print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
    
    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
    with torch.set_grad_enabled(False): # save memory during inference
        test_accuracy, labels, predictions = compute_accuracy_MLP(model, test_loader, detail=True)
        print('Test accuracy: %.2f%%' % (test_accuracy))
    
    return minibatch_cost, epoch_cost, labels, predictions

####################
# MLP taking RGB images as input
####################
class MLP3(nn.Module):
  def __init__(self, num_features, drop_proba, 
               num_hidden1, num_hidden2, num_classes):
    super(MLP3, self).__init__()
    
    self.network = nn.Sequential(
        nn.Linear(num_features, num_hidden1),
        nn.BatchNorm1d(num_hidden1), 
        nn.ReLU(),
        nn.Dropout(drop_proba),
        nn.Linear(num_hidden1, num_hidden2),
        nn.BatchNorm1d(num_hidden2),
        nn.ReLU(),
        nn.Dropout(drop_proba),
        nn.Linear(num_hidden2, num_classes)
    )  
   
  def forward(self, x):
    logits = self.network(x)
    probas = F.softmax(logits, dim=1)
    return logits, probas

#no cropping, no momentum
torch.manual_seed(RANDOM_SEED)
model3 = MLP3(num_features=NUM_FEATURES*3, 
              drop_proba=0.2,
              num_hidden1=3000,
              num_hidden2=1500,
              num_classes=NUM_CLASSES)

model3 = model3.to(DEVICE)

optimizer = torch.optim.SGD(model3.parameters(), lr=LEARNING_RATE)

GRAYSCALE = False

minibatch_cost3, epoch_cost3, labels3, predictions3 = train_MLP(model3, train_loader_RGB, valid_loader_RGB, test_loader_RGB)

plt.plot(range(len(minibatch_cost3)), minibatch_cost3)
plt.ylabel('Cross Entropy')
plt.xlabel('Minibatch')
plt.show()

#No cropping, momentum
torch.manual_seed(RANDOM_SEED)
model3 = MLP3(num_features=NUM_FEATURES*3, 
              drop_proba=0.2,
              num_hidden1=3000,
              num_hidden2=1500,
              num_classes=NUM_CLASSES)

model3 = model3.to(DEVICE)

optimizer = torch.optim.SGD(model3.parameters(), lr=LEARNING_RATE, momentum = 0.9)

GRAYSCALE = False

minibatch_cost3, epoch_cost3, labels3, predictions3 = train_MLP(model3, train_loader_RGB, valid_loader_RGB, test_loader_RGB)

plt.plot(range(len(minibatch_cost3)), minibatch_cost3)
plt.ylabel('Cross Entropy')
plt.xlabel('Minibatch')
plt.show()

#cropping, momentum
torch.manual_seed(RANDOM_SEED)
model3 = MLP3(num_features=NUM_FEATURES*3, 
              drop_proba=0.2,
              num_hidden1=3000,
              num_hidden2=1500,
              num_classes=NUM_CLASSES)

model3 = model3.to(DEVICE)

optimizer = torch.optim.SGD(model3.parameters(), lr=LEARNING_RATE, momentum = 0.9)

GRAYSCALE = False

minibatch_cost3, epoch_cost3, labels3, predictions3 = train_MLP(model3, train_loader_RGBCrop, valid_loader_RGBCrop, test_loader_RGBCrop)

plt.plot(range(len(minibatch_cost3)), minibatch_cost3)
plt.ylabel('Cross Entropy')
plt.xlabel('Minibatch')
plt.show()

####################
# MLP taking grayscale images as input
####################

class MLP1(nn.Module):
  def __init__(self, num_features, drop_proba, 
               num_hidden1, num_hidden2, num_classes):
    super(MLP1, self).__init__()
    
    self.network = nn.Sequential(
        nn.Linear(num_features, num_hidden1),
        nn.BatchNorm1d(num_hidden1), 
        nn.ReLU(),
        nn.Dropout(drop_proba),
        nn.Linear(num_hidden1, num_hidden2),
        nn.BatchNorm1d(num_hidden2),
        nn.ReLU(),
        nn.Dropout(drop_proba),
        nn.Linear(num_hidden2, num_classes)
    )  
   
  def forward(self, x):
    logits = self.network(x)
    probas = F.softmax(logits, dim=1)
    return logits, probas 

# No momentum, no cropping
torch.manual_seed(RANDOM_SEED)
model1 = MLP1(num_features=NUM_FEATURES, 
              drop_proba=0.2,
              num_hidden1=1000,
              num_hidden2=1000,
              num_classes=NUM_CLASSES)

model1 = model1.to(DEVICE)

optimizer = torch.optim.SGD(model1.parameters(), lr=LEARNING_RATE)

GRAYSCALE=True

minibatch_cost1, epoch_cost1, labels1, predictions1 = train_MLP(model1, train_loader_Grayscale, 
                                                                valid_loader_Grayscale, test_loader_Grayscale)

plt.plot(range(len(minibatch_cost1)), minibatch_cost1)
plt.ylabel('Cross Entropy')
plt.xlabel('Minibatch')
plt.show()

# Momentum, no cropping
torch.manual_seed(RANDOM_SEED)
model1 = MLP1(num_features=NUM_FEATURES, 
              drop_proba=0.2,
              num_hidden1=1000,
              num_hidden2=1000,
              num_classes=NUM_CLASSES)

model1 = model1.to(DEVICE)

optimizer = torch.optim.SGD(model1.parameters(), lr=LEARNING_RATE, momentum = 0.9)

GRAYSCALE=True

minibatch_cost1, epoch_cost1, labels1, predictions1 = train_MLP(model1, train_loader_Grayscale, 
                                                                valid_loader_Grayscale, test_loader_Grayscale)

plt.plot(range(len(minibatch_cost1)), minibatch_cost1)
plt.ylabel('Cross Entropy')
plt.xlabel('Minibatch')
plt.show()

# Momentum, cropping
torch.manual_seed(RANDOM_SEED)
model1 = MLP1(num_features=NUM_FEATURES, 
              drop_proba=0.2,
              num_hidden1=1000,
              num_hidden2=1000,
              num_classes=NUM_CLASSES)

model1 = model1.to(DEVICE)

optimizer = torch.optim.SGD(model1.parameters(), lr=LEARNING_RATE, momentum = 0.9)

GRAYSCALE=True

minibatch_cost1, epoch_cost1, labels1, predictions1 = train_MLP(model1, train_loader_GrayscaleCrop, 
                                                                valid_loader_GrayscaleCrop, test_loader_GrayscaleCrop)

plt.plot(range(len(minibatch_cost1)), minibatch_cost1)
plt.ylabel('Cross Entropy')
plt.xlabel('Minibatch')
plt.show()

####################
# MLP taking binary images as input
####################
class MLP2(nn.Module):
  def __init__(self, num_features, drop_proba, 
               num_hidden1, num_hidden2, num_classes):
    super(MLP2, self).__init__()
    
    self.network = nn.Sequential(
        nn.Linear(num_features, num_hidden1),
        nn.BatchNorm1d(num_hidden1), 
        nn.ReLU(),
        nn.Dropout(drop_proba),
        nn.Linear(num_hidden1, num_hidden2),
        nn.BatchNorm1d(num_hidden2),
        nn.ReLU(),
        nn.Dropout(drop_proba),
        nn.Linear(num_hidden2, num_classes)
    )  
   
  def forward(self, x):
    logits = self.network(x)
    probas = F.softmax(logits, dim=1)
    return logits, probas

# No momentum, no cropping
torch.manual_seed(RANDOM_SEED)
model2 = MLP2(num_features=NUM_FEATURES, 
              drop_proba=0.2,
              num_hidden1=1000,
              num_hidden2=1000,
              num_classes=NUM_CLASSES)

model2 = model2.to(DEVICE)

optimizer = torch.optim.SGD(model2.parameters(), lr=LEARNING_RATE)

GRAYSCALE=True

minibatch_cost2, epoch_cost2, labels2, predictions2 = train_MLP(model2, train_loader_Binary, valid_loader_Binary, test_loader_Binary)

plt.plot(range(len(minibatch_cost2)), minibatch_cost2)
plt.ylabel('Cross Entropy')
plt.xlabel('Minibatch')
plt.show()

# Momentum, no cropping
torch.manual_seed(RANDOM_SEED)
model2 = MLP2(num_features=NUM_FEATURES, 
              drop_proba=0.2,
              num_hidden1=1000,
              num_hidden2=1000,
              num_classes=NUM_CLASSES)

model2 = model2.to(DEVICE)

optimizer = torch.optim.SGD(model2.parameters(), lr=LEARNING_RATE, momentum=0.9)

GRAYSCALE=True

minibatch_cost2, epoch_cost2, labels2, predictions2 = train_MLP(model2, train_loader_Binary, valid_loader_Binary, test_loader_Binary)

plt.plot(range(len(minibatch_cost2)), minibatch_cost2)
plt.ylabel('Cross Entropy')
plt.xlabel('Minibatch')
plt.show()

# Momentum, cropping
torch.manual_seed(RANDOM_SEED)
model2 = MLP2(num_features=NUM_FEATURES, 
              drop_proba=0.2,
              num_hidden1=1000,
              num_hidden2=1000,
              num_classes=NUM_CLASSES)

model2 = model2.to(DEVICE)

optimizer = torch.optim.SGD(model2.parameters(), lr=LEARNING_RATE, momentum=0.9)

GRAYSCALE=True

minibatch_cost2, epoch_cost2, labels2, predictions2 = train_MLP(model2, train_loader_BinaryCrop, valid_loader_BinaryCrop, test_loader_BinaryCrop)

plt.plot(range(len(minibatch_cost2)), minibatch_cost2)
plt.ylabel('Cross Entropy')
plt.xlabel('Minibatch')
plt.show()