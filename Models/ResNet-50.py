# Convolutional Neural Network ResNet-50

import os
os.environ['CUDA_LAUNCH_BLOCKING']="1"
import time
import torch
import base64
from IPython.display import HTML
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
%matplotlib inline
from torchvision import models

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    
from google.colab import drive
drive.mount("/content/drive")    

####################
# Settings
####################
# Hyperparameters
RANDOM_SEED = 1
LEARNING_RATE = 0.1
NUM_EPOCHS = 30

# Architecture
NUM_CLASSES = 45
BATCH_SIZE = 256
DEVICE = 'cuda:0' # default GPU device

####################
# I. Custom dataset class
####################
class MyDataset_RGB(Dataset):

    def __init__(self, csv_path, img_dir, transform=None):
    
        df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.img_names = df['File Name']
        self.y = df['Class Label']
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir,
                                      self.img_names[index]))
        
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
        # If pixel value smaller than threshold, return 0. Otherwise return 1.
        filter_func = lambda x: 0 if x < threshold else 1
        img=img.point(filter_func, "1")
          
        if self.transform is not None:
            img = self.transform(img)
        
        label = self.y[index]
        return img, label

    def __len__(self):
        return self.y.shape[0]
    
class MyDataset_Grayscale(Dataset):

    def __init__(self, csv_path, img_dir, transform=None):
    
        df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.img_names = df['File Name']
        self.y = df['Class Label']
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir,
                                      self.img_names[index]))
                  
        if self.transform is not None:
            img = self.transform(img)
        
        label = self.y[index]
        return img, label

    def __len__(self):
        return self.y.shape[0]    
    
####################
# II. Data Augmentation
####################    
# Transformation for grayscale images
train_transforms_Grayscale = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomAffine(degrees=(-20, 20), translate=(0.15, 0.15),
                                        resample=Image.BILINEAR),
    transforms.Resize(size=(40, 40)),
    transforms.RandomCrop(32),
    transforms.ToTensor()
])

valid_transforms_Grayscale = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(size=(40, 40)),
    transforms.CenterCrop(32),
    transforms.ToTensor()
])

test_transforms_Grayscale = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(size=(40, 40)),
    transforms.CenterCrop(32),
    transforms.ToTensor()
])

# Transformation for binary images
train_transforms_Binary = transforms.Compose([
    transforms.RandomAffine(degrees=(-20, 20), translate=(0.15, 0.15),
                                        resample=Image.BILINEAR),
    transforms.Resize(size=(40, 40)),
    transforms.RandomCrop(32),
    transforms.ToTensor()
])

valid_transforms_Binary = transforms.Compose([
    transforms.Resize(size=(40, 40)),
    transforms.CenterCrop(32),
    transforms.ToTensor()
])

test_transforms_Binary = transforms.Compose([
    transforms.Resize(size=(40, 40)),
    transforms.CenterCrop(32),
    transforms.ToTensor()
])

# Transformation for RGB images
train_transforms_RGB = transforms.Compose([
    transforms.RandomAffine(degrees=(-20, 20), translate=(0.15, 0.15),
                                        resample=Image.BILINEAR),
    transforms.Resize(size=(40, 40)),
    transforms.RandomCrop(32),
    transforms.ToTensor()
])

valid_transforms_RGB = transforms.Compose([
    transforms.Resize(size=(40, 40)),
    transforms.CenterCrop(32),
    transforms.ToTensor()
])

test_transforms_RGB = transforms.Compose([
    transforms.Resize(size=(40, 40)),
    transforms.CenterCrop(32),
    transforms.ToTensor()
])

####################
# III. Custom Data Loader
####################  
# Loader for grayscale images
train_dataset_Grayscale = MyDataset_Grayscale(csv_path='/content/drive/My Drive/STAT 479 Project/TrainingCorrect_combinedULcases.csv',
                                              img_dir='/content/drive/My Drive/STAT 479 Project/RenamedData',
                                              transform=train_transforms_Grayscale)
train_loader_Grayscale = DataLoader(dataset=train_dataset_Grayscale,
                                    batch_size=BATCH_SIZE,
                                    shuffle=True, num_workers=4)

valid_dataset_Grayscale = MyDataset_Grayscale(csv_path='/content/drive/My Drive/STAT 479 Project/ValidationCorrect_combinedULcases.csv',
                                              img_dir='/content/drive/My Drive/STAT 479 Project/RenamedData',
                                              transform=valid_transforms_Grayscale)
valid_loader_Grayscale = DataLoader(dataset=valid_dataset_Grayscale,
                                    batch_size=BATCH_SIZE,
                                    shuffle=False, num_workers=4)

test_dataset_Grayscale = MyDataset_Grayscale(csv_path='/content/drive/My Drive/STAT 479 Project/TestingCorrect_combinedULcases.csv',
                                             img_dir='/content/drive/My Drive/STAT 479 Project/RenamedData',
                                             transform=test_transforms_Grayscale)
test_loader_Grayscale = DataLoader(dataset=test_dataset_Grayscale,
                                   batch_size=BATCH_SIZE,
                                   shuffle=False, num_workers=4)

# Loader for binary images
train_dataset_Binary = MyDataset_Binary(csv_path='/content/drive/My Drive/STAT 479 Project/TrainingCorrect_combinedULcases.csv',
                                              img_dir='/content/drive/My Drive/STAT 479 Project/RenamedData',
                                              transform=train_transforms_Binary)
train_loader_Binary = DataLoader(dataset=train_dataset_Binary,
                                    batch_size=BATCH_SIZE,
                                    shuffle=True, num_workers=4)

valid_dataset_Binary = MyDataset_Binary(csv_path='/content/drive/My Drive/STAT 479 Project/ValidationCorrect_combinedULcases.csv',
                                              img_dir='/content/drive/My Drive/STAT 479 Project/RenamedData',
                                              transform=valid_transforms_Binary)
valid_loader_Binary = DataLoader(dataset=valid_dataset_Binary,
                                    batch_size=BATCH_SIZE,
                                    shuffle=False, num_workers=4)

test_dataset_Binary = MyDataset_Binary(csv_path='/content/drive/My Drive/STAT 479 Project/TestingCorrect_combinedULcases.csv',
                                             img_dir='/content/drive/My Drive/STAT 479 Project/RenamedData',
                                             transform=test_transforms_Binary)
test_loader_Binary = DataLoader(dataset=test_dataset_Binary,
                                   batch_size=BATCH_SIZE,
                                   shuffle=False, num_workers=4)

# Loader for RGB images
train_dataset_RGB = MyDataset_RGB(csv_path='/content/drive/My Drive/STAT 479 Project/TrainingCorrect_combinedULcases.csv',
                                  img_dir='/content/drive/My Drive/STAT 479 Project/RenamedData',
                                  transform=train_transforms_RGB)
train_loader_RGB = DataLoader(dataset=train_dataset_RGB,
                              batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=4)

valid_dataset_RGB = MyDataset_RGB(csv_path='/content/drive/My Drive/STAT 479 Project/ValidationCorrect_combinedULcases.csv',
                                  img_dir='/content/drive/My Drive/STAT 479 Project/RenamedData',
                                  transform=valid_transforms_RGB)
valid_loader_RGB = DataLoader(dataset=valid_dataset_RGB,
                              batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=4)

test_dataset_RGB = MyDataset_RGB(csv_path='/content/drive/My Drive/STAT 479 Project/TestingCorrect_combinedULcases.csv',
                                 img_dir='/content/drive/My Drive/STAT 479 Project/RenamedData',
                                 transform=test_transforms_RGB)
test_loader_RGB = DataLoader(dataset=test_dataset_RGB,
                              batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=4)

####################
# IV. Model
#################### 
class Bottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, inplanes, planes, stride=1, downsample=None):
      
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes, grayscale):
        
        self.inplanes = 64
        
        if grayscale:
            in_dim = 1
        else:
            in_dim = 3
            
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AvgPool2d(7, stride=1)
        
        self.fc1 = nn.Linear(512 * block.expansion, 512 * block.expansion)
        self.fcbn1 = nn.BatchNorm1d(512 * block.expansion)
        self.fcdropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512 * block.expansion, 512)
        self.fcbn2 = nn.BatchNorm1d(512)
        self.fcdropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(512, NUM_CLASSES)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n)**.5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # because our dataset is already 1x1 here:
        # disable avg pooling
        # x = self.avgpool(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fcbn1(x)
        x = F.relu(x)
        x = self.fcdropout1(x)
        x = self.fc2(x)
        x = self.fcbn2(x)
        x = F.relu(x)
        x = self.fcdropout2(x)
        logits = self.fc3(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas

  
def resnet50_RGB(num_classes):
    """Constructs a ResNet-50 model, using RGB images as input."""
    model = ResNet(block=Bottleneck,
                   layers=[3, 4, 6, 3],
                   num_classes=NUM_CLASSES,
                   grayscale=False)
    return model

def resnet50_grayscale(num_classes):
    """Constructs a ResNet-50 model, using grayscale images as input, either binary or not."""
    model = ResNet(block=Bottleneck, 
                   layers=[3, 4, 6, 3],
                   num_classes=NUM_CLASSES,
                   grayscale=True)
    return model

# Training
### detail: if true, return the prediction output
def compute_accuracy(model, data_loader, detail):
    model.eval()
    correct_pred, num_examples = 0, 0
    labels, predictions, result = [], [], []
    for i, (features, targets) in enumerate(data_loader):
            
        features = features.to(DEVICE)
        targets = targets.to(DEVICE)

        logits, probas = model(features)
        _, predicted_labels = torch.max(probas, 1)
        
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


def compute_epoch_loss(model, data_loader):
    model.eval()
    curr_loss, num_examples = 0., 0
    with torch.no_grad():
        for features, targets in data_loader:
            features = features.to(DEVICE)
            targets = targets.to(DEVICE)
            logits, probas = model(features)
            loss = F.cross_entropy(logits, targets, reduction='sum')
            num_examples += targets.size(0)
            curr_loss += loss

        curr_loss = curr_loss / num_examples
        return curr_loss

####################
# ResNet-50 taking RGB images as input
####################     
torch.manual_seed(RANDOM_SEED)

model = resnet50_RGB(NUM_CLASSES)
model.to(DEVICE)

optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    
minibatch_cost, epoch_cost = [], []

start_time = time.time()

train_loader = train_loader_RGB
valid_loader = valid_loader_RGB
test_loader = test_loader_RGB
    
for epoch in range(NUM_EPOCHS):
    model.train()
    for batch_idx, (features, targets) in enumerate(train_loader):
        
        features = features.to(DEVICE)
        targets = targets.to(DEVICE)
            
        ### FORWARD AND BACK PROP
        logits, probas = model(features)
        cost = F.cross_entropy(logits, targets)
        optimizer.zero_grad()
        
        cost.backward()
        minibatch_cost.append(cost)
        
        ### UPDATE MODEL PARAMETERS
        optimizer.step()
        
        ### LOGGING
        if not batch_idx % 50:
            print ('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f' 
                   %(epoch+1, NUM_EPOCHS, batch_idx, 
                     len(train_loader), cost))

    model.eval()
    with torch.set_grad_enabled(False): # save memory during inference
        if not (epoch+1) % 10:
            cost = compute_epoch_loss(model, train_loader)
            epoch_cost.append(cost)
            train_acc = compute_accuracy(model, train_loader, detail = False)
            valid_acc = compute_accuracy(model, valid_loader, detail = False)
            
            print('Epoch: %03d/%03d | Train Cost: %.4f' % (epoch+1, NUM_EPOCHS, cost))
            print('Train Accuracy: %.3f%% | Validation Accuracy: %.3f%%' % (train_acc, valid_acc))
            
    print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
    
print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
with torch.set_grad_enabled(False): # save memory during inference
    test_accuracy, labels, predictions = compute_accuracy(model, test_loader, detail=True)
    print('Test accuracy: %.2f%%' % (test_accuracy))
    
plt.plot(range(len(minibatch_cost)), minibatch_cost)
plt.ylabel('Cross Entropy')
plt.xlabel('Minibatch')
plt.ylim([0, 4])
plt.show()    

####################
# ResNet-50 taking grayscale images as input
####################     
torch.manual_seed(RANDOM_SEED)

model = resnet50_grayscale(NUM_CLASSES)
model.to(DEVICE)

optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    
minibatch_cost, epoch_cost = [], []

train_loader = train_loader_Grayscale
valid_loader = valid_loader_Grayscale
test_loader = test_loader_Grayscale

start_time = time.time()
for epoch in range(NUM_EPOCHS):
    model.train()
    for batch_idx, (features, targets) in enumerate(train_loader):
        
        features = features.to(DEVICE)
        targets = targets.to(DEVICE)
            
        ### FORWARD AND BACK PROP
        logits, probas = model(features)
        cost = F.cross_entropy(logits, targets)
        optimizer.zero_grad()
        
        cost.backward()
        minibatch_cost.append(cost)
        
        ### UPDATE MODEL PARAMETERS
        optimizer.step()
        
        ### LOGGING
        if not batch_idx % 50:
            print ('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f' 
                   %(epoch+1, NUM_EPOCHS, batch_idx, 
                     len(train_loader), cost))

    model.eval()
    with torch.set_grad_enabled(False): # save memory during inference
        if not (epoch+1) % 10:
            cost = compute_epoch_loss(model, train_loader)
            epoch_cost.append(cost)
            train_acc = compute_accuracy(model, train_loader, detail = False)
            valid_acc = compute_accuracy(model, valid_loader, detail = False)
            
            print('Epoch: %03d/%03d | Train Cost: %.4f' % (epoch+1, NUM_EPOCHS, cost))
            print('Train Accuracy: %.3f%% | Validation Accuracy: %.3f%%' % (train_acc, valid_acc))
            
    print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
    
print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
with torch.set_grad_enabled(False): # save memory during inference
    test_accuracy, labels, predictions = compute_accuracy(model, test_loader, detail=True)
    print('Test accuracy: %.2f%%' % (test_accuracy))
    
plt.plot(range(len(minibatch_cost)), minibatch_cost)
plt.ylabel('Cross Entropy')
plt.xlabel('Minibatch')
plt.ylim([0, 4])
plt.show()

####################
# ResNet-50 taking binary grayscale images as input
####################  
torch.manual_seed(RANDOM_SEED)

model = resnet50_grayscale(NUM_CLASSES)
model.to(DEVICE)

optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

train_loader = train_loader_Binary
valid_loader = valid_loader_Binary
test_loader = test_loader_Binary
    
minibatch_cost, epoch_cost = [], []

start_time = time.time()
for epoch in range(NUM_EPOCHS):
    model.train()
    for batch_idx, (features, targets) in enumerate(train_loader):
        
        features = features.to(DEVICE)
        targets = targets.to(DEVICE)
            
        ### FORWARD AND BACK PROP
        logits, probas = model(features)
        cost = F.cross_entropy(logits, targets)
        optimizer.zero_grad()
        
        cost.backward()
        minibatch_cost.append(cost)
        
        ### UPDATE MODEL PARAMETERS
        optimizer.step()
        
        ### LOGGING
        if not batch_idx % 50:
            print ('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f' 
                   %(epoch+1, NUM_EPOCHS, batch_idx, 
                     len(train_loader), cost))

    model.eval()
    with torch.set_grad_enabled(False): # save memory during inference
        if not (epoch+1) % 10:
            cost = compute_epoch_loss(model, train_loader)
            epoch_cost.append(cost)
            train_acc = compute_accuracy(model, train_loader, detail = False)
            valid_acc = compute_accuracy(model, valid_loader, detail = False)
            
            print('Epoch: %03d/%03d | Train Cost: %.4f' % (epoch+1, NUM_EPOCHS, cost))
            print('Train Accuracy: %.3f%% | Validation Accuracy: %.3f%%' % (train_acc, valid_acc))
            
    print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
    
print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
with torch.set_grad_enabled(False): # save memory during inference
    test_accuracy, labels, predictions = compute_accuracy(model, test_loader, detail=True)
    print('Test accuracy: %.2f%%' % (test_accuracy))
    
plt.plot(range(len(minibatch_cost)), minibatch_cost)
plt.ylabel('Cross Entropy')
plt.xlabel('Minibatch')
plt.ylim([0, 5])
plt.show()