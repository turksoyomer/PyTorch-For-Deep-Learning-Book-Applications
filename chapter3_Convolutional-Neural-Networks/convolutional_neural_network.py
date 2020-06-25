# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 15:03:24 2020

@author: Omer
"""


import torch
import torchvision
from torchvision import transforms
from torch.utils import data
import torch.optim as optim
from torch import nn
from torch.nn import functional as F

#%% Data Loaders from directory
transforms = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


train_data_path = "./train/"
train_data = torchvision.datasets.ImageFolder(root=train_data_path,transform=transforms)

val_data_path = "./val/"
val_data = torchvision.datasets.ImageFolder(root=val_data_path,transform=transforms)

test_data_path = "./test/"
test_data = torchvision.datasets.ImageFolder(root=test_data_path,transform=transforms)


batch_size=64
train_data_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_data_loader = data.DataLoader(val_data, batch_size=batch_size)
test_data_loader = data.DataLoader(test_data, batch_size=batch_size)

#%% Creating convolutional neural network and def training process

class CNNNet(nn.Module): # AlexNet implementation!
    def __init__(self, num_classes=2):
        super(CNNNet, self).__init__()
        # Sequential allows us to create chains
        self.features = nn.Sequential(
                            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2), # in_channels,out_channels, kernel_size, stride, padding
                            nn.ReLU(),
                            nn.MaxPool2d(kernel_size=3, stride=2), # can use padding with pooling.
                            nn.Conv2d(64, 192, kernel_size=5, padding=2), # don't have to set padding, PyTorch can handle it.
                            nn.ReLU(),
                            nn.MaxPool2d(kernel_size=3, stride=2),
                            nn.Conv2d(192, 384, kernel_size=3, padding=1),
                            nn.ReLU(),
                            nn.Conv2d(384, 256, kernel_size=3, padding=1),
                            nn.ReLU(),
                            nn.Conv2d(256, 256, kernel_size=3, padding=1),
                            nn.ReLU(),
                            nn.MaxPool2d(kernel_size=3, stride=2),
                        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6)) # works independently of the incoming input tensor’s dimensions, returns 6x6 pixels.
        self.classifier = nn.Sequential(
                            nn.Dropout(), # default dropout rate is 0.5
                            nn.Linear(256 * 6 * 6, 4096),
                            nn.ReLU(),
                            nn.Dropout(), # nn.Dropout(p=0.2)
                            # Note for Dropout: We strongly need to use model.train() or model.eval() as to inform the model.
                            # If we don't do that, our model can use dropout technique on testing.
                            nn.Linear(4096, 4096),
                            nn.ReLU(),
                            nn.Linear(4096, num_classes)
                        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    

alexnet = CNNNet()

optimizer = optim.Adam(alexnet.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()


def train(model, optimizer, loss_fn, train_loader, val_loader, epochs=20, device="cpu"):
    for epoch in range(epochs):
        training_loss = 0.0
        valid_loss = 0.0
        model.train()   # training mode
        for batch in train_loader:
            optimizer.zero_grad()
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = model(inputs)
            loss = loss_fn(output, targets)
            loss.backward()
            optimizer.step()
            training_loss += loss.data.item() * inputs.size(0)
        training_loss /= len(train_loader.dataset)
        
        model.eval()   # evaluation mode for test.
        num_correct = 0
        num_examples = 0
        for batch in val_loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            output = model(inputs)
            targets = targets.to(device)
            loss = loss_fn(output,targets)
            valid_loss += loss.data.item() * inputs.size(0)
            correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1], targets).view(-1)
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]
        valid_loss /= len(val_loader.dataset)
        print('Epoch: {}, Training Loss: {:.2f}, Validation Loss: {:.2f}, accuracy = {:.2f}'.format(epoch, training_loss, valid_loss, num_correct / num_examples))
        
        
#%% Training
train(alexnet, optimizer, loss_fn, train_data_loader, val_data_loader)

#%% Testing
corrects = 0
total = 0
for batch in test_data_loader:
    inputs, targets = batch
    prediction = alexnet(inputs)
    correct = torch.eq(torch.max(F.softmax(prediction, dim=1), dim=1)[1], targets).view(-1)
    correct_num = torch.sum(correct)
    total_instance = correct.size()[0]
    corrects += correct_num.item()
    total += total_instance
print("Model accuracy on test data: {}%".format(round(corrects/total*100,2)))