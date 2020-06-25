# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 19:12:19 2020

@author: Omer
"""

import torchvision
import torch
from torchvision import transforms
from torch import nn, optim
import torch.nn.functional as F
from torch.utils import data

from PIL import Image
from IPython.display import display

#%% Data Loaders from directory
transforms = transforms.Compose([transforms.Resize((64,64)), 
                                 transforms.ToTensor(), 
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

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

#%% Creating neural network and def training process

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(12288, 84) # 3*64*64 = 12288 (3 color channel, 64x64 pixels)
        self.fc2 = nn.Linear(84, 50) # Linear is equal to Dense on Keras.
        self.fc3 = nn.Linear(50,2) # we have two classes.
        
    def forward(self, x):
        x = x.view(-1, 12288)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # As we are using cross entropy loss, there is no need to use softmax. I tried to add softmax on this row but after that my model didn't converge.
        return x
    
simplenet = SimpleNet()


optimizer = optim.Adam(simplenet.parameters(), lr=0.001)

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

train(simplenet, optimizer, torch.nn.CrossEntropyLoss(), train_data_loader, val_data_loader)

#%% Testing

labels = ['cat','fish']
path = "./test/cat/100970828_a2e6ade482.jpg"
img = Image.open(path)
display(img)
img = transforms(img)
print(img.size())
img = img.unsqueeze(0) # reshapeing input from 3*64*64 to 1*3*64*64
print(img.size())
prediction = simplenet(img)
prediction = prediction.argmax()

print(labels[prediction])

corrects = 0
total = 0
for batch in test_data_loader:
    inputs, targets = batch
    prediction = simplenet(img)
    correct = torch.eq(torch.max(F.softmax(prediction, dim=1), dim=1)[1], targets).view(-1)
    correct_num = torch.sum(correct)
    total_instance = correct.size()[0]
    corrects += correct_num.item()
    total += total_instance
print("Model accuracy on test data: {}%".format(round(corrects/total*100,2)))