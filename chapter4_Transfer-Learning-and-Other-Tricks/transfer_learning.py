# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 18:58:14 2020

@author: Omer
"""


from torchvision import models
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
import torch.utils.data as data
import torch
import torch.optim as optim
import torch.nn.functional as F

#%% Using ResNet50
transfer_model = models.resnet50(pretrained=True)

for name, param in transfer_model.named_parameters():
    if("bn" not in name):
        param.requires_grad = False

transfer_model.fc = nn.Sequential(nn.Linear(transfer_model.fc.in_features,500), 
                                  nn.ReLU(), 
                                  nn.Dropout(), 
                                  nn.Linear(500,2))

#%% Data Loaders from directory

transforms = transforms.Compose([transforms.Resize((64, 64)),
                                 # Changes brightness, contrast and saturation values of images randomly.
                                 transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0), 
                                 # Random flip process on images. p is chance of the reflection.
                                 transforms.RandomHorizontalFlip(p=0.5),
                                 transforms.RandomVerticalFlip(p=0.0),
                                 # Applying gray scale on images by the chance of 10%.
                                 transforms.RandomGrayscale(p=0.1),
                                 # Random crop process on imgaes.
                                 # transforms.RandomCrop(size, padding=None, pad_if_needed=False, fill=0, padding_mode='reflect'),
                                 # transforms.RandomResizedCrop(size, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2),
                                 # Applying rotation to images
                                 # transforms.RandomRotation(degrees, resample=False,expand=False, center=None),
                                 # Adding padding
                                 # transforms.Pad(padding, fill=0, padding_mode=constant),
                                 # Adding shear, degree etc.
                                 transforms.RandomAffine(45, translate=None, scale=None, shear=0.4, resample=False, fillcolor=0),
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


#%% Def training process

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
optimizer = optim.Adam(transfer_model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()
train(transfer_model, optimizer, loss_fn, train_data_loader, val_data_loader)


#%% Testing
corrects = 0
total = 0
for batch in test_data_loader:
    inputs, targets = batch
    prediction = transfer_model(inputs)
    correct = torch.eq(torch.max(F.softmax(prediction, dim=1), dim=1)[1], targets).view(-1)
    correct_num = torch.sum(correct)
    total_instance = correct.size()[0]
    corrects += correct_num.item()
    total += total_instance
print("Model accuracy on test data: {}%".format(round(corrects/total*100,2)))

#%% FINDING THAT LEARNING RATE
#%% Using AlexNet

alexnet = models.alexnet(num_classes=2)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(alexnet.parameters(), lr=0.001)

#%% Finding Learning Rates and Their Losses
import math
def find_lr(model, loss_fn, optimizer,  train_loader, init_value=1e-4, final_value=1):
    number_in_epoch = len(train_loader) - 1
    update_step = (final_value / init_value) ** (1 / number_in_epoch)
    lr = init_value
    optimizer.param_groups[0]["lr"] = lr
    best_loss = 0.0
    batch_num = 0
    losses = []
    log_lrs = []
    for data in train_loader:
        batch_num += 1
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        # Crash out if loss explodes
        if batch_num > 1 and loss > 4 * best_loss:
            return log_lrs, losses
        # Record the best loss
        if loss < best_loss or batch_num == 1:
            best_loss = loss
        # Store the values
        losses.append(loss)
        log_lrs.append(math.log10(lr))
        # Do the backward pass and optimize
        loss.backward()
        optimizer.step()
        # Update the lr for the next step and store
        lr *= update_step
        optimizer.param_groups[0]["lr"] = lr
    return log_lrs, losses

#%% LR and loss Plot
    
import matplotlib.pyplot as plt
logs,losses = find_lr(alexnet, loss_fn, optimizer, train_data_loader)
plt.plot(logs,losses)

#%% Differential Learning Rates

found_lr = 10**(-2.5)
optimizer = optim.Adam([{ 'params': transfer_model.layer4.parameters(), 'lr': found_lr /3}, { 'params': transfer_model.layer3.parameters(), 'lr': found_lr /9}], 
                           lr=found_lr)

# For unfreezing layers.
unfreeze_layers = [transfer_model.layer3, transfer_model.layer4]
for layer in unfreeze_layers:
    for param in layer.parameters():
        param.requires_grad = True