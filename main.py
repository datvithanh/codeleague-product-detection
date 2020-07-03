from __future__ import print_function, division

import time
import os
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from tensorboardX import SummaryWriter
import argparse
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet

from dataset import LoadDataset



parser = argparse.ArgumentParser()

parser.add_argument('--data_path', required=True, type=str)
parser.add_argument('--pretrained', action='store_false', default=True)
parser.add_argument('--load', default='', type=str)
parser.add_argument('--name', default='', type=str)
parser.add_argument('--start', type=int, default=0)

params = parser.parse_args()

data_dir = params.data_path

if params.name == '':
    params.name = data_dir.split('/')[-1]
    if params.name == '':
        params.name = data_dir.split('/')[-2]

dataloaders = {x: LoadDataset(x, data_dir, batch_size=6) for x in ['train', 'val']}

os.makedirs(os.path.join('log', params.name), exist_ok=True)
logger = SummaryWriter(os.path.join('log', params.name))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

os.makedirs(os.path.join('ckpt', params.name), exist_ok=True)
ckpt_dir = os.path.join('ckpt', params.name)

def train_model(model, criterion, optimizer, scheduler, num_epochs=25, start=0):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(start, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            cnt = 0
            ds_len = 0
            print(phase)
            for inputs, labels in tqdm(dataloaders[phase]):
                cnt += 1
                inputs = inputs.to(device, dtype=torch.float)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)

                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                ds_len += len(preds)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / ds_len
            epoch_acc = running_corrects.double() / ds_len
            
            logger.add_scalars('loss', {phase: epoch_loss}, epoch)
            logger.add_scalars('acc', {phase: epoch_acc}, epoch)
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model, os.path.join(ckpt_dir, f'model_epoch{epoch}_{best_acc}'))


        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

# if params.load == '':
#     model_ft = models.resnet18(pretrained=True)
# else:
#     model_ft = torch.load(params.load)

print(params.pretrained)
# print(params.load)

# if params.load == '':
# model_ft = models.resnet101(pretrained=params.pretrained)
if params.load == '':
    model_ft = EfficientNet.from_pretrained('efficientnet-b5', num_classes=42) 
else:
    model_ft = torch.load(params.load)

#print(model_ft)
for parameter in model_ft.parameters():
    print(parameter.requires_grad)

# model_ft = models.resnet101(pretrained=params.pretrained)
# else:
#     model_ft = torch.load(params.load)

# num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 42.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
# model_ft.fc = nn.Linear(num_ftrs, 42)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.0001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
#
# It should take around 15-25 min on CPU. On GPU though, it takes less than a
# minute.
#

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=50, start=params.start)

