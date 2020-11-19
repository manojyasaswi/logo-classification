# %%
from __future__ import print_function, division
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from read_images import read_images
from torch.utils.tensorboard import SummaryWriter

import os
import torch
import pandas as pd
import pickle
import tensorflow as tf
import tensorboard as tb
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
plt.ion()   # interactive mode

is_store_data = False
is_read_data = True
# %%
if is_read_data:
    img_list, class_list = read_images()

# %%
# Converting classes to onehotencoding
class_set = set(class_list)
class_int = [x for x in range(0,len(class_set))]

class_one_hot = [list(class_set).index(class_list[x]) for x in range(len(class_list))]
one_hot = torch.nn.functional.one_hot(torch.tensor(class_one_hot))
print(class_one_hot)
# Problems where CrossEntropy Loss doesn't accept onehot value so reverting to class indices values
class_list = np.asarray(class_one_hot)

# %%

def store_data(is_store_true, img_list, class_list):
    if is_store_true:
        file_wr = open('img_data.pkl', 'wb')
        pickle.dump(img_list, file_wr)
        file_wr.close()

        file_wr_c = open('class_data.pkl', 'wb')
        pickle.dump(class_list, file_wr_c)
        file_wr_c.close()

        file_rd = open('img_data.pkl', 'rb')
        img_list = pickle.load(file_rd)
        file_rd.close()

        file_rd_c = open('class_data.pkl', 'rb')
        class_list = pickle.load(file_rd_c)
        file_rd_c.close()
    else:
        file_rd = open('img_data.pkl', 'rb')
        img_list = pickle.load(file_rd)
        file_rd.close()
        file_rd_c = open('class_data.pkl', 'rb')
        class_list = pickle.load(file_rd)
        file_rd_c.close()

    return img_list, class_list

# %%

class Logo_Dataset(Dataset):

    def __init__(self, img_list, class_list, transform=None):
        self.img_list = img_list
        self.class_list = class_list
        self.transform = transform
    
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img = img_list[idx]
        class_name = class_list[idx]
        sample = {'image': img, 'class_name': class_name}

        if self.transform:
            sample = self.transform(sample)
        
        return sample

class Rescale(object):
    """ Rescaling image to given size - output_size
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int,tuple))
        self.output_size = output_size
    
    def __call__(self, sample):
        image, class_name = sample['image'], sample['class_name']
        h, w = image.shape[:2]
        new_h =  self.output_size
        new_w =  self.output_size
        img = transform.resize(image, (new_h, new_w))

        return {'image': img, 'class_name': class_name}

#class RandomCrop(object):

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, class_name = sample['image'], sample['class_name']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'class_name': torch.tensor(class_name)}
        
# %%
transformed_dataset = Logo_Dataset(img_list, class_list  , transform = transforms.Compose([Rescale(128), ToTensor()]))
dataloader = DataLoader(transformed_dataset, batch_size=4,
                        shuffle=True, num_workers=0)

# %% Visualize Random Samples
dataiter = iter(dataloader)
sample_batched = dataiter.next()

for i in range(4):
    plt.figure(i)
    plt.imshow(sample_batched['image'][i].numpy().transpose((1, 2, 0)))
    plt.title(sample_batched['class_name'][i])
# %%  Define NN
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(32, 64, 5)
        #self.fc1 = nn.Linear(64 * 5 * 5, 400)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        #x = x.view(-1, 64 * 5 * 5)
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

net = Net().double() ### Problem 1
criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=0.0001, betas =(0.9,0.999))
# %%
loss_capture = []

for epoch in range(2):  # loop over the dataset multiple times
    running_loss = 0.0
    for i_batch, sample_batched in enumerate(dataloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs = sample_batched['image'].double()
        labels = sample_batched['class_name']
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        #print("Outputs:\n",outputs) # Debug
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        loss_capture.append(loss.item())
        if i_batch % 20 == 19:    # print every 20 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i_batch + 1, running_loss / 20))
            running_loss = 0.0

print('Finished Training')

plt.plot(loss_capture)
plt.title("Loss Curve")
# %% TensorBoard

# default `log_dir` is "runs" 
writer = SummaryWriter('runs/logos_10_class_exp_1')
dataiter = iter(dataloader)
batch_tb = dataiter.next()

# create grid of images
img_grid = torchvision.utils.make_grid(batch_tb['image'])

# write to tensorboard
writer.add_image('four images', img_grid)
## Run in command Line tensorboard --logdir=runs

writer.add_graph(net.float(), batch_tb['image'])  ## Problems
#writer.close()

# %%
# n Random Images
import random
n_images = 24
max_range = len(img_list)
img_group = []
class_group = []
for i in range(n_images):
    sample_test = transformed_dataset.__getitem__(random.randint(0,max_range))
    img_group.append(sample_test['image'][0])
    class_group.append(sample_test['class_name'])

img_group_t = torch.stack(img_group)
class_group_t = torch.stack(class_group)

tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
features = img_group_t.view(-1, 128 * 128)
writer.add_embedding(features,
                    metadata=class_group_t,
                    label_img=img_group_t.unsqueeze(1))

writer.close()
