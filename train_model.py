# training a deep denoiser model

import random
from torchvision import transforms
import torch
from torch import nn
import numpy as np
import cv2

from torchsummary import summary


# image noise add -> RadnomNoise
# sp_noise: salt pepper
# gaussian blur

# changing noise type
# edit RandomNoise class

# changing model type -> import a different model

# data loading
from data_loader import *


import matplotlib.pyplot as plt

# tensor plotting helper function

def plot_sample(a, b):
    a = a.numpy()
    b = b.numpy()
    a = np.transpose(a, (1, 2, 0))
    b = np.transpose(b, (1, 2, 0))
    #print(a.shape)
    #print(b.shape)
    #print(np.min(a))
    #print(np.max(a))
    #print(np.min(b))
    #print(np.max(b))
    plt.imshow(a)
    plt.show()
    plt.imshow(b)
    plt.show()


# creating the data pipeline

path_list = ['hdr_data/']
img_format = ['.png']

Dataset = LPDataset(path = path_list, format = img_format, transform = transforms.Compose([Rescale(), 
                                                                       RandomNoise(),
                                                                       ToTensor(), Normalize() ]))



print(f'Total images loaded: {len(Dataset)}')
time.sleep(2)

batch_size = 16

dataloader = DataLoader(Dataset, batch_size=batch_size,
                        shuffle=True, num_workers=4)


# select model

from unet import unet

model = unet().cuda()


print('Model summary')

input_shape = (3, 128, 128) # pytorch - channel first

summary(model, input_shape)


# optimizer and loss

import torch.optim as optim

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# training loop

EPOCH = 50

loss_c = []

from tqdm import tqdm

for epoch in range(EPOCH):  # loop over the dataset multiple times

    running_loss = 0.0

    print(f'Epoch {epoch+1}')
    
    for i, data in tqdm(enumerate(dataloader, 0)):
        # get the inputs; data is a list of [inputs, labels]
        X, y = data
        X = X.cuda()
        y = y.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

    loss_c.append(running_loss)
    print('EPOCH: [%d] loss: %.3f' %
          (epoch + 1, running_loss))

print('Finished Training')
torch.save(model.state_dict(), 'unet_50ep_128.pth')

plt.plot(loss_c)
plt.savefig('model_loss.png', dpi = 200)




