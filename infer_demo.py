import torch
from torch import nn
import torchvision

from torchsummary import summary

import matplotlib.pyplot as plt

from unet import unet
import numpy as np

def plot_save(a, b, path = 'unet_demo/', cnt = 1):
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
    # side by side comparison
    ab = np.concatenate((a,b), axis = 1)
    plt.imshow(ab)
    plt.savefig(path + 'demo' + str(cnt) + '.png', dpi = 150)

model = unet().cuda()

print('Model summary')

input_shape = (3, 128, 128) # pytorch - channel first

summary(model, input_shape)

# load weight

model_path = 'unet_50ep_128.pth'
model.load_state_dict(torch.load(model_path))

# creating the data pipeline

path_list = ['hdr_data/']
img_format = ['.png']

from data_loader import *
import time

Dataset = LPDataset(path = path_list, format = img_format, transform = transforms.Compose([Rescale(), 
                                                                       RandomNoise(),
                                                                       ToTensor(), Normalize() ]))



print(f'Total images loaded: {len(Dataset)}')
time.sleep(2)

batch_size = 16

dataloader = DataLoader(Dataset, batch_size=batch_size,
                        shuffle=True, num_workers=4)



model.eval()


cnt = 0
t = 0
for i, data in enumerate(dataloader, 0):
    # get the inputs; data is a list of [inputs, labels]
    X, y = data
    t1 = time.time()
    X = X.cuda()
    y = y.cuda()
    outputs = model(X)
    t2 = time.time()
    
    t += (t2-t1)

    plot_save(X[0,:,:,:].cpu(), y[0,:,:,:].cpu(), 'unet_demo/', cnt)
    
    cnt += 1
    
    if cnt == 100:
        break

print(f'Time average: {t/100.}')
