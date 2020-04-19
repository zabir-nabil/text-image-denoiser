
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


def sp_noise(image,prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

class Rescale(object):
    def __init__(self, output_size = (128,128,3)):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        X = sample
        
        X = cv2.resize(X, (self.output_size[0], self.output_size[1]))

        return X


class RandomNoise(object):
    def __call__(self, sample):
        wk = random.choice([5, 7, 9]) # increase for large images, decrease for smaller ones
        hk = random.choice([5, 7, 9])
        
        y = sample[:]
        X = sample[:]

        X = sp_noise(X, random.randint(5,10)/300.) # increase 300. if you have large dimension image, decrease for small images

        X = cv2.GaussianBlur(X,(wk,hk),cv2.BORDER_DEFAULT)
        



        return X, y


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        X, y = sample[0], sample[1]

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        X = X.transpose((2, 0, 1))
        y = y.transpose((2, 0, 1))
        return torch.from_numpy(X), torch.from_numpy(y)
    
class Normalize(object):
    def __call__(self, sample):

        X = sample[0].float()/255.
        y = sample[1].float()/255.
        return X, y


# creating LP dataset

from torch.utils.data import Dataset, DataLoader
import glob
import os
import cv2
import time


# loads all the images from a list of folders


class LPDataset(Dataset):
    def __init__(self, path, format = '.jpg', transform = None): # path is a list of path or a single path, format is a list of format or a single string
        # paths, names of all the images in a folder
        if type(path) == str:
            self.img_list = glob.glob(os.path.join(path, '*' + format))
        else: # a list of str
            self.img_list = []
            i = 0
            for p in path:
                frm = format
                if type(format) == list:
                    frm = format[i]
                    i += 1

                self.img_list += glob.glob(os.path.join(p, '*' + frm))

        self.transform = transform
        
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        img = cv2.imread(self.img_list[idx])
        
        if self.transform:
            img = self.transform(img)
        
        return img