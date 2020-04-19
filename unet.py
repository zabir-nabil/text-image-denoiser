# UNet pytorch implementation
# author: github/zabir-nabil
# input_shape (3, k*2^5, k*2^5) k is an integer

import torch.nn as nn
import torch
from torchsummary import summary

class unet(nn.Module):
    # input height, width has to be multiple of 2^5
    def __init__(self):
        super(unet, self).__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(3, 64, 3, padding = 1), #0
            nn.ReLU(), #1 
            nn.Conv2d(64, 64, 3, padding = 1), #2
            nn.ReLU(), #3
            nn.MaxPool2d(2), #4
            
            nn.Conv2d(64, 128, 3, padding = 1), #5
            nn.ReLU(), #6
            nn.Conv2d(128, 128, 3, padding = 1), #7
            nn.ReLU(), #8
            nn.MaxPool2d(2), #9
            
            
            nn.Conv2d(128, 256, 3, padding = 1), #10
            nn.ReLU(), #11
            nn.Conv2d(256, 256, 3, padding = 1), #12
            nn.ReLU(), #13
            nn.MaxPool2d(2), #14
            
            
            nn.Conv2d(256, 512, 3, padding = 1), #15
            nn.ReLU(), #16
            nn.Conv2d(512, 512, 3, padding = 1), #17
            nn.ReLU(), #18
            nn.Dropout(0.5), #19
            nn.MaxPool2d(2), #20
            
            
            nn.Conv2d(512, 1024, 3, padding = 1), #21
            nn.ReLU(), #22
            nn.Conv2d(1024, 1024, 3, padding = 1), #23
            nn.ReLU(), #24
            nn.Dropout(0.5), #25
            
            nn.Upsample(scale_factor=2), # 26
            
            nn.Conv2d(1024, 512, 3, padding = 1), #27
            nn.ReLU(), #28
            
            # a internal concat -> 28, 19
            nn.Conv2d(1024, 512, 3, padding = 1), #29
            nn.ReLU(), #30
            nn.Conv2d(512, 512, 3, padding = 1), #31
            nn.ReLU(), #32
            
            nn.Upsample(scale_factor=2), # 33
            
            nn.Conv2d(512, 256, 3, padding = 1), #34
            nn.ReLU(), #35
            
            # a internal concat -> 35, 13
            nn.Conv2d(512, 256, 3, padding = 1), #36
            nn.ReLU(), #37
            nn.Conv2d(256, 256, 3, padding = 1), #38
            nn.ReLU(), #39
            
            nn.Upsample(scale_factor=2), # 40
            
            nn.Conv2d(256, 128, 3, padding = 1), #41
            nn.ReLU(), #42
            
            # a internal concat -> 42, 8
            nn.Conv2d(256, 128, 3, padding = 1), #43
            nn.ReLU(), #44
            nn.Conv2d(128, 128, 3, padding = 1), #45
            nn.ReLU(), #46
            
            
            nn.Upsample(scale_factor=2), # 47
            
            nn.Conv2d(128, 64, 3, padding = 1), #48
            nn.ReLU(), #49
            
            # a internal concat -> 49, 3
            nn.Conv2d(128, 64, 3, padding = 1), #50
            nn.ReLU(), #51
            nn.Conv2d(64, 64, 3, padding = 1), #52
            nn.ReLU(), #53
            
            nn.Conv2d(64, 3, 3, padding = 1), #54
            nn.ReLU(), #55
            
            nn.Conv2d(3, 3, 1), #56
            nn.Sigmoid() #57
            
            
        ])

    def forward(self, x):
                
        concat_map = {29: (28, 19), 36: (35, 13), 43: (42, 8), 50: (49, 3)}
        
        concat_tensors = {19: None, 
                         28: None,
                         13: None,
                         35: None,
                         8: None,
                         42: None,
                         3: None,
                         49: None}
        
        for i in range(len(self.layers)):
            
            # save tensors for later concatenation
            
            if i in concat_tensors.keys():
                concat_tensors[i] = self.layers[i](x)
            
            
            if i in concat_map.keys():
                x = torch.cat( (concat_tensors[concat_map[i][0]], concat_tensors[concat_map[i][1]] )
                                , dim = 1)
            
            x = self.layers[i](x)
            
            #print(i)
            
            #print(x.shape)
            
            #print('---------------')
            
        return x