import torch 
import pandas as pd
import torch.nn as nn
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn import preprocessing

# Define the generators
class GeneratorMeas(nn.Module):
    def __init__(self):
        super(GeneratorMeas, self).__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose1d(236, 512, 114, 1, 0, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),

            nn.ConvTranspose1d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),

            nn.ConvTranspose1d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),

            nn.ConvTranspose1d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(True),

            nn.ConvTranspose1d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )       

        self.resnet = nn.Sequential(
            nn.Linear(1824+236, 1024),
            nn.ReLU(True), 
            nn.Linear(1024, 608),
        )
        
        
    def forward(self, x):
        y = self.net(x.view(-1, 236, 1))
        res = torch.cat((x.view(-1, 236), y.view(-1, 1824)), 1)
        res = self.resnet(res)
        # print(res.shape)
        return res
    
class GeneratorState(nn.Module):
    def __init__(self):
        super(GeneratorState, self).__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose1d(608, 512, 114, 1, 0, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),

            nn.ConvTranspose1d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),

            nn.ConvTranspose1d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),

            nn.ConvTranspose1d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(True),

            nn.ConvTranspose1d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        ) 
        
        self.resnet = nn.Sequential(
            nn.Linear(1824+608, 1024),   
            nn.ReLU(True),
            nn.Linear(1024, 236),
        )  
        
        
    def forward(self, x):
        y = self.net(x.view(-1, 608, 1))
        res = torch.cat((x.view(-1, 608), y.view(-1, 1824)), 1)
        res = self.resnet(res)
        # print(res.shape)
        return res
    
# Define the discriminators for measurements
class DiscriminatorMeas(nn.Module):
    def __init__(self):
        super(DiscriminatorMeas, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(608, 64, kernel_size=1, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size 912
            nn.Conv1d(64, 128, kernel_size=1, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # state size 456
            nn.Conv1d(128, 256, kernel_size=1,
                      stride=2, padding=0, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # state size 228
            nn.Conv1d(256, 512, kernel_size=1,
                      stride=2, padding=0, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # state size 114
            nn.Conv1d(512, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Flatten(),
        )
        
        self.resnet = nn.Sequential(
            nn.Linear(256+608, 256),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        
        
    def forward(self, x):
        y =  self.net(x.view(-1, 608,1))
        res = torch.cat((x.view(-1, 608), y.view(-1, 256)), 1)
        res = self.resnet(res)
        # print(y.shape)
        return res
    
# Define the discriminators for states
class DiscriminatorState(nn.Module):
    def __init__(self):
        super(DiscriminatorState, self).__init__()
        self.net = nn.Sequential(
            # input 1824
            nn.Conv1d(236, 64, kernel_size=1, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size 912
            nn.Conv1d(64, 128, kernel_size=1, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # state size 456
            nn.Conv1d(128, 256, kernel_size=1,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # state size 228
            nn.Conv1d(256, 512, kernel_size=1,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # state size 114
            nn.Conv1d(512, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Flatten(),
        )
        self.resnet = nn.Sequential(
            nn.Linear(512+236, 256),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        
    def forward(self, x):
        y = self.net(x.view(-1, 236,1))
        res = torch.cat((x.view(-1, 236), y.view(-1, 512)), 1)
        res = self.resnet(res)
        # print(y.shape)
        return res
