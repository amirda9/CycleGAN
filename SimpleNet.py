import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn import preprocessing
import numpy as np
import os
from datetime import datetime
import itertools
from sklearn.model_selection import train_test_split

from Arch import GeneratorState, GeneratorMeas, DiscriminatorState, DiscriminatorMeas

# weight initialization
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif isinstance(m, nn.Conv2d): 
        nn.init.xavier_uniform_.weight.data(m.weight)
        m.bias.data.fill_(0.01)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    else:
        pass

# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, x_file, y_file, transform=None):
        self.x_data = x_file
        self.y_data = y_file
        # to fix dtype
        self.x_data = self.x_data.astype('float32')
        self.y_data = self.y_data.astype('float32')
        # self.x = self.x_data.values
        # min_max_scaler = preprocessing.MinMaxScaler()
        # self.x_scaled = min_max_scaler.fit_transform(self.x)    
        # self.x_data = pd.DataFrame(self.x_scaled)
        # self.y = self.y_data.values
        # min_max_scaler = preprocessing.MinMaxScaler()
        # self.y_scaled = min_max_scaler.fit_transform(self.y)
        # self.y_data = pd.DataFrame(self.y_scaled)
        

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x_sample = self.x_data.iloc[idx, :].values
        y_sample = self.y_data.iloc[idx, :].values
        return x_sample, y_sample


# train test split 
states_pd = pd.read_csv('states.csv')
meas_pd = pd.read_csv('measures.csv')
states_test, states_train, meas_test, meas_train = train_test_split(states_pd, meas_pd, test_size=0.2, random_state=42)

# Create train dataset
train_dataset = CustomDataset(states_train, meas_train)
test_dataset = CustomDataset(states_test, meas_test)

# Create train dataloader
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)


# Define the loss functions binary cross entropy and MSE
loss_criteria = nn.MSELoss()

# Define the generators and discriminators
G_measurement2state = GeneratorState()

G_measurement2state.apply(weights_init) 


# configs
learning_rate = 0.0001
num_epochs = 20


optimizer_G = torch.optim.Adam(G_measurement2state.parameters(), lr=learning_rate, betas=(0.5, 0.999))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

G_measurement2state.to(device)  



loss_main = []
loss_eval = []
for epoch in range(num_epochs):
    
    print(epoch)
    for i, (state_data, measurement_data) in enumerate(train_dataloader):
        print(i)
        real_measurement = measurement_data.to(device)
        real_state = state_data.to(device)
        
        G_measurement2state.train()
        optimizer_G.zero_grad()
        
        # identity loss
        
        loss_identity_state = loss_criteria(G_measurement2state(real_measurement), real_state)
        
        loss_identity = loss_identity_state
        loss_identity.backward()
        optimizer_G.step()
        loss_main.append(loss_identity.item())
        
        # evaluate the model
        
        
for i, (state_data, measurement_data) in enumerate(test_dataloader):
    print('eval', i)
    real_measurement = measurement_data.to(device)
    real_state = state_data.to(device)
    
    G_measurement2state.eval()

    with torch.no_grad():
        loss_identity_state = loss_criteria(G_measurement2state(real_measurement), real_state)
        loss_eval.append(loss_identity.item())


        
now = datetime.now()
dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")

torch.save(G_measurement2state.state_dict(), './G_{}.pth'.format(dt_string))

# save the plot for six losses with the date
plt.figure()
plt.plot(loss_main)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Network loss')
plt.savefig('./G_{}.png'.format(dt_string))
plt.figure()
plt.plot(loss_eval)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Network Eval loss')
plt.savefig('./G_eval_{}.png'.format(dt_string))
plt.close('all')
