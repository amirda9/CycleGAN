import torch 
import pandas as pd
import torch.nn as nn
import numpy as np

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
    


# loading the cycleGAN model
G_meas2state = GeneratorState()
G_meas2state.load_state_dict(torch.load('./recovery/G_measurement2state_07-06-2023_13-17-43.pth'))
G_state2meas = GeneratorMeas()
G_state2meas.load_state_dict(torch.load('./recovery/G_state2measurement_07-06-2023_13-17-43.pth'))
D_state = torch.load('./recovery/D_state_07-06-2023_13-17-43.pth')
D_meas = torch.load('./recovery/D_measurement_07-06-2023_13-17-43.pth')

# inporting the data
states = pd.read_csv('./attackData/st.csv')
meas = pd.read_csv('./attackData/meas.csv')
meas_attacked = pd.read_csv('./attackData/meas_attacked.csv')



loss_criterion = nn.L1Loss()

# loss_init = loss_criterion(meas_attacked, meas)

# print(loss_init)


bus_idx = [4,5,11,12,13]
lines_idx = [3,10,11,12,14,16]
meas_temp_attacked = meas_attacked.copy()
meas_attacked_tensor = torch.tensor(meas_attacked, dtype=torch.float32)
meas_gt_tensor = torch.tensor(meas, dtype=torch.float32)
optimizer = torch.optim.Adam([meas_attacked_tensor], lr=0.01)
for i in range(10):
    optimizer.zero_grad()
    meas_tensor = torch.tensor(meas_temp_attacked, dtype=torch.float32)
    recon_state = G_meas2state(meas_tensor)
    recon_meas = G_state2meas(recon_state)
    loss_temp = loss_criterion(recon_meas, meas_gt_tensor.reshape(-1, 608))
    loss_temp.backward()
    print(loss_temp)
    
    
    