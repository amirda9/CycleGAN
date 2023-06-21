import torch 
import pandas as pd
import torch.nn as nn
import numpy as np
import scipy.io
import matplotlib.pyplot as plt


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


# loading the cycleGAN model
G_meas2state = GeneratorState()
G_meas2state.load_state_dict(torch.load('./recovery/G_measurement2state_07-06-2023_13-17-43.pth'))
G_state2meas = GeneratorMeas()
G_state2meas.load_state_dict(torch.load('./recovery/G_state2measurement_07-06-2023_13-17-43.pth'))
D_state = DiscriminatorState()
D_state.load_state_dict(torch.load('./recovery/D_state_07-06-2023_13-17-43.pth'))
D_meas = DiscriminatorMeas()
D_meas.load_state_dict(torch.load('./recovery/D_measurement_07-06-2023_13-17-43.pth'))


state_data = scipy.io.loadmat('./attackData/st.mat')
state = state_data['state']

meas_data = scipy.io.loadmat('./attackData/meas.mat')   
meas = meas_data['meas']

meas_attacked_data = scipy.io.loadmat('./attackData/meas_attacked.mat') 
meas_attacked = meas_attacked_data['meas_attacked']

st_data = scipy.io.loadmat('./attackData/st.mat')
st = st_data['state']



# importing the data
# states = pd.read_csv('./attackData/st.csv')
# meas = pd.read_csv('./attackData/meas.csv')
# meas_attacked = pd.read_csv('./attackData/meas_attacked.csv')


# meas = np.array(meas.columns.values.reshape(-1,608))
# meas_attacked = np.array(meas_attacked.columns.values.reshape(-1,608))

loss_criterion = nn.L1Loss()

# loss_init = loss_criterion(meas_attacked, meas)
# print columns
print(meas_attacked)


bus_idx = [4,5,11,12,13]
lines_idx = [3,10,11,12,14,16]
meas_attacked_tensor = torch.tensor(meas_attacked, dtype=torch.float32)
meas_gt_tensor = torch.tensor(meas, dtype=torch.float32)
meas_temp = meas_attacked_tensor
# optimizer = torch.optim.Adam([meas_temp], lr=0.01)
print(meas_attacked_tensor)
loss_gt = []
loss_cycle = []
for i in range(500):
    # optimizer.zero_grad()
    meas_temp.requires_grad = True
    recon_state = G_meas2state(meas_temp)
    recon_meas = G_state2meas(recon_state)
    # loss_temp = loss_criterion(recon_meas, meas_temp.reshape(-1,608)) + loss_criterion(recon_state, st.reshape(-1,236))
    loss_temp = loss_criterion(recon_meas, meas_temp.reshape(-1,608)) - D_meas(recon_meas) + loss_criterion(recon_state, st.reshape(-1,236)) - D_state(recon_state)
    loss_temp.backward()
    # optimizer.step()
    with torch.no_grad():
        meas_temp_arr = np.array(meas_temp.detach().numpy())
        grad_arr = np.array(meas_temp.grad.detach().numpy())
        meas_temp_arr = meas_temp_arr - 0.01*grad_arr 
        # for idx in bus_idx:
        #     meas_temp_arr[0,idx] = meas_temp_arr[0,idx] - 0.3*grad_arr[0,idx]
        #     meas_temp_arr[0,idx+118] = meas_temp_arr[0,idx+118] - 0.3*grad_arr[0,idx+118]
        # for idx in lines_idx:
        #     meas_temp_arr[0,idx+236] = meas_temp_arr[0,idx+236] - 0.3*grad_arr[0,idx+236]
        #     meas_temp_arr[0,idx+422] = meas_temp_arr[0,idx+422] - 0.3*grad_arr[0,idx+422]
        meas_temp = torch.tensor(meas_temp_arr, dtype=torch.float32)
        loss_gt.append(loss_criterion(meas_gt_tensor, meas_temp.reshape(-1,608)))
        loss_cycle.append(loss_criterion(recon_meas, meas_temp.reshape(-1,608)))
        # print(loss_criterion(meas_gt_tensor, meas_temp.reshape(-1,608)),loss_criterion(recon_meas, meas_temp.reshape(-1,608)))
        # pass
     
plt.figure()   
plt.plot(loss_gt)
plt.title('L1 loss of GT and attacked measurement')
plt.savefig('./results/loss_gt.png')
plt.figure()
plt.plot(loss_cycle)
plt.title('L1 loss of cycle consistency')
plt.savefig('./results/loss_cycle.png')
# plt.show()
plt.close()
    
    
    