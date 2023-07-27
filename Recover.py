import torch 
import pandas as pd
import torch.nn as nn
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn import preprocessing
from Arch import GeneratorState, GeneratorMeas, DiscriminatorState, DiscriminatorMeas
import datetime
from torch.utils.data import DataLoader,Dataset
import pickle




# loading the cycleGAN model
G_meas2state = GeneratorState()
G_meas2state.load_state_dict(torch.load('./G_measurement2state_21-07-2023_19-40-53.pth', map_location=torch.device('cpu')))
G_state2meas = GeneratorMeas()
G_state2meas.load_state_dict(torch.load('./G_state2measurement_21-07-2023_19-40-53.pth', map_location=torch.device('cpu')))
D_state = DiscriminatorState()
D_state.load_state_dict(torch.load('./D_state_21-07-2023_19-40-53.pth', map_location=torch.device('cpu')))
D_meas = DiscriminatorMeas()
D_meas.load_state_dict(torch.load('./D_measurement_21-07-2023_19-40-53.pth', map_location=torch.device('cpu')))


# state_data = scipy.io.loadmat('./attackData/st2.mat')
# state_mat = state_data['state']

# meas_data = scipy.io.loadmat('./attackData/meas2.mat')   
# meas_mat = meas_data['meas']

# meas_attacked_data = scipy.io.loadmat('./attackData/meas_attacked2.mat') 
# meas_attacked_mat = meas_attacked_data['meas_attacked']


min_max_scaler = preprocessing.MinMaxScaler()
y_data = pd.read_csv('measures.csv')
y = y_data.values
y_scaled = min_max_scaler.fit_transform(y)

# #########################################


# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, x_file, y_file, transform=None):
        self.x_data = pd.read_csv(x_file)
        self.y_data = pd.read_csv(y_file)
        # to fix dtype
        self.x_data = self.x_data.astype('float32')
        self.y_data = self.y_data.astype('float32')

        
        self.y = self.y_data.values
        min_max_scaler = preprocessing.MinMaxScaler()
        self.y_scaled = min_max_scaler.fit_transform(self.y)
        self.y_data = pd.DataFrame(self.y_scaled)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x_sample = self.x_data.iloc[idx, :].values
        y_sample = self.y_data.iloc[idx, :].values
        return x_sample, y_sample


# Create train dataset
train_dataset = CustomDataset('states.csv', 'measures.csv')

# Create train dataloader
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)



test_dataset = CustomDataset('X_test.csv', 'Y_test.csv')
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)


# #########################################

# meas_attacked_mat = min_max_scaler.transform(meas_attacked_mat)
# meas_mat = min_max_scaler.transform(meas_mat)




loss_criterion = nn.L1Loss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

G_state2meas.to(device)
G_meas2state.to(device)  
D_meas.to(device)
D_state.to(device)

# D_state.eval()
# D_meas.eval()
# G_meas2state.eval()
# G_state2meas.eval()

# rand = np.random.randint(0, 2000)
# st,meas = train_dataset.__getitem__(rand)


# meas_mat = meas 
# state_mat = st
# meas_attacked_mat = np.array(meas_mat)


# bus_idx = [4,5,11,12,13,14,40,50,70,53,64,23,64,86,92,99,103,106]
# lines_idx = [3,10,11,12,14,16,17,30,40,50,60,53,13,65,87,90,92,95,99]
# bus_idx = [4,5,11,12,13]
# lines_idx = [3,10,11,12,14]

# for idx in bus_idx:
#     meas_attacked_mat[idx] *= np.random.uniform(0.7, 1.3)
#     meas_attacked_mat[idx+118] *= np.random.uniform(0.7, 1.3)
    
# for idx in lines_idx:
#     meas_attacked_mat[idx+236] *= np.random.uniform(0.7, 1.3)
#     meas_attacked_mat[idx+422] *= np.random.uniform(0.7, 1.3)


# meas_temp = torch.tensor(meas_attacked_mat, dtype=torch.float32)
# meas_gt_tensor = torch.tensor(meas_mat, dtype=torch.float32)
# st = torch.tensor(state_mat, dtype=torch.float32)



# loss_gt = []
# loss_cycle = []



idx = 5
# read from pickle
with open('./AttackedPickle/meas_attacked_'+str(idx)+'.pkl', 'rb') as f:
    meas_temp = pickle.load(f)
with open('./AttackedPickle/meas_'+str(idx)+'.pkl', 'rb') as f:
    meas_gt_tensor = pickle.load(f)
with open('./AttackedPickle/st_'+str(idx)+'.pkl', 'rb') as f:
    st = pickle.load(f)
    
meas_temp = meas_temp.to(device)
meas_gt_tensor = meas_gt_tensor.to(device)
st = st.to(device)


recon_state_attacked = G_meas2state(meas_temp)
recon_meas_attacked = G_state2meas(recon_state_attacked)

recon_state_healthy = G_meas2state(meas_gt_tensor)
recon_meas_healthy = G_state2meas(recon_state_healthy)

cycle_state_attacked = G_meas2state(recon_meas_attacked)
cycle_state_healthy = G_meas2state(recon_meas_healthy)

print('difference between gt state and reconstructed state attacked: ', loss_criterion(st.reshape(-1,236), recon_state_attacked.reshape(-1,236)).item())
print('difference between gt state and reconstructed state healthy: ', loss_criterion(st.reshape(-1,236), recon_state_healthy.reshape(-1,236)).item())

print('difference between gt meas and reconstructed meas attacked: ', loss_criterion(meas_gt_tensor.reshape(-1,608), recon_meas_attacked.reshape(-1,608)).item())
print('difference between gt meas and reconstructed meas healthy: ', loss_criterion(meas_gt_tensor.reshape(-1,608), recon_meas_healthy.reshape(-1,608)).item())
print('difference between gt meas and attacked meas', loss_criterion(meas_gt_tensor.reshape(-1,608), meas_temp.reshape(-1,608)).item())


print('gt for cycle product for state attacked:', loss_criterion(st.reshape(-1,236), cycle_state_attacked.reshape(-1,236)).item())
print('gt for cycle product for state healthy:', loss_criterion(st.reshape(-1,236), cycle_state_healthy.reshape(-1,236)).item())

print('cycle for state attacked:', loss_criterion(recon_state_attacked.reshape(-1,236), cycle_state_attacked.reshape(-1,236)).item())
print('cycle for state healthy:', loss_criterion(recon_state_healthy.reshape(-1,236), cycle_state_healthy.reshape(-1,236)).item())
    

loss_main = []
loss_st = []

alpha = 0.01


state_attacked = G_meas2state(meas_temp).detach().cpu()
corr_state = state_attacked.detach().clone()
for i in range(2000):
    corr_state.requires_grad = True
    corr_meas = G_state2meas(corr_state)
    cycle_state = G_meas2state(corr_meas)
    loss = loss_criterion(corr_state.reshape(-1,236), cycle_state.reshape(-1,236))
    loss.backward()
    with torch.no_grad():
        # alpha *= 0.999
        grad_arr = np.array(corr_state.grad.detach().cpu())
        corr_state_arr = np.array(corr_state.detach().cpu())
        corr_state_arr -= alpha * grad_arr
        corr_state = torch.tensor(corr_state_arr, dtype=torch.float32)
        loss_main.append(loss.item())
        loss_st.append(loss_criterion(st.reshape(-1,236), corr_state.reshape(-1,236)).item())
        print(loss_criterion(corr_state.reshape(-1,236), cycle_state.reshape(-1,236)).item())
        if i % 20 == 0:
            print('epoch: ', i)
        # early stopping
        if np.linalg.norm(grad_arr) < 0.01:
            break

plt.figure()
plt.plot(loss_main)
plt.title('loss_main')
plt.figure()
plt.plot(loss_st)
plt.title('loss_st')
# print a line with a point
plt.axhline(y=loss_criterion(st.reshape(-1,236), recon_state_attacked.reshape(-1,236)).item(), color='r', linestyle='-')
plt.axhline(y= loss_criterion(st.reshape(-1,236), recon_state_healthy.reshape(-1,236)).item(), color='g', markersize=3)

plt.show()
