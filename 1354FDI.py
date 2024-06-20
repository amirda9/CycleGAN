import torch 
import pandas as pd
import torch.nn as nn
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn import preprocessing
import datetime
from torch.utils.data import DataLoader,Dataset
import pickle
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler


class DynamicNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation_func):
        super(DynamicNet, self).__init__()
        layers = []

        # Add first hidden layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(activation_func())

        # Add subsequent hidden layers (if any)
        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            layers.append(activation_func())

        # Add output layer
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

StateModel = DynamicNet(3228, [4096], 2708, nn.ReLU).to(device)
StateModel.load_state_dict(torch.load('./models/1354model_reverse.pth', map_location=device))

MeasModel = DynamicNet(2708, [1024,1024], 3228, nn.ReLU).to(device)
MeasModel.load_state_dict(torch.load('./models/1354model_forward.pth', map_location=device))



# Data preprocessing
df_y = pd.read_csv('./datasets/1354_volt.csv')
df_x = pd.read_csv('./datasets/1354_meas.csv')

x_train, x_test, y_train, y_test = train_test_split(df_x.values, df_y.values, test_size=0.2, random_state=42)


# Convert to torch tensors
x_train = torch.tensor(x_train).float().to(device)
x_test = torch.tensor(x_test).float().to(device)
y_train = torch.tensor(y_train).float().to(device)
y_test = torch.tensor(y_test).float().to(device)


rand = np.random.randint(0, 5000)
st = x_test[rand, :]
meas = y_test[rand, :]

meas_attacked = meas.clone()  # Clone to avoid modifying the original tensor


loss_criterion = nn.MSELoss()



loss_arr = []
StateModel.eval()
alpha = 0.1

# select 35 bus and 5 lines randomly
bus_idx = np.random.randint(0, 3228, 100)


for bus in bus_idx:
    meas_attacked[bus] *= np.random.normal(1, 0.1)
    meas_attacked[bus+118] *= np.random.normal(1, 0.1)

for i in range(5000): 
    # alpha *= 0.99
    meas_attacked.requires_grad = True
    loss = loss_criterion(meas_attacked, meas) - loss_criterion(StateModel(meas_attacked), StateModel(meas))
    loss_arr.append(loss.item())
    loss.backward()
    with torch.no_grad():
        grad_arr = meas_attacked.grad.data.cpu().numpy()
        meas_attacked = meas_attacked.cpu().numpy()

        for bus in bus_idx:
            meas_attacked[bus] -= alpha*grad_arr[bus]
        meas_attacked = torch.from_numpy(meas_attacked).to(device)

        if (loss_criterion(meas_attacked, meas).item() > 10):
            break
    
# plt.plot(loss_arr)
# plt.show()

            
# get from args
idx = sys.argv[1]
            
# save the attack with pickle

with open('./1354FDI/meas_'+str(idx)+'.pkl', 'wb') as f:
    pickle.dump(meas, f)

with open('./1354FDI/meas_attacked_'+str(idx)+'.pkl', 'wb') as f:
    pickle.dump(meas_attacked, f)
    

print(loss_criterion(meas_attacked, meas))
print(loss_criterion(StateModel(meas_attacked), StateModel(meas)))
# print(loss)