import itertools
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np


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

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


hidden_layer_configurations = [
    [1024],
    [2048],
    [4096],
    [1024,1024],
    [2048,1024],
    [2048,2048],
    [4096,4096],
    [2048, 1024, 512],
    [4096, 2048, 1024],
    [8192, 4096, 2048],
    [4096, 2048],
    [8192, 4096],
    [8192, 4096, 2048],
    [1024,512],
    [512,1024]
]

# activation_functions = [nn.ReLU, nn.Sigmoid, nn.Tanh]
activation_functions = [nn.ReLU, nn.Tanh]
loss_functions = [F.mse_loss, F.l1_loss]
learning_rates = [0.01, 0.001]
epochs = [10000, 2000]
lr_schedulers = [('StepLR', {'step_size': 1000, 'gamma': 0.9}), ('ExponentialLR', {'gamma': 0.95})]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Data preprocessing
df_y = pd.read_csv('./datasets/1354_volt.csv')
df_x = pd.read_csv('./datasets/1354_meas.csv')

# Data preprocessing (assuming you have already loaded your data into df_x and df_y)
x_train, x_test, y_train, y_test = train_test_split(df_x.values, df_y.values, test_size=0.2, random_state=42)

# Convert to torch tensors
x_train = torch.tensor(x_train).float().to(device)
x_test = torch.tensor(x_test).float().to(device)
y_train = torch.tensor(y_train).float().to(device)
y_test = torch.tensor(y_test).float().to(device)

# DataLoader
train_batch = DataLoader(TensorDataset(x_train, y_train), batch_size=512, shuffle=True)
test_batch = DataLoader(TensorDataset(x_test, y_test), batch_size=128, shuffle=True)

# Grid search
results = []

i = 0
for hidden_layers, activation, loss_fn, lr, epoch, (scheduler_name, scheduler_params) in itertools.product(hidden_layer_configurations, activation_functions, loss_functions, learning_rates, epochs, lr_schedulers):
    i += 1
    if i > 0 :
        print(hidden_layers, activation, loss_fn, lr, epoch, (scheduler_name, scheduler_params))
        
        # Initialize model
        model = DynamicNet(3228, hidden_layers, 2708, activation).to(device)
        model.apply(init_weights)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Initialize LR scheduler
        if scheduler_name == 'StepLR':
            scheduler = optim.lr_scheduler.StepLR(optimizer, **scheduler_params)
        elif scheduler_name == 'ExponentialLR':
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, **scheduler_params)

        best_test_loss = float('inf')
        epochs_without_improvement = 0
        max_no_improvement_epochs = 2000

        # Training loop
        for ep in range(epoch):
            model.train()
            for x, y in train_batch:
                optimizer.zero_grad()
                output = model(x)
                loss = loss_fn(output, y)
                loss.backward()
                optimizer.step()
            scheduler.step()
            
            if ep % 10 == 0:  # Evaluate every 10 epochs
                model.eval()
                with torch.no_grad():
                    total_test_loss = 0
                    for x, y in test_batch:
                        output = model(x)
                        test_loss = F.mse_loss(output, y).item()
                        total_test_loss += test_loss
                    avg_test_loss = total_test_loss / len(test_batch)
                    print(f'Epoch {ep}: Train Loss: {loss.item()}, Test Loss: {avg_test_loss}')
                    
                # Check for early stopping
                if avg_test_loss < best_test_loss:
                    best_test_loss = avg_test_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 10

                if epochs_without_improvement >= max_no_improvement_epochs:
                    print(f'Stopping early at epoch {ep} due to no improvement.')
                    break

        # save the model to disk
        torch.save(model.state_dict(), f'./models/1354model_reverse.pth')
        