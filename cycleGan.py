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

# Define the generators
class GeneratorMeas(nn.Module):
    def __init__(self):
        super(GeneratorMeas, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(236, 256),
            nn.ReLU(),
            # nn.Linear(256, 128),
            # nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            # nn.Linear(256, 512),
            # nn.ReLU(),
            nn.Linear(256, 608)
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
class GeneratorState(nn.Module):
    def __init__(self):
        super(GeneratorState, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(608, 512),
            nn.ReLU(),
            # nn.Linear(512, 256),
            # nn.ReLU(),
            nn.Linear(512, 256)
        )
        self.decoder = nn.Sequential(
            # nn.Linear(128, 256),
            # nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 236)
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    


# Define the discriminators for measurements
class DiscriminatorMeas(nn.Module):
    def __init__(self):
        super(DiscriminatorMeas, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(608, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.net(x)
    
# Define the discriminators for states
class DiscriminatorState(nn.Module):
    def __init__(self):
        super(DiscriminatorState, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(236, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.net(x)

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


# Define the loss functions binary cross entropy and MSE
adversarial_loss = nn.BCELoss()
cycle_consistency_loss = nn.MSELoss()

# Define the generators and discriminators
G_state2measurement = GeneratorMeas()
G_measurement2state = GeneratorState()
D_measurement = DiscriminatorMeas()
D_state = DiscriminatorState()

# configs
learning_rate = 0.01
num_epochs = 1000
lambda_cycle = 1


# Define the optimizers
optimizer_G = optim.Adam(list(G_state2measurement.parameters()) + list(G_measurement2state.parameters()) + list(D_measurement.parameters()) + list(D_state.parameters()), lr=learning_rate, betas=(0.5, 0.999))
optimizer_D_measurement = optim.Adam(D_measurement.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_D_state = optim.Adam(D_state.parameters(), lr=learning_rate, betas=(0.5, 0.999))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# nets to gpu
G_state2measurement.to(device)
G_measurement2state.to(device)  

D_measurement.to(device)
D_state.to(device)


# Define the training loop
l_G_measurement = []
l_G_state = []
l_D_measurement = []
l_D_state = []
l_cycle = []
l_total = []
for epoch in range(num_epochs):
    if epoch <100:
        lambda_cycle = 0.1
    elif epoch < 500:
        lambda_cycle = 0.2
    else: 
        lambda_cycle = 0.5
    l_G_measurement_raw = []
    l_G_state_raw = []
    l_D_measurement_raw = []
    l_D_state_raw = []
    l_cycle_raw = []
    l_total_raw = []
    for i, (state_data, measurement_data) in enumerate(train_dataloader):
        
        # Train the discriminators
        D_measurement.zero_grad()
        D_state.zero_grad()
        
        # Train with real data
        real_measurement = measurement_data.to(device)
        real_state = state_data.to(device)
        label_real = torch.ones(real_measurement.size(0), 1).to(device)
        label_fake = torch.zeros(real_measurement.size(0), 1).to(device)
        
        output_real_measurement = D_measurement(real_measurement)
        output_real_state = D_state(real_state)
        loss_real_measurement = adversarial_loss(output_real_measurement, label_real)
        loss_real_state = adversarial_loss(output_real_state, label_real)
        
        
        # Train with fake data
        fake_measurement = G_state2measurement(real_state)
        fake_state = G_measurement2state(real_measurement)
        
        output_fake_measurement = D_measurement(fake_measurement.detach())
        output_fake_state = D_state(fake_state.detach())
        loss_fake_measurement = adversarial_loss(output_fake_measurement, label_fake)
        loss_fake_state = adversarial_loss(output_fake_state, label_fake)
        
        
        
        loss_D_measurement = (loss_real_measurement + loss_fake_measurement) 
        loss_D_state = (loss_real_state + loss_fake_state)
        
        optimizer_D_measurement.step()
        optimizer_D_state.step()

        
        # Train the generators
        G_state2measurement.zero_grad()
        G_measurement2state.zero_grad()
        
        # Forward pass
        fake_measurement = G_state2measurement(real_state)
        fake_state = G_measurement2state(real_measurement)
        
        output_fake_measurement = D_measurement(fake_measurement)
        output_fake_state = D_state(fake_state)
        
        # Adversarial loss
        loss_G_state2measurement = adversarial_loss(output_fake_measurement, label_real)
        loss_G_measurement2state = adversarial_loss(output_fake_state, label_real)
        
        # Cycle consistency loss
        reconstructed_measurement = G_state2measurement(fake_state)
        reconstructed_state = G_measurement2state(fake_measurement)
        
        loss_cycle_state = cycle_consistency_loss(reconstructed_state, real_state)
        loss_cycle_measurement = cycle_consistency_loss(reconstructed_measurement, real_measurement)
        
        loss_cycle = loss_cycle_state + loss_cycle_measurement
        
        # Total loss
        loss_G = loss_G_state2measurement + loss_G_measurement2state + lambda_cycle * loss_cycle
        loss_G.backward()
        
        optimizer_G.step()

        # save the loss
        l_G_measurement_raw.append(loss_G_measurement2state.detach().cpu().numpy())
        l_G_state_raw.append(loss_G_state2measurement.detach().cpu().numpy())
        l_cycle_raw.append(loss_cycle.detach().cpu().numpy())
        l_total_raw.append(loss_G.detach().cpu().numpy())
        l_D_measurement_raw.append(loss_real_measurement.detach().cpu().numpy() + loss_fake_measurement.detach().cpu().numpy())
        l_D_state_raw.append(loss_real_state.detach().cpu().numpy() + loss_fake_state.detach().cpu().numpy())
        
        # Print losses
        if i % 100 == 0:
            print('[Epoch %d/%d] [Batch %d/%d] [D loss: %.4f] [G loss: %.4f] [Cycle loss: %.4f]' %
                  (epoch, num_epochs, i, len(train_dataloader),
                   (loss_D_measurement + loss_D_state).item(), loss_G.item(), loss_cycle.item()))

    l_G_measurement.append(np.mean(l_G_measurement_raw))
    l_G_state.append(np.mean(l_G_state_raw))
    l_cycle.append(np.mean(l_cycle_raw))
    l_total.append(np.mean(l_total_raw))
    l_D_measurement.append(np.mean(l_D_measurement_raw))
    l_D_state.append(np.mean(l_D_state_raw))
    

# make a dated folder
now = datetime.now()
dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")


# save the model with the date 
torch.save(G_state2measurement.state_dict(), './results/G_state2measurement_' + dt_string + '.pth')
torch.save(G_measurement2state.state_dict(), './results/G_measurement2state_' + dt_string + '.pth')
torch.save(D_measurement.state_dict(), './results/D_measurement_' + dt_string + '.pth')
torch.save(D_state.state_dict(), './results/D_state_' + dt_string + '.pth')

# pickle.dump(l_G_measurement, open('l_G_measurement.pkl', 'wb')) 
# pickle.dump(l_G_state, open('l_G_state.pkl', 'wb'))
# pickle.dump(l_D_measurement, open('l_D_measurement.pkl', 'wb'))
# pickle.dump(l_D_state, open('l_D_state.pkl', 'wb'))
# pickle.dump(l_cycle, open('l_cycle.pkl', 'wb'))
# pickle.dump(l_total, open('l_total.pkl', 'wb'))


# save the plot for six losses with the date
plt.figure()
plt.plot(l_G_measurement)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Generator loss for measurement')
plt.savefig('./results/G_measurement_{}.png'.format(dt_string))
plt.close()
plt.figure()
plt.plot(l_G_state)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Generator loss for state')   
plt.savefig('./results/G_state_{}.png'.format(dt_string))
plt.close()
plt.figure()
plt.plot(l_D_measurement)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Discriminator loss for measurement')
plt.savefig('./results/D_measurement_{}.png'.format(dt_string))
plt.close()
plt.figure()
plt.plot(l_D_state)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Discriminator loss for state')
plt.savefig('./results/D_state_{}.png'.format(dt_string))
plt.close()
plt.figure()
plt.plot(l_cycle)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Cycle consistency loss')
plt.savefig('./results/cycle_{}.png'.format(dt_string))
plt.close()
plt.figure()
plt.plot(l_total)
plt.xlabel('Epoch')
plt.ylabel('Loss')  
plt.title('Total loss')
plt.savefig('./results/total_{}.png'.format(dt_string))
plt.close()
