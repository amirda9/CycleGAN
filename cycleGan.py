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
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 608)
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
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 200),
            nn.ReLU(),
            nn.Linear(200, 236),
            # nn.ReLU(),
            # nn.Linear(256, 236)
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
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        
        
    def forward(self, x):
        return self.net(x)
    
# Define the discriminators for states
class DiscriminatorState(nn.Module):
    def __init__(self):
        super(DiscriminatorState, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(236, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        
    def forward(self, x):
        return self.net(x)

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

G_measurement2state.apply(weights_init) 
G_state2measurement.apply(weights_init)
D_measurement.apply(weights_init)
D_state.apply(weights_init)


# configs
learning_rate = 0.001
num_epochs = 200
lambda_cycle = 1


# Define the optimizers
# optimizer_G_measurement = optim.Adam(G_state2measurement.parameters(), lr=learning_rate, betas=(0.5, 0.999))
# optimizer_G_state = optim.Adam(G_measurement2state.parameters(), lr=learning_rate, betas=(0.5, 0.999))
# optimizer_D_measurement = optim.Adam(D_measurement.parameters(), lr=learning_rate, betas=(0.5, 0.999))
# optimizer_D_state = optim.Adam(D_state.parameters(), lr=learning_rate, betas=(0.5, 0.999))

optimizer_G = optim.Adam(list(G_state2measurement.parameters()) + list(G_measurement2state.parameters()), lr=learning_rate, betas=(0.5, 0.999))
optimizer_D = optim.Adam(list(D_measurement.parameters()) + list(D_state.parameters()), lr=learning_rate, betas=(0.5, 0.999))
optim_total = optim.Adam(list(G_state2measurement.parameters()) + list(G_measurement2state.parameters()) + list(D_measurement.parameters()) + list(D_state.parameters()), lr=learning_rate, betas=(0.5, 0.999))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# nets to gpu
G_state2measurement.to(device)
G_measurement2state.to(device)  
D_measurement.to(device)
D_state.to(device)

# lr schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda epoch: 0.95 ** epoch)
lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=lambda epoch: 0.95 ** epoch)



# Define the training loop
l_G = []
l_D = []
for epoch in range(num_epochs):
    
    D_measurement.train()
    D_state.train()
    G_state2measurement.train()
    G_measurement2state.train()
        
    # lambda_cycle = 0.1 * (epoch/100)
        
    
    loss_G_raw = []
    loss_D_raw = []
    for i, (state_data, measurement_data) in enumerate(train_dataloader):
        D_measurement.zero_grad()
        D_state.zero_grad()
        G_state2measurement.zero_grad()
        G_measurement2state.zero_grad()
        
        real_measurement = measurement_data.to(device)
        real_state = state_data.to(device)
        
        label_real = torch.ones(real_measurement.size(0), 1).to(device)
        label_fake = torch.zeros(real_measurement.size(0), 1).to(device)
        
        fake_measurement = G_state2measurement(real_state)
        fake_state = G_measurement2state(real_measurement)
        
        
        loss_D_measurement = adversarial_loss(D_measurement(real_measurement), label_real) + adversarial_loss(D_measurement(fake_measurement), label_fake)
        loss_D_state = adversarial_loss(D_state(real_state), label_real) + adversarial_loss(D_state(fake_state), label_fake)
        loss_D = loss_D_measurement + loss_D_state
        loss_D.backward()
        optimizer_D.step()
        
        label_real = torch.ones(real_measurement.size(0), 1).to(device)
        label_fake = torch.zeros(real_measurement.size(0), 1).to(device)
        
        fake_measurement = G_state2measurement(real_state)
        recov_state = G_measurement2state(fake_measurement)
        fake_state = G_measurement2state(real_measurement)
        recov_measurement = G_state2measurement(fake_state)
        
        
        
        
        loss_cycle = cycle_consistency_loss(recov_state, real_state) + cycle_consistency_loss(recov_measurement, real_measurement)
        loss_identity = cycle_consistency_loss(fake_measurement, real_measurement) + cycle_consistency_loss(fake_state, real_state)
        loss_G_measurement = adversarial_loss(D_measurement(fake_measurement), label_real)
        loss_G_state = adversarial_loss(D_state(fake_state), label_real)
        loss_G = loss_G_measurement + loss_G_state + lambda_cycle * loss_cycle + lambda_cycle * loss_identity
        
        loss_G.backward()
        optimizer_G.step()
        
        if epoch % 5 == 0:
            fake_measurement = G_state2measurement(real_state)
            fake_state = G_measurement2state(real_measurement)
            recov_measurement = G_state2measurement(fake_state)
            recov_state = G_measurement2state(fake_measurement)
            loss_cycle = cycle_consistency_loss(recov_state, real_state) + cycle_consistency_loss(recov_measurement, real_measurement)
            loss_identity = cycle_consistency_loss(fake_measurement, real_measurement) + cycle_consistency_loss(fake_state, real_state)
            loss_G_measurement = adversarial_loss(D_measurement(fake_measurement), label_real)
            loss_G_state = adversarial_loss(D_state(fake_state), label_real)
            loss_G = loss_G_measurement + loss_G_state + lambda_cycle * loss_cycle + lambda_cycle * loss_identity
            loss_D_measurement = adversarial_loss(D_measurement(real_measurement), label_real) + adversarial_loss(D_measurement(fake_measurement), label_fake)
            loss_D_state = adversarial_loss(D_state(real_state), label_real) + adversarial_loss(D_state(fake_state), label_fake)
            loss_D = loss_D_measurement + loss_D_state
            loss_total = loss_G + loss_D
            loss_total.backward()
            optim_total.step()
            
            
          
        
        # train measurement discriminator
        # loss_real_measurement = adversarial_loss(D_measurement(real_measurement), label_real)
        # loss_fake_measurement = adversarial_loss(D_measurement(fake_measurement ), label_fake)
        # loss_D_measurement = (loss_real_measurement + loss_fake_measurement) 
        # loss_D_measurement.backward()
        # optimizer_D_measurement.step()
        
        # train state discriminator
        # loss_real_state = adversarial_loss(D_state(real_state), label_real)
        # loss_fake_state = adversarial_loss(D_state(fake_state), label_fake)
        # loss_D_state = (loss_real_state + loss_fake_state)
        # loss_D_state.backward()
        # optimizer_D_state.step()
        
        # train state2measurement generator using adversarial and cycle loss
        # loss_G_state2measurement = adversarial_loss(D_measurement(fake_measurement), label_real) + lambda_cycle * cycle_consistency_loss(recov_state, real_state)
        # loss_G_state2measurement.backward()
        # optimizer_G_measurement.step()
        
        # train measurement2state generator
        # loss_G_measurement2state = adversarial_loss(D_state(fake_state), label_real) + lambda_cycle * cycle_consistency_loss(recov_measurement, real_measurement)
        # loss_G_measurement2state.backward()

        
        
        
        # Print losses
        if i % 100 == 0:
            print('[Epoch %d/%d] [Batch %d/%d] [D loss: %.4f] [G loss: %.4f] [Cycle loss: %.4f]' %
                (epoch, num_epochs, i, len(train_dataloader),
                (loss_D_measurement + loss_D_state).item(), loss_G.item(), loss_cycle.item()))
            
        # save the loss
        loss_G_raw.append(loss_G.detach().cpu().numpy())
        loss_D_raw.append((loss_D.detach().cpu().numpy()))
        


    l_G.append(np.mean(loss_G_raw))
    l_D.append(np.mean(loss_D_raw))
    lr_scheduler_D.step()
    lr_scheduler_G.step()
# make a dated folder
now = datetime.now()
dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")


# save the model with the date 
# torch.save(G_state2measurement.state_dict(), './results/G_state2measurement_' + dt_string + '.pth')
# torch.save(G_measurement2state.state_dict(), './results/G_measurement2state_' + dt_string + '.pth')
# torch.save(D_measurement.state_dict(), './results/D_measurement_' + dt_string + '.pth')
# torch.save(D_state.state_dict(), './results/D_state_' + dt_string + '.pth')

# save the plot for six losses with the date
plt.figure()
plt.plot(l_G)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Generator loss')
plt.savefig('./results/G_{}.png'.format(dt_string))
plt.close()
plt.figure()
plt.plot(l_D)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Discriminator loss')
plt.savefig('./results/D_{}.png'.format(dt_string))
plt.close()
plt.close('all')