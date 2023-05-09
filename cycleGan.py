import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
import pandas as pd

# Define the generators
class GeneratorMeas(nn.Module):
    def __init__(self):
        super(GeneratorMeas, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(236, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
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
            nn.Linear(128, 256),
            nn.ReLU(),
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
            nn.Linear(608, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
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

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x_sample = self.x_data.iloc[idx, :].values
        y_sample = self.y_data.iloc[idx, :].values
        return x_sample, y_sample


# Create train dataset
train_dataset = CustomDataset('states.csv', 'measures.csv')

# Create train dataloader
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)


# Define the loss functions
adversarial_loss = nn.BCELoss()
cycle_consistency_loss = nn.L1Loss()

# Define the generators and discriminators
G_state2measurement = GeneratorMeas()
G_measurement2state = GeneratorState()
D_measurement = DiscriminatorMeas()
D_state = DiscriminatorState()

# configs
learning_rate = 0.01
num_epochs = 100

# Define the optimizers
optimizer_G = optim.Adam(list(G_state2measurement.parameters()) + list(G_measurement2state.parameters()), lr=learning_rate, betas=(0.5, 0.999))
optimizer_D_measurement = optim.Adam(D_measurement.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_D_state = optim.Adam(D_state.parameters(), lr=learning_rate, betas=(0.5, 0.999))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the training loop
for epoch in range(num_epochs):
    for i, (state_data, measurement_data) in enumerate(train_dataloader):
        
        # Train the discriminators
        D_measurement.zero_grad()
        D_state.zero_grad()
        
        # Train with real data
        real_measurement = measurement_data.to(device)
        real_state = state_data.to(device)
        label_real = torch.ones(real_measurement.size(0), 1).to(device)
        label_fake = torch.zeros(real_measurement.size(0), 1).to(device)
        
        output_real_measurement = D_measurement(real_measurement).to(device)
        output_real_state = D_state(real_state).to(device)
        loss_real_measurement = adversarial_loss(output_real_measurement, label_real)
        loss_real_state = adversarial_loss(output_real_state, label_real)
        
        loss_real = loss_real_measurement + loss_real_state
        loss_real.backward()
        
        # Train with fake data
        fake_measurement = G_state2measurement(real_state)
        fake_state = G_measurement2state(real_measurement)
        
        output_fake_measurement = D_measurement(fake_measurement.detach())
        output_fake_state = D_state(fake_state.detach())
        loss_fake_measurement = adversarial_loss(output_fake_measurement, label_fake)
        loss_fake_state = adversarial_loss(output_fake_state, label_fake)
        
        loss_fake = loss_fake_measurement + loss_fake_state
        loss_fake.backward()
        
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
        
        # Print losses
        if i % 100 == 0:
            print('[Epoch %d/%d] [Batch %d/%d] [D loss: %.4f] [G loss: %.4f] [Cycle loss: %.4f]' %
                  (epoch, num_epochs, i, len(train_dataloader),
                   (loss_real + loss_fake).item(), loss_G.item(), loss_cycle.item()))
