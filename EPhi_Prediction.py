import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
import re
import random
import AEP

def file_to_dict(filename):
    data_dict = {}
    # Regex to match complex numbers
    complex_num_pattern = re.compile(r'-?\d+\.\d+[e\-\+\d\s]+[+-]\d+\.[e\-\+\d\s]+j')

    with open(filename, 'r') as file:
        current_key = None
        values = []
        for line_number, line in enumerate(file):
            if line_number % 91 == 0:  # Every 91 lines a new block starts
                if current_key is not None:
                    data_dict[current_key] = values
                key_part, complex_numbers = re.split(r'\),\[', line)
                key_part += ')'
                current_key = tuple(map(float, re.findall(r"[-+]?\d*\.\d+|\d+", key_part)))
                values = []
                # Process initial line of complex numbers
                if complex_numbers:
                    complex_matches = complex_num_pattern.findall(complex_numbers)
                    values.extend([complex(num.replace(' ','')) for num in complex_matches])
            else:
                # Continue collecting values
                if line:  # Ensure it's not empty
                    complex_matches = complex_num_pattern.findall(line)
                    values.extend([complex(num.replace(' ','')) for num in complex_matches])

        # Add the last key-value pair
        if current_key is not None:
            data_dict[current_key] = values

    return data_dict

class rEPhiDataset:
    def __init__(self, file_path, transform=None, target_transform=None):
        self.EPhi_positions = file_to_dict(file_path)
        self.positions = torch.tensor([k for k in self.EPhi_positions.keys()], dtype=torch.float)
        self.EPhi = torch.tensor([v for v in self.EPhi_positions.values()], dtype=torch.cfloat)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.EPhi_positions)
    
    def __getitem__(self, idx):
        position = self.positions[idx]
        position = position.view(1, -1)
        EPhi = self.EPhi[idx]
        if self.transform:
            EPhi = self.transform(EPhi)
        if self.target_transform:
            position = self.target_transform(position)
        return position, EPhi

class ImprovedNetwork(nn.Module):
    def __init__(self):
        super(ImprovedNetwork, self).__init__()
        conv1=8
        conv2=64
        conv3=512
        self.relu=nn.ReLU()
        self.conv1=nn.Conv1d(1,conv1,3,padding=1)
        self.conv2=nn.Conv1d(conv1,conv2,3,padding=1)
        self.conv3=nn.Conv1d(conv2,conv3,3,padding=1)
        self.fc01 = nn.Linear(4*conv3, 362)
        
        self.batch_norm1 = nn.BatchNorm1d(conv1)
        self.batch_norm2 = nn.BatchNorm1d(conv2)
        self.ConvReluBatchNorm1 = nn.Sequential(
            self.conv1, self.relu, self.batch_norm1
            )
        
        # self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.batch_norm1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.batch_norm2(x)
        x = x.view(-1, 4*128)
        x = self.fc01(x)
        # x = self.dropout(x)
        return x
    

dataset = rEPhiDataset(file_path='F:\\pythontxtfile\\eEPhi.txt')
# Assuming dataset is a PyTorch Dataset object
total_size = len(dataset)
train_size = int(0.8 * total_size)  # 80% of the dataset
test_size = total_size - train_size  # Remaining 20%
print(f'Train size: {train_size}, Test size: {test_size}')
# Splitting the dataset
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# Instantiate the model, loss function, and optimizer

criterion = torch.nn.MSELoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImprovedNetwork().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train_model(dataloader, model, criterion, device, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for position, EPhi in dataloader:
            position, EPhi = position.to(device), EPhi.to(device)

            optimizer.zero_grad()
            # Forward pass
            outputs = model(position)
            loss = criterion(outputs, torch.view_as_real(EPhi).view(-1, 362))

            # Backward and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}')

def test_model(dataloader, model, criterion, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():  # No gradients needed
        for position, EPhi in dataloader:
            position = position.to(device)
            EPhi = EPhi.to(device)

            outputs = model(position)
            loss = criterion(outputs, torch.view_as_real(EPhi).view(-1, 362))

            total_loss += loss.item() * position.size(0)
            total_samples += position.size(0)

    average_loss = total_loss / total_samples
    print(f'Average Loss: {average_loss:.4f}')

num_epochs = 50  # Define the number of epochs for training
train_model(train_loader, model, criterion, device, optimizer, num_epochs)
test_model(test_loader, model, criterion, device)

# positionlist = [0]
# for i in range(17):
#     if not i == 0:
#         positionlist.append(positionlist[i-1]+(random.uniform(15,30)))
# rEPhi_sim = AEP.validateAEP(positionlist)

# distribution = AEP.positionlist2positionDistribution(positionlist)

# rep = [complex(0,0)] * 181
# for value in rEPhi_sim.values():
#     rep += value['rEPhi']
# rep = torch.view_as_real(torch.tensor(rep).to(device)).view(362)

# rEPhi_model = torch.tensor([0] * 362, dtype=torch.float).to(device)
# for value in distribution.values():
#     rEPhi_model += model(torch.tensor(value, dtype=torch.float).to(device))

# AEPcriterion = criterion(rEPhi_model, rep)