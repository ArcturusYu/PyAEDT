import torch
import torch.nn as nn
from torch.utils.data import random_split
from torch.utils.data import Dataset, TensorDataset, DataLoader
import re
import fileinput

def file_to_dict(filename):
    data_dict = {}
    # Regex to match complex numbers
    complex_num_pattern = re.compile(r'[-+]?\d*\.?\d+e?[-+]?\d*j')

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
                if complex_numbers.strip():
                    complex_matches = complex_num_pattern.findall(complex_numbers)
                    values.extend([complex(num) for num in complex_matches])
            else:
                # Continue collecting values
                line = line.strip().rstrip(']')
                if line.strip():  # Ensure it's not empty
                    complex_matches = complex_num_pattern.findall(line)
                    values.extend([complex(num) for num in complex_matches])

        # Add the last key-value pair
        if current_key is not None:
            data_dict[current_key] = values

    return data_dict

class rEPhiDataset:
    def __init__(self, file_path, transform=None, target_transform=None):
        self.EPhi_positions = file_to_dict(file_path)
        self.positions = torch.tensor([k for k in self.EPhi_positions.keys()], dtype=torch.cfloat)
        self.EPhi = torch.tensor([v for v in self.EPhi_positions.values()], dtype=torch.cfloat)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.EPhi_positions)
    
    def __getitem__(self, idx):
        position = self.positions[idx]
        EPhi = self.EPhi[idx]
        if self.transform:
            EPhi = self.transform(EPhi)
        if self.target_transform:
            position = self.target_transform(position)
        return position, EPhi

class ComplexNetwork(nn.Module):
    def __init__(self):
        super(ComplexNetwork, self).__init__()
        self.fc1 = nn.Linear(8, 128)  # Correct input dimension
        self.fc2 = nn.Linear(128, 362)  # Correct output dimension to match 181 complex numbers represented as 362 real values

    def forward(self, x):
        # Convert complex to real and flatten
        x = torch.view_as_real(x)  # Converts complex numbers to real, shape becomes (batch_size, 4, 2)
        x = x.view(x.size(0), -1)  # Flatten input to shape (batch_size, 8) #flatten to the first dimension
        x = torch.relu(self.fc1(x)) # https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
        x = self.fc2(x)
        # Assuming output needs to be complex again, reshaping and converting
        x = torch.view_as_complex(x.view(x.size(0), -1, 2))  # Reshape and convert to complex
        return x

dataset = rEPhiDataset(file_path='F:\\pythontxtfile\\eEPhi_example.txt')
# dataset = rEPhiDataset(file_path='F:\\pythontxtfile\\eEPhi.txt')

total_size = len(dataset)
train_size = int(0.8 * total_size)  # 80% of the dataset
test_size = total_size - train_size  # Remaining 20%

# Splitting the dataset
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Assume dataloader is already defined and yields position (1x4) and EPhi (1x181)
# Example usage in a training loop

num_epochs = 50  # Define the number of epochs for training

# Instantiate the model, loss function, and optimizer

criterion = torch.nn.MSELoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ComplexNetwork().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train_model(dataloader, model, criterion, optimizer, num_epochs=25):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for position, EPhi in dataloader:
            position = position.to(device)  # Move position tensor to the device
            EPhi = EPhi.to(device)          # Move EPhi tensor to the device

            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(position)
            loss = criterion(torch.view_as_real(outputs), torch.view_as_real(EPhi))

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
            loss = criterion(torch.view_as_real(outputs), torch.view_as_real(EPhi))

            total_loss += loss.item() * position.size(0)
            total_samples += position.size(0)

    average_loss = total_loss / total_samples
    print(f'Average Loss: {average_loss:.4f}')

train_model(train_loader, model, criterion, optimizer, num_epochs=num_epochs)
test_model(test_loader, model, criterion, device)

# torch.save(model,'F:\pythontxtfile')