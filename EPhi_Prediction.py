import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
import re
import torch.profiler as profiler

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
        self.positions = torch.tensor([k for k in self.EPhi_positions.keys()], dtype=torch.complex64)
        self.EPhi = torch.tensor([v for v in self.EPhi_positions.values()], dtype=torch.complex64)
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

class ComplexConvNetwork(nn.Module):
    def __init__(self):
        super(ComplexConvNetwork, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=256, out_channels=181, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=181, out_channels=64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(2 * 181, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 4)
        
    def forward(self, x):
        x = torch.view_as_real(x)  # Converts complex numbers to real, shape becomes (batch_size, 4, 2)
        x = x.view(x.size(0), -1)  # Flatten input to shape (batch_size, 8) #flatten to the first dimension

        # Apply convolutional layers
        x = F.silu(self.conv1(x))
        x = F.silu(self.conv2(x))
        x = F.silu(self.fc1(x))
        x = F.silu(self.fc2(x))
        x = self.fc3(x)

        return x


dataset = rEPhiDataset(file_path='F:\\pythontxtfile\\eEPhi_example.txt')
# Assuming dataset is a PyTorch Dataset object
total_size = len(dataset)
train_size = int(0.8 * total_size)  # 80% of the dataset
test_size = total_size - train_size  # Remaining 20%
print(f'Train size: {train_size}, Test size: {test_size}')
# Splitting the dataset
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, num_workers=0, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, num_workers=0, batch_size=256, shuffle=False)

# Assume dataloader is already defined and yields position (1x4) and EPhi (1x181)
# Example usage in a training loop

num_epochs = 50  # Define the number of epochs for training

# Instantiate the model, loss function, and optimizer

criterion = torch.nn.MSELoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ComplexConvNetwork().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train_model(dataloader, model, criterion, device, optimizer, num_epochs=25):
    model.train()
    with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
    schedule=profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
    on_trace_ready=profiler.tensorboard_trace_handler('C:\\Users\\bacto\\Documents\\PyAEDT\\log\\train'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
        for epoch in range(num_epochs):
            running_loss = 0.0
            for position, EPhi in dataloader:
                position, EPhi = position.to(device), EPhi.to(device)

                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(EPhi)
                loss = criterion(position, outputs)

                # Backward and optimize
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                prof.step()

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

train_model(train_loader, model, criterion, device, optimizer, num_epochs=num_epochs)
test_model(test_loader, model, criterion, device)


