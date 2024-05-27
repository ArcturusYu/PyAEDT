import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
import re


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
        # Assuming 'file_to_dict' function reads the file and returns a dictionary which is then converted to DataFrame.
        # 'file_to_dict' is not defined in this snippet so ensure you have such a function defined or use pandas.read_csv or similar.
        self.EPhi_positions = file_to_dict(file_path)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.EPhi_positions)

    def __getitem__(self, idx):
        position,EPhi = list(self.EPhi_positions.items())[idx]
        position=torch.tensor(position, dtype=torch.complex64)
        EPhi=torch.tensor(EPhi, dtype=torch.complex64)
        if self.transform:
            EPhi = self.transform(EPhi)
        if self.target_transform:
            position = self.target_transform(position)
        return EPhi, position

import torch
import torch.nn as nn

class ComplexNetwork(nn.Module):
    def __init__(self):
        super(ComplexNetwork, self).__init__()
        # Assuming the input is concatenated real and imaginary parts, each of size 4
        # Total input size will be 8 (4 real + 4 imaginary)
        self.fc1 = nn.Linear(8, 128)  # Adjust input dimension based on your actual input size
        self.fc2 = nn.Linear(128, 362)  # Adjust accordingly

    def forward(self, x):
        # x should be a tensor containing both real and imaginary parts
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x



class ComplexNumberDataset(torch.utils.data.Dataset):
    def __init__(self, file_path):
        self.data = file_to_dict(file_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        keys = list(self.data.keys())
        key = keys[idx]
        real_parts = torch.tensor([c.real for c in self.data[key]], dtype=torch.float32)
        imag_parts = torch.tensor([c.imag for c in self.data[key]], dtype=torch.float32)
        return key, real_parts, imag_parts

# Usage
dataset = ComplexNumberDataset(file_path='F:\\pythontxtfile\\eEPhi.txt')
dataloader = DataLoader(dataset, batch_size=64, shuffle=False, drop_last=True)
# for EPhi, position in dataloader:
#     print(EPhi)
#     print(position)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ComplexNetwork().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train_model(dataloader, model, criterion, optimizer, num_epochs=25):
    model.train()  # Ensure the model is in training mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for key, real_parts, imag_parts in dataloader:
            # Prepare combined input, concatenate real and imaginary parts along feature dimension
            combined_input = torch.cat((real_parts, imag_parts), dim=1).to(device)
            
            # Move target to the device if necessary and prepare it similarly
            # Assuming you have a target that needs similar processing
            
            optimizer.zero_grad()
            outputs = model(combined_input)
            # Ensure targets are prepared similar to inputs if they are complex
            loss = criterion(outputs, combined_input)  # Example, adjust according to your specific case
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}')
        print(len(dataloader))

train_model(dataloader, model, criterion, optimizer, num_epochs=25)
