import ansys.aedt.core
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import re
import os

hfss = ansys.aedt.core.Hfss(project="C:\\Users\\bacto\\Documents\\Ansoft\\branchlinecoupler-1c.aedt",
                            design="coupler-1a5",
                            version='242',
                            non_graphical=True)

class CustomDataset(Dataset):
    def __init__(self, data_file='data.pt', load_from_file=True):
        self.data_file = data_file

        if load_from_file and os.path.exists(data_file):
            # Load data from the saved file
            saved_data = torch.load(data_file)
            self.dB_S31_minus_S21 = saved_data.get('dB_S31_minus_S21', [])
            self.cang_S31_minus_S21 = saved_data.get('cang_S31_minus_S21', [])
            self.keys = saved_data.get('keys', [])
            self.length = len(self.keys)
        else:
            # Initialize empty lists if no data is loaded
            self.dB_S31_minus_S21 = []
            self.cang_S31_minus_S21 = []
            self.keys = []
            self.length = 0

    def add_data(self, dB_S31_minus_S21, cang_S31_minus_S21, key):
        # Convert to tensors
        dB_tensor = torch.tensor(dB_S31_minus_S21, dtype=torch.float32)
        cang_tensor = torch.tensor(cang_S31_minus_S21, dtype=torch.float32)
        key_tensor = torch.tensor(key, dtype=torch.float32)

        # Append data to lists
        self.dB_S31_minus_S21.append(dB_tensor)
        self.cang_S31_minus_S21.append(cang_tensor)
        self.keys.append(key_tensor)

        # Update length
        self.length += 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        key = self.keys[idx]
        dB_S31_minus_S21 = self.dB_S31_minus_S21[idx]
        cang_S31_minus_S21 = self.cang_S31_minus_S21[idx]

        # Check for invalid data
        if torch.all(dB_S31_minus_S21 == dB_S31_minus_S21[0]):
            if idx + 1 < len(self):
                return self.__getitem__(idx + 1)
            else:
                raise IndexError("No valid data found in the dataset.")
        else:
            # Combine the data and return
            combined_data = torch.cat((dB_S31_minus_S21, cang_S31_minus_S21), dim=0)
            return combined_data, key

    def save(self):
        # Save data to file
        torch.save({
            'dB_S31_minus_S21': self.dB_S31_minus_S21,
            'cang_S31_minus_S21': self.cang_S31_minus_S21,
            'keys': self.keys,
            'length': self.length
        }, self.data_file)

def load_data_from_hfss(hfss, dataset):
    # Load data from HFSS
    variation_dic = hfss.available_variations.variations(output_as_dict=True)
    for i in variation_dic:
        data = hfss.post.get_solution_data(
            expressions=['dB(S(3,1))-dB(S(2,1))', 'abs(cang_deg(S(3,1))-cang_deg(S(2,1)))'],
            variations=i,
            setup_sweep_name='Setup1 : Sweep'
        )
        dB_data = data.data_real('dB(S(3,1))-dB(S(2,1))')
        cang_tmp = data.data_real('abs(cang_deg(S(3,1))-cang_deg(S(2,1)))')
        canglist_tmp = []
        for angle in cang_tmp:
            if angle > 180:
                canglist_tmp.append(angle - 180)
            else:
                canglist_tmp.append(angle)
        key = (
            float(re.findall(r"[-+]?\d*\.\d+|\d+", i['lambda4'])[0]),
            float(re.findall(r"[-+]?\d*\.\d+|\d+", i['z0'])[0]),
            float(re.findall(r"[-+]?\d*\.\d+|\d+", i['z0707'])[0])
        )

        # Add data to the dataset
        dataset.add_data(dB_data, canglist_tmp, key)

    # After loading data, save the dataset
    dataset.save()
    
class CustomNN(nn.Module):
    def __init__(self):
        super(CustomNN, self).__init__()
        self.fc1 = nn.Linear(202, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 3)

    def forward(self, x):
        residual = x  # 保存输入以用于残差连接
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x
    
dataset = CustomDataset(data_file='F:\\pythontxtfile\\coupler-1c', load_from_file=True)
hfss.release_desktop()
# Assuming dataset is a PyTorch Dataset object
total_size = len(dataset)
train_size = int(0.9 * total_size)  # 80% of the dataset
test_size = total_size - train_size  # Remaining 20%
print(f'Train size: {train_size}, Test size: {test_size}')
# Splitting the dataset
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Instantiate the model, loss function, and optimizer

criterion = torch.nn.MSELoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train_earlystop(model, criterion, optimizer, num_epochs, patience=5):
    best_loss = float('inf')
    epochs_no_improve = 0
    for epoch in range(num_epochs):
        # 训练过程
        model.train()
        train_loss = 0.0
        for value, label in train_loader:
            value, label = value.to(device), label.to(device)
            optimizer.zero_grad()
            outputs = model(value)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # 验证过程
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for value, label in test_loader:
                value, label = value.to(device), label.to(device)
                outputs = model(value)
                loss = criterion(outputs, label)
                val_loss += loss.item()

        val_loss /= len(test_loader)

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # 早停检查
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
            best_model_wts = model.state_dict()
        else:
            epochs_no_improve += 1

        if epochs_no_improve == patience:
            print('Early stopping!')
            model.load_state_dict(best_model_wts)
            break

train_earlystop(model, criterion, optimizer, num_epochs=200, patience=10)
idealpattern = torch.tensor([0] * 101 + [90] * 101, dtype=torch.float32).view(1,-1).to(device)
model(idealpattern)