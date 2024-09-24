
from matplotlib.pyplot import cla
from regex import R
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import re
import os

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

def load_data_from_hfss(dataset):
    import ansys.aedt.core
    hfss = ansys.aedt.core.Hfss(project="C:\\Users\\bacto\\Documents\\Ansoft\\branchlinecoupler-1c.aedt",
                                design="coupler-1a5",
                                version='242',
                                non_graphical=True)
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
    hfss.release_desktop()

class DeepCustomNN(nn.Module):
    def __init__(self):
        super(DeepCustomNN, self).__init__()
        
        self.fc1 = nn.Linear(3, 1024)
        self.dropout1 = nn.Dropout(p=0.3)

        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(p=0.3)
        
        self.fc3 = nn.Linear(512, 512)
        self.dropout3 = nn.Dropout(p=0.3)

        self.fc4 = nn.Linear(512, 256)
        self.dropout4 = nn.Dropout(p=0.3)

        self.fc5 = nn.Linear(256, 256)
        self.dropout5 = nn.Dropout(p=0.3)

        self.fc6 = nn.Linear(256, 128)
        self.dropout6 = nn.Dropout(p=0.3)

        self.fc7 = nn.Linear(128, 128)
        self.dropout7 = nn.Dropout(p=0.3)

        self.fc8 = nn.Linear(128, 64)
        self.dropout8 = nn.Dropout(p=0.3)

        self.fc9 = nn.Linear(64, 64)
        self.dropout9 = nn.Dropout(p=0.3)

        self.fc10 = nn.Linear(64, 202)
        self.variableBase = torch.tensor([2.265, 0.201, 0.202], dtype=torch.float32).to(device)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        # out = self.dropout1(out)
        
        out = self.fc2(out)
        out = F.relu(out)
        # out = self.dropout2(out)

        out = self.fc3(out)
        out = F.relu(out)
        # out = self.dropout3(out)

        out = self.fc4(out)
        out = F.relu(out)
        # out = self.dropout4(out)

        out = self.fc5(out)
        out = F.relu(out)
        # out = self.dropout5(out)

        out = self.fc6(out)
        out = F.relu(out)
        # out = self.dropout6(out)

        out = self.fc7(out)
        out = F.relu(out)
        # out = self.dropout7(out)

        out = self.fc8(out)
        out = F.relu(out)
        # out = self.dropout8(out)

        out = self.fc9(out)
        out = F.relu(out)
        # out = self.dropout9(out)

        out = self.fc10(out)
        return out
    
dataset = CustomDataset(data_file='F:\\pythontxtfile\\coupler-1c', load_from_file=True)

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
model = DeepCustomNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train_earlystop(model, criterion, optimizer, num_epochs, patience=10, min_delta=0.001):
    best_loss = float('inf')
    epochs_no_improve = 0
    best_model_wts = None
    for epoch in range(num_epochs):
        # Training loop
        model.train()
        train_loss = 0.0
        for label, value in train_loader:
            value, label = value.to(device), label.to(device)
            optimizer.zero_grad()
            outputs = model(value)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for label, value in test_loader:
                value, label = value.to(device), label.to(device)
                outputs = model(value)
                loss = criterion(outputs, label)
                val_loss += loss.item()
        val_loss /= len(test_loader)

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # Early stopping check with min_delta
        if best_loss - val_loss > min_delta:
            best_loss = val_loss
            epochs_no_improve = 0
            best_model_wts = model.state_dict()
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print('Early stopping!')
            if best_model_wts is not None:
                model.load_state_dict(best_model_wts)
            break


train_earlystop(model, criterion, optimizer, num_epochs=200, patience=10)
# idealpattern = torch.tensor([0] * 101 + [90] * 101, dtype=torch.float32).view(1,-1).to(device)
# model(idealpattern)

class S_Ang_OptNN(nn.Module):
    def __init__(self):
        super(S_Ang_OptNN, self).__init__()
        self.fc1 = nn.Linear(2, 512)
        self.dropout1 = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(512, 512)
        self.dropout2 = nn.Dropout(p=0.5)

        self.fc3 = nn.Linear(512, 512)
        self.dropout3 = nn.Dropout(p=0.5)

        self.fc4 = nn.Linear(512, 512)
        self.dropout4 = nn.Dropout(p=0.5)

        self.fc5 = nn.Linear(512, 512)
        self.dropout5 = nn.Dropout(p=0.5)

        self.fc6 = nn.Linear(512, 512)
        self.dropout6 = nn.Dropout(p=0.5)

        self.fc7 = nn.Linear(512, 3)
        self.variableBase = torch.tensor([2.265, 0.201, 0.202], dtype=torch.float32).to(device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)

        x = F.relu(self.fc2(x))
        x = self.dropout2(x)

        x = F.relu(self.fc3(x))
        x = self.dropout3(x)

        x = F.relu(self.fc4(x))
        x = self.dropout4(x)

        x = F.relu(self.fc5(x))
        x = self.dropout5(x)

        x = F.relu(self.fc6(x))
        x = self.dropout6(x)

        x = torch.sigmoid(self.fc7(x)) * 0.1 + self.variableBase
        return x

model_opt = S_Ang_OptNN().to(device)

def S_Ang_Opt(Optmodel, NNmodel, criterion, optimizer, num_epochs, idealpattern=(0, 90)):
    idealpattern = torch.tensor(idealpattern, dtype=torch.float32).view(-1).to(device)
    Optmodel.train()
    NNmodel.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = Optmodel(idealpattern)
        NNmodeloutputs = NNmodel(outputs)
        mean1 = torch.mean(NNmodeloutputs[:101])
        mean2 = torch.mean(NNmodeloutputs[101:])
        NNmodeloutputs_compressed = torch.stack((mean1, mean2))
        loss = criterion(mean1, idealpattern[0])
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {loss.item():.4f}, Outputs: {outputs.detach().cpu().numpy()}')

optimizer1 = torch.optim.Adam(model_opt.parameters(), lr=0.001)
S_Ang_Opt(model_opt, model, criterion, optimizer1, num_epochs=100, idealpattern=(0.2, 90))