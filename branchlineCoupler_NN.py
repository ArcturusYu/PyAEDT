import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
import os
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, meanS=None, updated_values=None):
        """
        Args:
            meanS (list): A list of meanS values.
            updated_values (list): A list of updated_values lists.
        """
        self.meanS = meanS if meanS is not None else []
        self.updated_values = updated_values if updated_values is not None else []
    
    def __len__(self):
        return len(self.meanS)
    
    def __getitem__(self, idx):
        updated_meanS = torch.tensor(self.meanS[idx], dtype=torch.float32)
        updated_meanS = []
        for i in range(4):
            temp = self.meanS[idx][i]
            if i == 0:
                updated_meanS.append(np.mean(np.abs(temp))-90)
            else:
                updated_meanS.append(np.mean(np.abs(temp)))
            updated_meanS.append(np.std(temp))
            updated_meanS.append(np.var(temp))
        updated_meanS = torch.tensor(updated_meanS, dtype=torch.float32)
        cleaned_values = [float(value.replace('mm', '')) for value in self.updated_values[idx]]
        updated_values_tensor = torch.tensor(cleaned_values, dtype=torch.float32)
        return updated_meanS, updated_values_tensor

    def add_data(self, new_meanS, new_updated_values):
        self.meanS.append(new_meanS)
        self.updated_values.append(new_updated_values)

def load_or_initialize_dataset(filename):
    if os.path.exists(filename):
        return torch.load(filename)
    else:
        return CustomDataset()


filename = 'F:\\pythontxtfile\\Bcplrdataset.pth'
filename1 = 'F:\\pythontxtfile\\Bcplrdataset317.pth'
filename2 = 'F:\\pythontxtfile\\Bcplrdataset315.pth'
# filename3 = 'F:\\pythontxtfile\\dataset317.pth'

dataset = load_or_initialize_dataset(filename)
dataset1 = load_or_initialize_dataset(filename1)
dataset2 = load_or_initialize_dataset(filename2)
# dataset3 = load_or_initialize_dataset(filename3)

combined_dataset = ConcatDataset([dataset1, dataset, dataset2])
# Assuming dataset is a PyTorch Dataset object
total_size = len(combined_dataset)
train_size = int(0.9 * total_size)
test_size = total_size - train_size
print(f'Train size: {train_size}, Test size: {test_size}')
# Splitting the dataset
train_dataset, test_dataset = random_split(combined_dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class DeepCustomNN(nn.Module):
    def __init__(self):
        super(DeepCustomNN, self).__init__()
        
        self.fc1 = nn.Linear(5, 64)
        self.bn1 = nn.BatchNorm1d(64)
        
        self.fc2 = nn.Linear(64, 256)
        self.bn2 = nn.BatchNorm1d(256)
        
        self.fc3 = nn.Linear(256, 512)
        self.bn3 = nn.BatchNorm1d(512)
        
        self.fc4 = nn.Linear(512, 512)
        self.bn4 = nn.BatchNorm1d(512)
        
        self.fc5 = nn.Linear(512, 512)
        self.bn5 = nn.BatchNorm1d(512)
        
        self.fc6 = nn.Linear(512, 512)
        self.bn6 = nn.BatchNorm1d(512)
        
        self.fc7 = nn.Linear(512, 512)
        self.bn7 = nn.BatchNorm1d(512)
        
        self.fc8 = nn.Linear(512, 512)
        self.bn8 = nn.BatchNorm1d(512)
        
        self.fc9 = nn.Linear(512, 512)
        self.bn9 = nn.BatchNorm1d(512)
        
        self.fc10 = nn.Linear(512, 12)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.fc2(out)
        out = self.bn2(out)
        out = F.relu(out)
    
        out = self.fc3(out)
        out = self.bn3(out)
        out = F.relu(out)
    
        # Residual Block 1
        residual = out
        out = self.fc4(out)
        out = self.bn4(out)
        out = F.relu(out)
        
        out = self.fc5(out)
        out = self.bn5(out)
        out += residual  # Add residual connection
        out = F.relu(out)
    
        # Residual Block 2
        residual = out
        out = self.fc6(out)
        out = self.bn6(out)
        out = F.relu(out)
        
        out = self.fc7(out)
        out = self.bn7(out)
        out += residual  # Add residual connection
        out = F.relu(out)
    
        # Residual Block 3
        residual = out
        out = self.fc8(out)
        out = self.bn8(out)
        out = F.relu(out)
        
        out = self.fc9(out)
        out = self.bn9(out)
        out += residual  # Add residual connection
        out = F.relu(out)
    
        out = self.fc10(out)
        return out
    
criterion = torch.nn.MSELoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepCustomNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train_earlystop(model, criterion, optimizer, num_epochs, patience=5):
    best_loss = float('inf')
    epochs_no_improve = 0
    for epoch in range(num_epochs):
        # 训练过程
        model.train()
        train_loss = 0.0
        for value, label in train_loader:
            value, label = value.view(-1,12).to(device), label.view(-1,5).to(device)
            optimizer.zero_grad()
            outputs = model(label)
            loss = criterion(outputs, value)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # 验证过程
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for value, label in test_loader:
                value, label = value.view(-1,12).to(device), label.view(-1,5).to(device)
                outputs = model(label)
                loss = criterion(outputs, value)
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

train_earlystop(model, criterion, optimizer, num_epochs=100, patience=30)

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

        self.fc7 = nn.Linear(512, 5)
        self.variableBase = torch.tensor([2.2015, 0.2, 0.1984, 0.2], dtype=torch.float32).to(device)

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

        x = torch.sigmoid(self.fc7(x))
        x = x = torch.concat([((x[:4] * 0.4 + 0.8) * self.variableBase),(x[4:] * 1.2 - 0.2)], dim=0)
        return x

model_opt = S_Ang_OptNN().to(device)

def S_Ang_Opt(Optmodel, NNmodel, criterion, optimizer, num_epochs, idealpattern=(90, 0)):
    idealpattern = torch.tensor(idealpattern, dtype=torch.float32).view(-1).to(device)
    Optmodel.train()
    NNmodel.eval()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = Optmodel(idealpattern)
        NNmodeloutputs = NNmodel(outputs.view(-1, 5))
        # mean1 = torch.mean(NNmodeloutputs[0,:101])
        # mean2 = torch.mean(torch.abs(NNmodeloutputs[0,101:202]-NNmodeloutputs[0,202:303]))
        # NNmodeloutputs_compressed = torch.stack((mean1, mean2))
        loss = criterion(NNmodeloutputs, torch.tensor([0]*12, dtype=torch.float32).to(device))
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {loss.item():.4f}, Outputs: {outputs.detach().cpu().numpy()}')

optimizer1 = torch.optim.Adam(model_opt.parameters(), lr=0.001)
S_Ang_Opt(model_opt, model, criterion, optimizer1, num_epochs=100, idealpattern=(90, 0.2))