import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
import os

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
        meanS_value = torch.tensor(self.meanS[idx], dtype=torch.float32)
        updated_values_tensor = torch.tensor(self.updated_values[idx], dtype=torch.float32)
        return meanS_value, updated_values_tensor

    def add_data(self, new_meanS, new_updated_values):
        self.meanS.append(new_meanS)
        self.updated_values.append(new_updated_values)


def load_or_initialize_dataset(filename):
    if os.path.exists(filename):
        return torch.load(filename)
    else:
        return CustomDataset()


filename = 'F:\\pythontxtfile\\dataset.pth'
filename1 = 'F:\\pythontxtfile\\dataset317.pth'
filename2 = 'F:\\pythontxtfile\\dataset115.pth'
filename3 = 'F:\\pythontxtfile\\dataset317.pth'

dataset = load_or_initialize_dataset(filename)
dataset1 = load_or_initialize_dataset(filename1)
dataset2 = load_or_initialize_dataset(filename2)
dataset3 = load_or_initialize_dataset(filename3)

combined_dataset = ConcatDataset([dataset1, dataset, dataset2, dataset3])
# Assuming dataset is a PyTorch Dataset object
total_size = len(combined_dataset)
train_size = int(0.9 * total_size)
test_size = total_size - train_size
print(f'Train size: {train_size}, Test size: {test_size}')
# Splitting the dataset
train_dataset, test_dataset = random_split(combined_dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class DeepFcNetwork(nn.Module):
    def __init__(self):
        super(DeepFcNetwork, self).__init__()
        # Define average pooling layer to reduce feature size
        self.avg_pool = nn.AvgPool1d(kernel_size=3)  # Example: Reducing size by half
        
        # Define fully connected layers with dropout
        reduced_size = 501 // 3  # Assuming the pooling reduces the size by half
        self.fc1 = nn.Linear(6 * reduced_size, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(p=0.5)  # 50% dropout after the first layer
        
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(p=0.5)  # 50% dropout after the second layer
        
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(p=0.5)  # 50% dropout after the third layer
        
        self.fc4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.dropout4 = nn.Dropout(p=0.5)  # 50% dropout after the fourth layer
        
        self.fc5 = nn.Linear(128, 64)
        self.bn5 = nn.BatchNorm1d(64)
        self.dropout5 = nn.Dropout(p=0.5)  # 50% dropout after the fifth layer
        
        self.fc6 = nn.Linear(64, 32)
        self.bn6 = nn.BatchNorm1d(32)
        self.dropout6 = nn.Dropout(p=0.5)  # 50% dropout after the sixth layer
        
        self.fc7 = nn.Linear(32, 12)

        # Define shortcut layers for residual connections
        self.shortcut1 = nn.Linear(6 * reduced_size, 1024)
        self.shortcut2 = nn.Linear(1024, 512)
        self.shortcut3 = nn.Linear(512, 256)
        self.shortcut4 = nn.Linear(256, 128)
        self.shortcut5 = nn.Linear(128, 64)
        self.shortcut6 = nn.Linear(64, 32)

    def forward(self, x):
        # Apply average pooling to reduce feature size
        x = self.avg_pool(x)
        
        # Flatten the input from (batch_size, 6, reduced_size) to (batch_size, 6 * reduced_size)
        x = x.view(x.size(0), -1)
        
        # First layer with residual connection and dropout
        identity = self.shortcut1(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = x + identity  # Avoid in-place operation

        # Second layer with residual connection and dropout
        identity = self.shortcut2(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = x + identity

        # Third layer with residual connection and dropout
        identity = self.shortcut3(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        x = x + identity

        # Fourth layer with residual connection and dropout
        identity = self.shortcut4(x)
        x = F.relu(self.bn4(self.fc4(x)))
        x = self.dropout4(x)
        x = x + identity

        # Fifth layer with residual connection and dropout
        identity = self.shortcut5(x)
        x = F.relu(self.bn5(self.fc5(x)))
        x = self.dropout5(x)
        x = x + identity

        # Sixth layer with residual connection and dropout
        identity = self.shortcut6(x)
        x = F.relu(self.bn6(self.fc6(x)))
        x = self.dropout6(x)
        x = x + identity

        # Final layer without activation
        x = self.fc7(x)
        return x

criterion = torch.nn.MSELoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepFcNetwork().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


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

train_earlystop(model, criterion, optimizer, num_epochs=100, patience=50)
idealpattern = torch.tensor([-90] * 501 + [0] * 501 + [-90] * 501 + [0] * 501 + [-90] * 501 + [0] * 501, dtype=torch.float32).view(1,-1).to(device)
model(idealpattern)