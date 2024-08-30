import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data import ConcatDataset
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
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)


class DeepFcNetwork(nn.Module):
    def __init__(self):
        super(DeepFcNetwork, self).__init__()
        # 定义全连接层
        self.fc1 = nn.Linear(6, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.fc5 = nn.Linear(256, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.fc6 = nn.Linear(128, 64)
        self.bn6 = nn.BatchNorm1d(64)
        self.fc7 = nn.Linear(64, 12)

        # 残差连接
        self.shortcut1 = nn.Linear(6, 64)
        self.shortcut2 = nn.Linear(64, 128)
        self.shortcut3 = nn.Linear(128, 256)
        self.shortcut5 = nn.Linear(256, 128)
        self.shortcut6 = nn.Linear(128, 64)

    def forward(self, x):
        identity = x
        x = F.relu(self.bn1(self.fc1(x)))
        identity = self.shortcut1(identity)
        x += identity  # 残差连接

        identity = x
        x = F.relu(self.bn2(self.fc2(x)))
        identity = self.shortcut2(identity)
        x += identity  # 残差连接

        identity = x
        x = F.relu(self.bn3(self.fc3(x)))
        identity = self.shortcut3(identity)
        x += identity  # 残差连接

        x = F.relu(self.bn4(self.fc4(x)))  # 在这一层保持256个单元，所以没有残差连接

        identity = x
        x = F.relu(self.bn5(self.fc5(x)))
        identity = self.shortcut5(identity)
        x += identity  # 残差连接

        identity = x
        x = F.relu(self.bn6(self.fc6(x)))
        identity = self.shortcut6(identity)
        x += identity  # 残差连接

        x = self.fc7(x)  # 最后一层不使用激活函数
        return x


criterion = torch.nn.MSELoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepFcNetwork().to(device)
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


# train_earlystop(model, criterion, optimizer, num_epochs=1000, patience=5)
