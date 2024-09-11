import ansys.aedt.core
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import re

hfss = ansys.aedt.core.Hfss(project="C:\\Users\\bacto\Documents\\Ansoft\\branchlinecoupler-1c.aedt",
                            design="coupler-1a5",
                            version='242',
                            non_graphical=False)

class CustomDataset(Dataset):
    def __init__(self, hfss=None):
        self.variation_dic = hfss.available_variations.variations(output_as_dict=True)
        # 使用一次循环提取所有所需的数据
        self.dB_S31_minus_S21 = []
        self.cang_S31_minus_S21 = []

        for i in self.variation_dic:
            data = hfss.post.get_solution_data(expressions=['dB(S(3,1))-dB(S(2,1))', 'abs(cang_deg(S(3,1))-cang_deg(S(2,1)))'],
                                               variations=i,
                                               setup_sweep_name='Setup1 : Sweep')
            self.dB_S31_minus_S21.append(data.data_real('dB(S(3,1))-dB(S(2,1))'))
            cang_tmp = data.data_real('abs(cang_deg(S(3,1))-cang_deg(S(2,1)))')
            canglist_tmp = []
            for i in cang_tmp:
                if i > 180:
                    canglist_tmp.append(i - 180)
                else:
                    canglist_tmp.append(i)
            self.cang_S31_minus_S21.append(canglist_tmp)
            
        self.dB_S31_minus_S21 = torch.tensor(self.dB_S31_minus_S21, dtype=torch.float32)
        self.cang_S31_minus_S21 = torch.tensor(self.cang_S31_minus_S21, dtype=torch.float32)
        self.keys = torch.tensor([(float(re.findall(r"[-+]?\d*\.\d+|\d+", k['lambda4'])[0]),
                                   float(re.findall(r"[-+]?\d*\.\d+|\d+", k['z0'])[0]),
                                   float(re.findall(r"[-+]?\d*\.\d+|\d+", k['z0707'])[0]))
                                   for k in self.variation_dic],
                                   dtype=torch.float) 

    def __len__(self):
        return len(self.variation_dic)

    def __getitem__(self, idx):
        key = self.keys[idx]
        dB_S31_minus_S21 = self.dB_S31_minus_S21[idx]
        can_S31_minus_S21 = self.cang_S31_minus_S21[idx]
        # 检查是否为异常值（例如，solution_data中的所有值是否都相同）
        if torch.all(dB_S31_minus_S21 == dB_S31_minus_S21[0]):
            # 如果数据无效，递归地调用下一个索引的数据，避免返回 None
            if idx + 1 < len(self):
                return self.__getitem__(idx + 1)
            else:
                raise IndexError("No valid data found in the dataset.")
        else:
            return torch.cat((dB_S31_minus_S21, can_S31_minus_S21), dim=0), key
    
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
    
dataset = CustomDataset(hfss=hfss)
hfss.release_desktop()
# Assuming dataset is a PyTorch Dataset object
total_size = len(dataset)
train_size = int(0.8 * total_size)  # 80% of the dataset
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