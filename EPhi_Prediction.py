import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
import re
import random
from pyswarm import pso
import numpy as np
import matplotlib.pyplot as plt
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
    
def positionlist2positionDistribution(positionlist):
    positionDistribution = {}
    for i in range(17):
        if i == 0:
            positionDistribution[i] = (0, 0, positionlist[i+1], positionlist[i+2] - positionlist[i+1])
        elif i == 1:
            positionDistribution[i] = (0, positionlist[i], positionlist[i+1] - positionlist[i], positionlist[i+2] - positionlist[i+1])
        elif i == 15:
            positionDistribution[i] = (positionlist[i-1] - positionlist[i-2], positionlist[i] - positionlist[i-1], positionlist[i+1] - positionlist[i], 0)
        elif i == 16:
            positionDistribution[i] = (positionlist[i-1] - positionlist[i-2], positionlist[i] - positionlist[i-1], 0, 0)
        else:
            positionDistribution[i] = (positionlist[i-1] - positionlist[i-2], positionlist[i] - positionlist[i-1], positionlist[i+1] - positionlist[i], positionlist[i+2] - positionlist[i+1])
    return positionDistribution

class ConvNetwork(nn.Module):
    def __init__(self):
        super(ConvNetwork, self).__init__()
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
    
class FcNetwork(nn.Module):
    def __init__(self):
        super(FcNetwork, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 362)
        self.fc3 = nn.Linear(362, 362)
        self.fc4 = nn.Linear(362, 362)
        self.fc5 = nn.Linear(362, 362)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        return x
    
class CustomFcNetwork(nn.Module):
    def __init__(self):
        super(CustomFcNetwork, self).__init__()
        self.fc1 = nn.Linear(4, 128)  # Input layer with 4 features
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 362)  # Output layer with 362 features

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # 假设这是回归任务，这里没有激活函数
        x = torch.squeeze(x, 1)  # 移除第二维度（索引为1的维度），如果这是多余的维度
        return x
class DeepFcNetwork(nn.Module):
    def __init__(self):
        super(DeepFcNetwork, self).__init__()
        self.fc1 = nn.Linear(4, 128)  # Input layer with 4 features
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 1024)  # Increasing depth
        self.fc5 = nn.Linear(1024, 512)  # Start decreasing
        self.fc6 = nn.Linear(512, 256)
        self.fc7 = nn.Linear(256, 362)  # Output layer with 362 features

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = self.fc7(x)  # Assuming a regression task, no activation here
        x = x.view(-1, 362)
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
model = DeepFcNetwork().to(device)
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

def train_earlystop(model, criterion, optimizer, num_epochs, patience=5):
    best_loss = float('inf')
    epochs_no_improve = 0
    for epoch in range(num_epochs):
        # 训练过程
        model.train()
        train_loss = 0.0
        for position, EPhi in train_loader:
            position, EPhi = position.to(device), EPhi.to(device)
            optimizer.zero_grad()
            outputs = model(position)
            loss = criterion(outputs, torch.view_as_real(EPhi).view(-1, 362))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # 验证过程
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for position, EPhi in test_loader:
                position, EPhi = position.to(device), EPhi.to(device)
                outputs = model(position)
                loss = criterion(outputs, torch.view_as_real(EPhi).view(-1, 362))
                val_loss += loss.item()

        val_loss /= len(test_loader)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

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

# num_epochs = 5000  # Define the number of epochs for training
# train_model(train_loader, model, criterion, device, optimizer, num_epochs)
# test_model(test_loader, model, criterion, device)
train_earlystop(model, criterion, optimizer, num_epochs=5000, patience=5)
# 模型训练完成 #########################

# def generate_positionlist():
#     positionlist = [0]
#     for i in range(1, 17):
#         positionlist.append(positionlist[i-1] + random.uniform(15, 30))
#     return positionlist

def objective_function(positiondelta):
    positionlist = [0]
    for i in range(1, 17):
        positionlist.append(positionlist[i-1] + positiondelta[i-1])
    distribution = positionlist2positionDistribution(positionlist)
    rEPhi_model = torch.tensor([0] * 362, dtype=torch.float).to(device)
    for value in distribution.values():
            rEPhi_model += model(torch.tensor(value, dtype=torch.float).to(device)).view(362)
    rEPhi_model = torch.abs(torch.view_as_complex(rEPhi_model.view(181, 2))).detach().cpu().numpy()
    
    # 找到最大值和第二大值
    max_value = np.max(rEPhi_model)
    second_max_value = np.partition(rEPhi_model.flatten(), -2)[-2]
    
    # 返回目标值（注意这里取负，因为我们要最大化这个差值）
    return -(max_value - second_max_value)

# 定义变量边界
lb = [15]*16
ub = [30]*16

# 使用PSO优化
opt_positionlist, opt_value = pso(objective_function, lb, ub, swarmsize=100, maxiter=100)

print('优化后的 positionlist:', positionlist)
print('优化后的目标值:', -opt_value)

# 用HFSS验证，画图 #################
positionlist = sorted(opt_positionlist)
rEPhiDic = AEP.validateAEP(positionlist)

distribution = positionlist2positionDistribution(positionlist)

rEPhi_sim = [complex(0,0)] * 181
for value in rEPhiDic.values():
    rEPhi_sim += value['rEPhi']
rEPhi_sim = torch.abs(torch.tensor(rEPhi_sim)).to(device)

rEPhi_model = torch.tensor([0] * 362, dtype=torch.float).to(device)
for value4 in distribution.values():
    rEPhi_model += model(torch.tensor(value4, dtype=torch.float).to(device)).view(362)
rEPhi_model = torch.abs(torch.view_as_complex(rEPhi_model.view(181,2)))
AEPcriterion = criterion(rEPhi_model, rEPhi_sim)
print(f'AEPcriterion: {AEPcriterion}')

x = value['Theta']
ymodel = rEPhi_model.cpu().detach().numpy()
ysim = rEPhi_sim.cpu().detach().numpy()

# 将 ymodel 和 ysim 转换为 dB 单位
ymodel_db = 20 * np.log10(np.abs(ymodel))
ysim_db = 20 * np.log10(np.abs(ysim))

# 绘制图表
fig, ax = plt.subplots()
ax.plot(x, ymodel_db, label='Model (dB)')
ax.plot(x, ysim_db, label='Sim (dB)')
ax.set_xlabel('Theta')
ax.set_ylabel('Magnitude (dB)')
ax.legend()
plt.show()