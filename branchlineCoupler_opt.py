from ansys.aedt.core import Hfss, Desktop
import numpy as np
import random

def normalize_angles(angles):
    normalized_angles = [(angle + 180) % 360 - 180 for angle in angles]
    return normalized_angles

import torch
from torch.utils.data import Dataset
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
        cleaned_values = [float(value.replace('mm', '')) for value in self.updated_values[idx]]
        updated_values_tensor = torch.tensor(cleaned_values, dtype=torch.float32)
        return meanS_value, updated_values_tensor

    def add_data(self, new_meanS, new_updated_values):
        self.meanS.append(new_meanS)
        self.updated_values.append(new_updated_values)

def load_or_initialize_dataset(filename):
    if os.path.exists(filename):
        return torch.load(filename)
    else:
        return CustomDataset()

# 文件路径
filename = 'D:\\Ansoft\\pythontxtfile\\Bcplrdataset.pth'

# 加载或初始化数据集
dataset = load_or_initialize_dataset(filename)

lambda4 =2.2015
z0=0.2
z0707=0.1984
zstub=0.2
variables = [lambda4, z0, z0707, zstub]
keys = ['lambda4', 'z0', 'z0707', 'zstub']

with Desktop(version='242', non_graphical=True):
    for i in range(1000):
        hfss = Hfss(project="D:\\Ansoft\\branchlinecoupler-2b.aedt",
                           design='coupler-1a-circleport',
                           solution_type='Modal')

        # 对每个变量进行扰动，并更新到仿真软件字典
        validation = False
        while not validation:
            updated_values = []
            for key, value in zip(keys, variables):
                # 计算扰动值，随机选择增加或减少20%
                disturbance = value * 0.2 * random.uniform(-1, 1)
                new_value = value + disturbance
                # 更新仿真软件字典
                hfss[key] = f'{new_value}mm'
                # 收集更新后的值到列表
                updated_values.append(hfss[key])
            hfss['lstubbias'] = f'{random.uniform(-0.2, 1)}mm'
            updated_values.append(hfss['lstubbias'])
            validation = hfss.analyze(setup='SetupKa',cores=8)

        solutiondata = hfss.post.get_solution_data()
        meanS = []
        sdiff = normalize_angles(np.subtract(solutiondata.data_phase('S(2,1)', False), solutiondata.data_phase('S(3,1)', False)))
        meanS.append(sdiff)
        sdiff = solutiondata.data_db10('S(2,1)')
        meanS.append(sdiff)
        sdiff = solutiondata.data_db10('S(3,1)')
        meanS.append(sdiff)
        sdiff = solutiondata.data_db10('S(1,1)')
        meanS.append(sdiff)
        # 添加数据到数据集
        dataset.add_data(meanS, updated_values)
        print(len(dataset))
        # 保存更新后的数据集
        torch.save(dataset, filename)
        hfss.close_project()
print('Done!')