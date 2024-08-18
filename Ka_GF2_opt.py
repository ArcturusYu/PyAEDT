import pyaedt
import numpy as np
import statistics
import random

def normalize_angles(angles):
    normalized_angles = [(angle + 180) % 360 - 180 for angle in angles]
    return normalized_angles

import pyaedt.hfss
import torch
from torch.utils.data import Dataset, DataLoader
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

# 文件路径
filename = 'F:\\pythontxtfile\\dataset.pth'

# 加载或初始化数据集
dataset = load_or_initialize_dataset(filename)

l0 = 0.721
lt = 1.66313
line2w = 0.47237
line1w = 0.352
line0w = 0.0299
line3w = 0.59447
line4w = 0.264786
lt3f = 1.2843
lt4f = 0.2215
line180w = 0.2933
line90w = 0.163
lt2f = 1.189
variables = [l0, lt, line2w, line1w, line0w, line3w, line4w, lt3f, lt4f, line180w, line90w, lt2f]
keys = ['l0', 'lt', 'line2w', 'line1w', 'line0w', 'line3w', 'line4w', 'lt3f', 'lt4f', 'line180w', 'line90w', 'lt2f']


with pyaedt.Desktop(version='242', non_graphical=True):
    for i in range(500):
        hfss = pyaedt.Hfss(project="C:\\Users\\bacto\Documents\\Ansoft\\Ka_GF2-liyu-1C-test.aedt",
                           design='HFSSDesign4',
                           solution_type='Modal')
        # 对每个变量进行扰动，并更新到仿真软件字典
        validation = False
        while not validation:
            updated_values = []
            for key, value in zip(keys, variables):
                # 计算扰动值，随机选择增加或减少20%
                disturbance = value * 0.2 * random.choice([-1, 1])
                new_value = value + disturbance
                # 更新仿真软件字典
                hfss[key] = f'{new_value}mm' if key != 'lt3f' and key != 'lt4f' and key != 'lt2f' else f'{new_value}'
                # 收集更新后的值到列表
                updated_values.append(hfss[key])
            validation = hfss.analyze(setup='SweepKa',cores=8)

        # expressions = ['Phase4_6', 'Phase6_8', 'Phase8_2', 'mag4_6', 'mag6_8', 'mag8_2']
        expressions = ['S(2,in)', 'S(4,in)', 'S(6,in)', 'S(8,in)']
        solutiondata = hfss.post.get_solution_data(expressions=expressions,
                                                   setup_sweep_name='SweepKa : Sweep',
                                                   domain='Sweep')
        meanS = []
        for i in (8,6,4):
            sdiff = normalize_angles(np.subtract(solutiondata.data_phase(f'S({i},in)',False), solutiondata.data_phase(f'S({i-2},in)',False)))
            meanS.append(statistics.fmean(sdiff))
            sdiff = np.subtract(solutiondata.data_magnitude(f'S({i},in)'), solutiondata.data_magnitude(f'S({i-2},in)'))
            meanS.append(statistics.fmean(sdiff))

        # 添加数据到数据集
        dataset.add_data(meanS, updated_values)

        # 保存更新后的数据集
        torch.save(dataset, filename)
        hfss.close_project()
print('Done!')