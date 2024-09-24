import os
import re
import torch
from torch.utils.data import Dataset
import ansys.aedt.core

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

hfss = ansys.aedt.core.Hfss(project="C:\\Users\\bacto\\Documents\\Ansoft\\branchlinecoupler-1c.aedt",
                            design="coupler-1a5",
                            version='242',
                            non_graphical=True)

load_data_from_hfss(hfss, CustomDataset(data_file='F:\\pythontxtfile\\coupler-1c', load_from_file=True))