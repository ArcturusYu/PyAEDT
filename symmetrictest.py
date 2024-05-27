import torch
from torch.utils.data import Dataset, DataLoader

class SymmetricDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        original = self.data[idx]
        reversed_input = original[::-1]  # Assuming the data is in a reversible format like a list or tensor
        return original, reversed_input

# Example data
data = [
    (torch.tensor([1, 2, 3, 4, 5]), torch.tensor([0, 1, 2, 3])),
    (torch.tensor([5, 4, 3, 2, 1]), torch.tensor([3, 2, 1, 0]))
]

dataset = SymmetricDataset(data)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

for data in dataloader:
    print(data)
