import numpy as np
from torch.utils.data import Dataset


class SensorDataset(Dataset):
    def __init__(self, dataset):
        # 导入数据，统一使用.npz格式
        self.x = dataset[0]
        self.y = dataset[1]
        self.x = np.transpose(np.expand_dims(self.x, 2), (0, 3, 2, 1)).astype(np.float32)

    def __getitem__(self, index):
        input = self.x[index]
        label = self.y[index]
        return input, label

    def __len__(self):
        return len(self.y)


