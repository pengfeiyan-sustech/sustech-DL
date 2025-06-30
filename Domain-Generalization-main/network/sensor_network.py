import torch
import torch.nn as nn
from torchsummary import summary


class Net(nn.Module):
    def __init__(self, dataset='DSADS'):
        super(Net, self).__init__()
        self.dataset = dataset
        self.var_size = {
            'DSADS': {
                'in_size': 45,
                'ker_size': 9,
                'fc_size': 32 * 25
            },
            'uschad': {
                'in_size': 6,
                'ker_size': 9,
                'fc_size': 32 * 119
            },
            'pamap': {
                'in_size': 27,
                'ker_size': 9,
                'fc_size': 32 * 122
            },
            'EMG': {
                'in_size': 8,
                'ker_size': 9,
                'fc_size': 32 * 58
            },
            'DB2': {
                'in_size': 12,
                'ker_size': 9,
                'fc_size': 32 * 124
            },
            'oymotion': {
                'in_size': 8,
                'ker_size': 9,
                'fc_size': 32 * 58
            },
        }

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.var_size[self.dataset]['in_size'], out_channels=16, kernel_size=(
                1, self.var_size[self.dataset]['ker_size'])),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(
                1, self.var_size[self.dataset]['ker_size'])),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        )

        self.in_features = self.var_size[self.dataset]['fc_size']

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        return x


if __name__ == '__main__':
    net = Net(dataset='pamap').cuda()
    print(summary(net, (27, 1, 512)))