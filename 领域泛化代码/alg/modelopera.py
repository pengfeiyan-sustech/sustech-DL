# coding=utf-8
import torch
from network import img_network
import torch
import torch.nn as nn
from torchsummary import summary

# 定义CNN模型
class sEMGModel(nn.Module):
    def __init__(self):
        super(sEMGModel, self).__init__()
        self.model = nn.Sequential(
            # 第一个卷积层: 输入通道=8, 输出通道=16, 核大小=3, 步长=1, 填充=1
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),  # 批量归一化
            nn.ReLU(),  # ReLU激活函数

            # 第二个卷积层: 输入通道=16, 输出通道=32, 核大小=3, 步长=1, 填充=1
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),  # 批量归一化
            nn.ReLU(),  # ReLU激活函数

            # 第三个卷积层: 输入通道=32, 输出通道=64, 核大小=3, 步长=1, 填充=1
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),  # 批量归一化
            nn.ReLU(),  # ReLU激活函数

            # 展平层，将多维输入一维化
            nn.Flatten(),

            # 全连接层: 输入=64*200, 输出=10 (比如用于分类)
            # nn.Linear(64 * 200, 16)
        )
        self.in_features = 64 * 200
    def forward(self, x):
        return self.model(x)

def get_fea(args):
    if args.dataset == 'dg5':
        net = img_network.DTNBase()
    elif args.dataset == 'sEMG':
        net = sEMGModel()
    elif args.net.startswith('res'):
        net = img_network.ResBase(args.net)
    else:
        net = img_network.VGGBase(args.net)
    return net



def accuracy(network, loader):
    correct = 0
    total = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network.eval()
    with torch.no_grad():
        for data in loader:
            x = data[0].to(device).float()
            y = data[1].to(device).long()
            p = network.predict(x)

            if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float()).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float()).sum().item()
            total += len(x)
    network.train()
    return correct / total






