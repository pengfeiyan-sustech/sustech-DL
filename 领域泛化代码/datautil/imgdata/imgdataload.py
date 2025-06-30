# coding=utf-8
from torch.utils.data import Dataset
import numpy as np
from datautil.util import Nmax
from datautil.imgdata.util import rgb_loader, l_loader
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader

import os
import torch
from torch.utils.data import Dataset


class EMGDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        自定义 EMG 数据集加载器
        :param root_dir: 数据根目录，例如 'data/s1'。
        :param transform: 数据变换函数，可用于数据增强或归一化。
        """
        self.root_dir = root_dir
        self.transform = transform
        self.data_paths = []
        self.labels = []

        # 遍历所有手势动作子文件夹
        for label, gesture_dir in enumerate(sorted(os.listdir(root_dir))):
            if label < 6:
                gesture_path = os.path.join(root_dir, gesture_dir)
                if os.path.isdir(gesture_path):
                    for sample_file in os.listdir(gesture_path):
                        if sample_file.endswith('.pth'):
                            self.data_paths.append(os.path.join(gesture_path, sample_file))
                            self.labels.append(label)  # 手势类别的标签

    def __len__(self):
        """返回数据集的样本数量"""
        return len(self.data_paths)

    def __getitem__(self, idx):
        """
        根据索引返回一个样本
        :param idx: 样本索引
        :return: 样本数据和标签
        """
        # 加载 .pth 文件中的数据
        data_path = self.data_paths[idx]
        data = torch.load(data_path, weights_only=True)  # 数据形状为 (200, 8)
        data = data.mT.reshape(8, 1, 200)
        data = data.to(torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.int64)

        # 应用数据变换（如果指定了 transform）
        if self.transform:
            data = self.transform(data)

        return data, label



class ImageDataset(object):
    def __init__(self, dataset, task, root_dir, domain_name, domain_label=-1, labels=None, transform=None, target_transform=None, indices=None, test_envs=[], mode='Default'):
        if dataset == "sEMG":
            self.imgs = EMGDataset(root_dir+domain_name)
        else:
            self.imgs = ImageFolder(root_dir+domain_name).imgs
        self.domain_num = 0
        self.task = task
        self.dataset = dataset
        imgs = [item[0] for item in self.imgs]
        labels = [item[1] for item in self.imgs]
        self.labels = np.array(labels)
        self.x = imgs
        self.transform = transform
        self.target_transform = target_transform
        if indices is None:
            self.indices = np.arange(len(imgs))
        else:
            self.indices = indices
        if mode == 'Default':
            self.loader = default_loader
        elif mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader
        self.dlabels = np.ones(self.labels.shape) * \
            (domain_label-Nmax(test_envs, domain_label))

    def set_labels(self, tlabels=None, label_type='domain_label'):
        assert len(tlabels) == len(self.x)
        if label_type == 'domain_label':
            self.dlabels = tlabels
        elif label_type == 'class_label':
            self.labels = tlabels

    def target_trans(self, y):
        if self.target_transform is not None:
            return self.target_transform(y)
        else:
            return y

    def input_trans(self, x):
        if self.transform is not None:
            return self.transform(x)
        else:
            return x

    def __getitem__(self, index):
        index = self.indices[index]
        if self.dataset == "sEMG":
            img = self.x[index]
            ctarget = self.labels[index]
            dtarget = self.dlabels[index]
        else:
            img = self.input_trans(self.loader(self.x[index]))
            ctarget = self.target_trans(self.labels[index])
            dtarget = self.target_trans(self.dlabels[index])
        return img, ctarget, dtarget

    def __len__(self):
        return len(self.indices)
