# coding=utf-8
import numpy as np
import sklearn.model_selection as ms
from torch.utils.data import DataLoader

import datautil.imgdata.util as imgutil
from datautil.imgdata.imgdataload import ImageDataset
from datautil.mydataloader import InfiniteDataLoader
from datautil.sensordata.sensordataload import SensorDataset
from datautil.sensordata.util import merge_split

def get_img_dataloader(args):
    rate = 0.2
    trdatalist, tedatalist = [], []

    names = args.img_dataset[args.dataset]
    args.domain_num = len(names)
    for i in range(len(names)):
        if i in args.test_envs:
            tedatalist.append(ImageDataset(args.dataset, args.task, args.data_dir,
                                           names[i], i, transform=imgutil.image_test(args.dataset),
                                           test_envs=args.test_envs))
        else:
            tmpdatay = ImageDataset(args.dataset, args.task, args.data_dir,
                                    names[i], i, transform=imgutil.image_train(args.dataset),
                                    test_envs=args.test_envs).labels
            l = len(tmpdatay)
            if args.split_style == 'strat':
                lslist = np.arange(l)
                stsplit = ms.StratifiedShuffleSplit(
                    2, test_size=rate, train_size=1 - rate, random_state=args.seed)
                stsplit.get_n_splits(lslist, tmpdatay)
                indextr, indexte = next(stsplit.split(lslist, tmpdatay))
            else:
                indexall = np.arange(l)
                np.random.seed(args.seed)
                np.random.shuffle(indexall)
                ted = int(l * rate)
                indextr, indexte = indexall[:-ted], indexall[-ted:]

            trdatalist.append(ImageDataset(args.dataset, args.task, args.data_dir,
                                           names[i], i, transform=imgutil.image_train(args.dataset), indices=indextr,
                                           test_envs=args.test_envs))
            tedatalist.append(ImageDataset(args.dataset, args.task, args.data_dir,
                                           names[i], i, transform=imgutil.image_test(args.dataset), indices=indexte,
                                           test_envs=args.test_envs))

    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=None,
        batch_size=args.batch_size,
        num_workers=args.N_WORKERS)
        for env in trdatalist]

    eval_loaders = [DataLoader(
        dataset=env,
        batch_size=64,
        num_workers=args.N_WORKERS,
        drop_last=False,
        shuffle=False)
        for env in trdatalist + tedatalist]

    return train_loaders, eval_loaders


def get_sensor_dataloader(args):
    names = args.sensor_dataset[args.dataset]
    args.domain_num = len(names)
    # 加载数据文件，被分成了args.domains个领域，每个领域包含[0]训练、[1]验证
    domain_dic = merge_split(args)
    # 用于保存训练数据集和测试数据集
    trdatalist, tedatalist = [], []
    for i in range(args.domain_num):
        if i in args.test_envs:
            tedatalist.append(SensorDataset(dataset=domain_dic[i][0]) + SensorDataset(dataset=domain_dic[i][1]))
        else:
            trdatalist.append(SensorDataset(dataset=domain_dic[i][0]))
            tedatalist.append(SensorDataset(dataset=domain_dic[i][1]))
    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=None,
        batch_size=args.batch_size,
        num_workers=0)
        for env in trdatalist]

    test_loaders = [DataLoader(
        dataset=env,
        batch_size=args.batch_size,
        num_workers=0,
        drop_last=False,
        shuffle=False)
        for env in trdatalist + tedatalist]

    return train_loaders, test_loaders
