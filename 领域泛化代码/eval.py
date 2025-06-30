# coding=utf-8

import os
import sys
import time
import numpy as np
import argparse

from alg.opt import *
from alg import alg, modelopera
from utils.util import set_random_seed, save_checkpoint, print_args, train_valid_target_eval_names, alg_loss_dict, Tee, img_param_init, print_environ
from datautil.getdataloader import get_img_dataloader
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import seaborn as sns

def get_args():
    parser = argparse.ArgumentParser(description='DG')
    parser.add_argument('--algorithm', type=str, default="DANN")
    parser.add_argument('--alpha', type=float,
                        default=1, help='DANN dis alpha')
    parser.add_argument('--anneal_iters', type=int,
                        default=500, help='Penalty anneal iters used in VREx')
    parser.add_argument('--batch_size', type=int,
                        default=32, help='batch_size')
    parser.add_argument('--beta', type=float,
                        default=1, help='DIFEX beta')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='Adam hyper-param')
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--checkpoint_freq', type=int,
                        default=1, help='Checkpoint every N epoch')
    parser.add_argument('--classifier', type=str,
                        default="linear", choices=["linear", "wn"])
    parser.add_argument('--data_file', type=str, default='',
                        help='root_dir')
    # 改动数据集，需改动下面两行
    parser.add_argument('--dataset', type=str, default='sEMG')
    parser.add_argument('--data_dir', type=str, default='dataset/sEMG/', help='数据集路径')
    parser.add_argument('--dis_hidden', type=int,
                        default=256, help='dis hidden dimension')
    parser.add_argument('--disttype', type=str, default='2-norm',
                        choices=['1-norm', '2-norm', 'cos', 'norm-2-norm', 'norm-1-norm'])
    parser.add_argument('--gpu_id', type=str, nargs='?',
                        default='0', help="device id to run")
    parser.add_argument('--groupdro_eta', type=float,
                        default=1, help="groupdro eta")
    parser.add_argument('--inner_lr', type=float,
                        default=1e-2, help="learning rate used in MLDG")
    parser.add_argument('--lam', type=float,
                        default=1, help="tradeoff hyperparameter used in VREx")
    parser.add_argument('--layer', type=str, default="bn",
                        choices=["ori", "bn"])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--lr_decay', type=float, default=0.75, help='for sgd')
    parser.add_argument('--lr_decay1', type=float,
                        default=1.0, help='for pretrained featurizer')
    parser.add_argument('--lr_decay2', type=float, default=1.0,
                        help='inital learning rate decay of network')
    parser.add_argument('--lr_gamma', type=float,
                        default=0.0003, help='for optimizer')
    parser.add_argument('--max_epoch', type=int,
                        default=10, help="max iterations")
    parser.add_argument('--mixupalpha', type=float,
                        default=0.2, help='mixup hyper-param')
    parser.add_argument('--mldg_beta', type=float,
                        default=1, help="mldg hyper-param")
    parser.add_argument('--mmd_gamma', type=float,
                        default=1, help='MMD, CORAL hyper-param')
    parser.add_argument('--momentum', type=float,
                        default=0.9, help='for optimizer')
    parser.add_argument('--net', type=str, default='resnet50',
                        help="featurizer: vgg16, resnet50, resnet101,DTNBase")
    parser.add_argument('--N_WORKERS', type=int, default=4)
    parser.add_argument('--rsc_f_drop_factor', type=float,
                        default=1/3, help='rsc hyper-param')
    parser.add_argument('--rsc_b_drop_factor', type=float,
                        default=1/3, help='rsc hyper-param')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    parser.add_argument('--schuse', action='store_true')
    parser.add_argument('--schusech', type=str, default='cos')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--split_style', type=str, default='strat',
                        help="the style to split the train and eval datasets")
    parser.add_argument('--task', type=str, default="img_dg",
                        choices=["img_dg"], help='now only support image tasks')
    parser.add_argument('--tau', type=float, default=1, help="andmask tau")
    parser.add_argument('--test_envs', type=int, nargs='+',
                        default=[0], help='target domains')
    parser.add_argument('--output', type=str,
                        default="train_output", help='result output path')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    args = parser.parse_args()
    args.steps_per_epoch = 100
    args.data_dir = args.data_file+args.data_dir
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    os.makedirs(args.output, exist_ok=True)
    sys.stdout = Tee(os.path.join(args.output, 'out.txt'))
    sys.stderr = Tee(os.path.join(args.output, 'err.txt'))
    args = img_param_init(args)
    print_environ()
    return args


def load_checkpoint(model, checkpoint_path, args=None):
    """
    加载模型权重
    :param model: 定义好的模型结构
    :param checkpoint_path: 保存权重的文件路径
    :param args: 可选的配置参数
    :return: 加载权重后的模型
    """
    # 加载保存的权重文件
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    # 应用权重到模型
    model.load_state_dict(checkpoint['model_dict'])

    if args is not None:
        # 可选：更新模型的配置参数
        args.__dict__.update(checkpoint['args'])

    return model


def test_model(model, test_loader, device):
    model.eval()  # 设置模型为评估模式

    all_labels = []
    all_preds = []
    all_probs = []
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():  # 在测试过程中不计算梯度
        for data in test_loader:
            inputs = data[0].to(device).float()
            labels = data[1].to(device).long()

            outputs = model.predict(inputs)  # 获取模型的输出
            _, preds = torch.max(outputs, 1)  # 获取预测类别
            probs = torch.softmax(outputs, dim=1)  # 计算每个类别的概率

            # 累加正确预测的数量
            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # 计算并打印准确率
    accuracy = correct_predictions / total_samples * 100
    print(f"Test Accuracy: {accuracy:.2f}%")

    # 计算并绘制混淆矩阵
    plot_confusion_matrix(all_labels, all_preds)

    # 计算并绘制混淆矩阵(百分比形式)
    plot_confusion_matrix_v1(all_labels, all_preds)

    # 计算并绘制ROC曲线
    plot_roc_curve(all_labels, all_probs)


def plot_confusion_matrix(true_labels, predicted_labels):
    # 计算混淆矩阵
    cm = confusion_matrix(true_labels, predicted_labels)

    # 使用 seaborn 绘制热力图
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=True, yticklabels=True)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()


def plot_confusion_matrix_v1(true_labels, predicted_labels):
    # 计算混淆矩阵
    cm = confusion_matrix(true_labels, predicted_labels)

    # 归一化为百分比
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # 使用 seaborn 绘制热力图
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_percentage, annot=True, fmt='.2f', cmap='Blues', xticklabels=True, yticklabels=True)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix (Percentage)')
    plt.show()


def plot_roc_curve(true_labels, predicted_probs):
    # 将标签二值化
    n_classes = len(set(true_labels))
    true_labels_bin = label_binarize(true_labels, classes=range(n_classes))

    # 计算 ROC 曲线和 AUC
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(true_labels_bin[:, i], [p[i] for p in predicted_probs])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 绘制每个类别的 ROC 曲线
    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

    # 绘制对角线（随机分类器的表现）
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = get_args()
    set_random_seed(args.seed)

    loss_list = alg_loss_dict(args)
    train_loaders, eval_loaders = get_img_dataloader(args)
    eval_name_dict = train_valid_target_eval_names(args)
    algorithm_class = alg.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(args).to(device)
    algorithm.eval()

    # 指定保存权重的文件路径
    checkpoint_path = 'train_output/DANN_[0]_model.pkl'  # 替换为您的权重文件路径

    # 加载权重
    algorithm = load_checkpoint(algorithm, checkpoint_path)

    opt = get_optimizer(algorithm, args)
    sch = get_scheduler(opt, args)

    s = print_args(args, [])
    print('=======hyper-parameter used========')
    print(s)

    if 'DIFEX' in args.algorithm:
        ms = time.time()
        n_steps = args.max_epoch*args.steps_per_epoch
        print('start training fft teacher net')
        opt1 = get_optimizer(algorithm.teaNet, args, isteacher=True)
        sch1 = get_scheduler(opt1, args)
        algorithm.teanettrain(train_loaders, n_steps, opt1, sch1)
        print('complet time:%.4f' % (time.time()-ms))

    acc_record = {}
    acc_type_list = ['train', 'valid', 'target']
    train_minibatches_iterator = zip(*train_loaders)
    best_valid_acc, target_acc = 0, 0
    print('===========start eval===========')
    # s = ''
    # for item in acc_type_list:
    #     acc_record[item] = np.mean(np.array([modelopera.accuracy(
    #         algorithm, eval_loaders[i]) for i in eval_name_dict[item]]))
    #     s += (item + '_acc:%.4f,' % acc_record[item])
    #
    # print(s)
    target_loader = eval_loaders[eval_name_dict['target'][0]]
    test_model(algorithm, target_loader, device)

