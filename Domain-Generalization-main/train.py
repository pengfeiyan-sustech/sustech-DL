# coding=utf-8

import os
import sys
import time
import numpy as np
import argparse
import torch
# torch.backends.cudnn.enable = False

from alg.opt import *
from alg import alg, modelopera
from utils.util import set_random_seed, save_checkpoint, print_args, train_valid_target_eval_names, alg_loss_dict, Tee, img_param_init, print_environ, sensor_param_init
from datautil.getdataloader import get_img_dataloader, get_sensor_dataloader


def get_args():
    parser = argparse.ArgumentParser(description='DG')
    parser.add_argument('--algorithm', type=str, default="AFFAR")
    parser.add_argument('--alpha', type=float,
                        default=1, help='DANN dis alpha')
    parser.add_argument('--anneal_iters', type=int,
                        default=500, help='Penalty anneal iters used in VREx')
    parser.add_argument('--batch_size', type=int,
                        default=64, help='batch_size')
    parser.add_argument('--beta', type=float,
                        default=1, help='DIFEX beta')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='Adam hyper-param')
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--checkpoint_freq', type=int,
                        default=3, help='Checkpoint every N epoch')
    parser.add_argument('--classifier', type=str,
                        default="linear", choices=["linear", "wn"])
    parser.add_argument('--data_file', type=str, default='',
                        help='root_dir')
    parser.add_argument('--dataset', type=str, default='EMG')
    parser.add_argument('--data_dir', type=str, default='./datasets/', help='data dir')
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
    parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
    # parser.add_argument('--lr', type=float, default=1e-6, help="learning rate")
    parser.add_argument('--lr_decay', type=float, default=0.75, help='for sgd')
    parser.add_argument('--lr_decay1', type=float,
                        default=1.0, help='for pretrained featurizer')
    parser.add_argument('--lr_decay2', type=float, default=1.0,
                        help='inital learning rate decay of network')
    parser.add_argument('--lr_gamma', type=float,
                        default=0.0003, help='for optimizer')
    parser.add_argument('--max_epoch', type=int,
                        default=100, help="max iterations")
    parser.add_argument('--mixupalpha', type=float,
                        default=0.2, help='mixup hyper-param')
    parser.add_argument('--mldg_beta', type=float,
                        default=1, help="mldg hyper-param")
    parser.add_argument('--mmd_gamma', type=float,
                        default=1, help='MMD, CORAL hyper-param')
    parser.add_argument('--momentum', type=float,
                        default=0.9, help='for optimizer')
    parser.add_argument('--net', type=str, default='resnet18',
                        help="featurizer: vgg16, resnet50, resnet101,DTNBase")
    parser.add_argument('--N_WORKERS', type=int, default=4)
    parser.add_argument('--rsc_f_drop_factor', type=float,
                        default=1/3, help='rsc hyper-param')
    parser.add_argument('--rsc_b_drop_factor', type=float,
                        default=1/3, help='rsc hyper-param')
    parser.add_argument('--save_model_every_checkpoint', type=str,
                        default='True')
    parser.add_argument('--schuse', action='store_true')
    parser.add_argument('--schusech', type=str, default='cos')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--split_style', type=str, default='strat',
                        help="the style to split the train and eval datasets")
    parser.add_argument('--task', type=str, default="sensor_dg",
                        choices=["img_dg", "sensor_dg"], help='not only support image tasks')
    parser.add_argument('--tau', type=float, default=1, help="andmask tau")
    parser.add_argument('--test_envs', type=int, nargs='+',
                        default=[3], help='target domains')
    parser.add_argument('--output', type=str,
                        default="train_output", help='result output path')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    args = parser.parse_args()
    args.steps_per_epoch = 1000
    args.data_dir = args.data_dir+args.dataset+'/'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    os.makedirs(args.output, exist_ok=True)
    sys.stdout = Tee(os.path.join(args.output, 'out.txt'))
    sys.stderr = Tee(os.path.join(args.output, 'err.txt'))
    args = img_param_init(args) if args.task == "img_dg" else sensor_param_init(args)
    print_environ()
    return args


if __name__ == '__main__':
    args = get_args()
    set_random_seed(args.seed)

    loss_list = alg_loss_dict(args)
    train_loaders, eval_loaders = get_img_dataloader(args) if args.task == 'img_dg' else get_sensor_dataloader(args)
    eval_name_dict = train_valid_target_eval_names(args)    # train+test
    algorithm_class = alg.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(args).cuda()
    algorithm.train()
    print(algorithm)
    opt = get_optimizer(algorithm, args)
    sch = get_scheduler(opt, args)

    s = print_args(args, [])
    print('=======hyper-parameter used========')
    print(s)

    # if 'DIFEX' in args.algorithm:
    if args.algorithm in ['DIFEX', 'AFFAR']:
        ms = time.time()
        n_steps = args.max_epoch*args.steps_per_epoch
        # n_steps = 1000
        print('start training fft teacher net')
        opt1 = get_optimizer(algorithm.teaNet, args, isteacher=True)
        sch1 = get_scheduler(opt1, args)
        algorithm.teanettrain(train_loaders, n_steps, opt1, sch1)
        print('complet time:%.4f' % (time.time()-ms))

    acc_record = {}
    acc_type_list = ['train', 'valid', 'target']
    train_minibatches_iterator = zip(*train_loaders)
    best_valid_acc, target_acc = 0, 0
    print('===========start training===========')
    sss = time.time()
    for epoch in range(args.max_epoch):
        for iter_num in range(args.steps_per_epoch):
            minibatches_device = [(data)
                                  for data in next(train_minibatches_iterator)]
            if args.algorithm == 'VREx' and algorithm.update_count == args.anneal_iters:
                opt = get_optimizer(algorithm, args)
                sch = get_scheduler(opt, args)
            step_vals = algorithm.update(minibatches_device, opt, sch)

        if (epoch in [int(args.max_epoch*0.4), int(args.max_epoch*0.7), int(args.max_epoch*0.9)]) and (not args.schuse):
            print('manually descrease lr')
            for params in opt.param_groups:
                params['lr'] = params['lr']*0.1

        if (epoch == (args.max_epoch-1)) or (epoch % args.checkpoint_freq == 0):
            print('===========epoch %d===========' % (epoch))
            s = ''
            for item in loss_list:
                s += (item+'_loss:%.4f,' % step_vals[item])
            print(s[:-1])
            s = ''
            for item in acc_type_list:
                acc_record[item] = np.mean(np.array([modelopera.accuracy(
                    algorithm, eval_loaders[i]) for i in eval_name_dict[item]]))
                s += (item+'_acc:%.4f,' % acc_record[item])
            print(s[:-1])
            if acc_record['valid'] > best_valid_acc:
                best_valid_acc = acc_record['valid']
                target_acc = acc_record['target']
                save_checkpoint(f'EMG_{args.algorithm}_T3.pkl', algorithm, args)
                print("模型已保存")
            # if args.save_model_every_checkpoint == 'True':
            #     save_checkpoint(f'model_epoch{epoch}.pkl', algorithm, args)
            print('total cost time: %.4f' % (time.time()-sss))
            algorithm_dict = algorithm.state_dict()

    # save_checkpoint(f'EMG_{args.algorithm}_T0.pkl', algorithm, args)

    print('valid acc: %.4f' % best_valid_acc)
    print('DG result: %.4f' % target_acc)

    with open(os.path.join(args.output, 'done.txt'), 'w') as f:
        f.write('done\n')
        f.write('total cost time:%s\n' % (str(time.time()-sss)))
        f.write('valid acc:%.4f\n' % (best_valid_acc))
        f.write('target acc:%.4f' % (target_acc))
