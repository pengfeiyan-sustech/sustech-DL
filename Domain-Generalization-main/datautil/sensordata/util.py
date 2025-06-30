import numpy as np
from sklearn.model_selection import train_test_split


def merge_split_dsads(args):
    d = np.load(args.data_dir + f'{args.dataset}_processwin.npz')
    x, y, s = d['x'], (d['y'] - 1).reshape(-1, ), (d['s'] - 1).reshape(-1, )
    data_lst = []
    for i in range(args.domain_num):
        data_i = []
        d_index = np.argwhere((s == 2 * i) | (s == 2 * i + 1)).reshape(-1, )
        x_i = x[d_index, :, :]
        y_i = y[d_index]
        data_i.append(x_i)
        data_i.append(y_i)
        data_lst.append(data_i)

    return devide_train_val(data_lst, args.domain_num, args.seed)


def merge_split_oymotion(args):
    d = np.load(args.data_dir + f'{args.dataset}_processwin.npz')
    x, y, s = d['x'], (d['y'] - 1).reshape(-1, ), (d['s'] - 1).reshape(-1, )
    data_lst = []
    for i in range(args.domain_num):
        data_i = []
        d_index = np.argwhere((s == i)).reshape(-1, )
        x_i = x[d_index, :, :]
        y_i = y[d_index]
        data_i.append(x_i)
        data_i.append(y_i)
        data_lst.append(data_i)

    return devide_train_val(data_lst, args.domain_num, args.seed)


def merge_split_emg(args):
    d = np.load(args.data_dir + f'{args.dataset}_processwin.npz')
    x, y, s = d['x'], (d['y'] - 1).reshape(-1, ), (d['s'] - 1).reshape(-1, )
    data_lst = []
    data_0, data_1, data_2, data_3, data_4 = [], [], [], [], []

    d_index_0 = np.argwhere(np.isin(s, np.arange(0, 36, 4))).reshape(-1, )
    x_0 = x[d_index_0]
    y_0 = y[d_index_0]
    data_0.append(x_0)
    data_0.append(y_0)
    data_lst.append(data_0)

    d_index_1 = np.argwhere(np.isin(s, np.arange(1, 36, 4))).reshape(-1, )
    x_1 = x[d_index_1]
    y_1 = y[d_index_1]
    data_1.append(x_1)
    data_1.append(y_1)
    data_lst.append(data_1)

    d_index_2 = np.argwhere(np.isin(s, np.arange(2, 36, 4))).reshape(-1, )
    x_2 = x[d_index_2]
    y_2 = y[d_index_2]
    data_2.append(x_2)
    data_2.append(y_2)
    data_lst.append(data_2)

    d_index_3 = np.argwhere(np.isin(s, np.arange(3, 36, 4))).reshape(-1, )
    x_3 = x[d_index_3]
    y_3 = y[d_index_3]
    data_3.append(x_3)
    data_3.append(y_3)
    data_lst.append(data_3)

    return devide_train_val(data_lst, args.domain_num, args.seed)


def merge_split_uschad(args):
    d = np.load(args.data_dir + f'{args.dataset}_processwin.npz')
    x, y, s = d['x'], (d['y'] - 1).reshape(-1, ), (d['s'] - 1).reshape(-1, )
    data_lst = []
    data_0, data_1, data_2, data_3, data_4 = [], [], [], [], []

    d_index_0 = np.argwhere(np.isin(s, np.arange(0, 12, 4))).reshape(-1, )
    x_0 = x[d_index_0]
    y_0 = y[d_index_0]
    data_0.append(x_0)
    data_0.append(y_0)
    data_lst.append(data_0)

    d_index_1 = np.argwhere(np.isin(s, np.arange(1, 12, 4))).reshape(-1, )
    x_1 = x[d_index_1]
    y_1 = y[d_index_1]
    data_1.append(x_1)
    data_1.append(y_1)
    data_lst.append(data_1)

    d_index_2 = np.argwhere(np.isin(s, np.arange(2, 12, 4))).reshape(-1, )
    x_2 = x[d_index_2]
    y_2 = y[d_index_2]
    data_2.append(x_2)
    data_2.append(y_2)
    data_lst.append(data_2)

    d_index_3 = np.argwhere(np.isin(s, np.arange(3, 12, 4))).reshape(-1, )
    x_3 = x[d_index_3]
    y_3 = y[d_index_3]
    data_3.append(x_3)
    data_3.append(y_3)
    data_lst.append(data_3)

    return devide_train_val(data_lst, args.domain_num, args.seed)


def merge_split_pamap(args):
    d = np.load(args.data_dir + f'{args.dataset}_processwin.npz')
    x, y, s = d['x'], d['y'].reshape(-1, ), (d['s'] - 1).reshape(-1, )
    data_lst = []
    data_0, data_1, data_2, data_3, data_4 = [], [], [], [], []

    d_index_0 = np.argwhere(np.isin(s, np.arange(0, 8, 4))).reshape(-1, )
    x_0 = x[d_index_0]
    y_0 = y[d_index_0]
    data_0.append(x_0)
    data_0.append(y_0)
    data_lst.append(data_0)

    d_index_1 = np.argwhere(np.isin(s, np.arange(1, 8, 4))).reshape(-1, )
    x_1 = x[d_index_1]
    y_1 = y[d_index_1]
    data_1.append(x_1)
    data_1.append(y_1)
    data_lst.append(data_1)

    d_index_2 = np.argwhere(np.isin(s, np.arange(2, 8, 4))).reshape(-1, )
    x_2 = x[d_index_2]
    y_2 = y[d_index_2]
    data_2.append(x_2)
    data_2.append(y_2)
    data_lst.append(data_2)

    d_index_3 = np.argwhere(np.isin(s, np.arange(3, 8, 4))).reshape(-1, )
    x_3 = x[d_index_3]
    y_3 = y[d_index_3]
    data_3.append(x_3)
    data_3.append(y_3)
    data_lst.append(data_3)

    return devide_train_val(data_lst, args.domain_num, args.seed)


def merge_split_DB2(args):
    d = np.load(args.data_dir + f'{args.dataset}_processwin.npz')
    x, y, s = d['x'], (d['y'] - 1).reshape(-1, ), (d['s'] - 1).reshape(-1, )
    data_lst = []
    data_0, data_1, data_2, data_3, data_4 = [], [], [], [], []

    d_index_0 = np.argwhere(np.isin(s, np.arange(0, 8, 4))).reshape(-1, )
    x_0 = x[d_index_0]
    y_0 = y[d_index_0]
    data_0.append(x_0)
    data_0.append(y_0)
    data_lst.append(data_0)

    d_index_1 = np.argwhere(np.isin(s, np.arange(1, 8, 4))).reshape(-1, )
    x_1 = x[d_index_1]
    y_1 = y[d_index_1]
    data_1.append(x_1)
    data_1.append(y_1)
    data_lst.append(data_1)

    d_index_2 = np.argwhere(np.isin(s, np.arange(2, 8, 4))).reshape(-1, )
    x_2 = x[d_index_2]
    y_2 = y[d_index_2]
    data_2.append(x_2)
    data_2.append(y_2)
    data_lst.append(data_2)

    d_index_3 = np.argwhere(np.isin(s, np.arange(3, 8, 4))).reshape(-1, )
    x_3 = x[d_index_3]
    y_3 = y[d_index_3]
    data_3.append(x_3)
    data_3.append(y_3)
    data_lst.append(data_3)

    return devide_train_val(data_lst, args.domain_num, args.seed)


def devide_train_val(data_lst, domain_num, seed):
    domain_dic = []
    for i in range(domain_num):
        train_i, val_i, dic_i = [], [], []
        x_i, y_i = data_lst[i][0], data_lst[i][1]
        x_train_i, x_val_i, y_train_i, y_val_i = train_test_split(
            x_i, y_i, test_size=0.2, random_state=seed, stratify=y_i)

        train_i.append(x_train_i)
        train_i.append(y_train_i)

        val_i.append(x_val_i)
        val_i.append(y_val_i)

        dic_i.append(train_i)
        dic_i.append(val_i)
        domain_dic.append(dic_i)

    return domain_dic


def merge_split(args):
    domain_dic = []
    if args.dataset == 'DSADS':
        domain_dic = merge_split_dsads(args)
    elif args.dataset == 'uschad':
        domain_dic = merge_split_uschad(args)
    elif args.dataset == 'pamap':
        domain_dic = merge_split_pamap(args)
    elif args.dataset == 'EMG':
        domain_dic = merge_split_emg(args)
    elif args.dataset == 'DB2':
        domain_dic = merge_split_DB2(args)
    elif args.dataset == 'oymotion':
        domain_dic = merge_split_oymotion(args)
    else:
        ...
    return domain_dic
