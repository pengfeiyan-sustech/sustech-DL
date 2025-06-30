# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

from alg.modelopera import get_fea
from network import common_network
from alg.algs.base import Algorithm


class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, args):
        super(ERM, self).__init__(args)
        self.args = args
        self.featurizer = get_fea(args)

        self.classifier = common_network.feat_classifier(
            args.num_classes, self.featurizer.in_features, args.classifier)
        self.network = nn.Sequential(
            self.featurizer, self.classifier)

    def update(self, minibatches, opt, sch):
        all_x = torch.cat([data[0].cuda().float() for data in minibatches])
        all_y = torch.cat([data[1].cuda().long() for data in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y, reduction='mean')

        opt.zero_grad()
        loss.backward()
        opt.step()
        if sch:
            sch.step()
        return {'class': loss.item()}

    def predict(self, x):
        # all_z = torch.angle(torch.fft.fftn(x, dim=(2, 3)))
        return self.network(x)