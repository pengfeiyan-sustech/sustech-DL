import torch.nn as nn
from Coral import CORAL
import mmd
import backbone
from lmmd import LMMD_loss
import torch.nn.functional as F
import torch
from JDA import JointMultipleKernelMaximumMeanDiscrepancy
from kernels import GaussianKernel

class Transfer_Net(nn.Module):
    def __init__(self, num_class, base_net='alexnet', transfer_loss='mmd', use_bottleneck=True, bottleneck_width=256, width=1024):
        super(Transfer_Net, self).__init__()
        self.num_class = num_class
        self.base_network = backbone.network_dict[base_net]()
        self.use_bottleneck = use_bottleneck
        self.transfer_loss = transfer_loss
        bottleneck_list = [nn.Linear(self.base_network.output_num(
        ), bottleneck_width), nn.BatchNorm1d(bottleneck_width), nn.ReLU(), nn.Dropout(0.5)]
        self.bottleneck_layer = nn.Sequential(*bottleneck_list)
        classifier_layer_list = [nn.Linear(self.base_network.output_num(), width), nn.ReLU(), nn.Dropout(0.5),
                                 nn.Linear(width, num_class)]
        self.classifier_layer = nn.Sequential(*classifier_layer_list)

        self.bottleneck_layer[0].weight.data.normal_(0, 0.005)
        self.bottleneck_layer[0].bias.data.fill_(0.1)
        for i in range(2):
            self.classifier_layer[i * 3].weight.data.normal_(0, 0.01)
            self.classifier_layer[i * 3].bias.data.fill_(0.0)

    def forward(self, source, target, source_label=None):
        source = self.base_network(source)
        target = self.base_network(target)
        source_clf = self.classifier_layer(source)
        target_label = self.classifier_layer(target)
        if self.use_bottleneck:
            source = self.bottleneck_layer(source)
            target = self.bottleneck_layer(target)
        transfer_loss = self.adapt_loss(source, target, self.transfer_loss, source_clf, target_label)
        return source_clf, transfer_loss

    def predict(self, x):
        features = self.base_network(x)
        clf = self.classifier_layer(features)
        return clf

    def adapt_loss(self, X, Y, adapt_loss, X_label=None, Y_label=None):
        """Compute adaptation loss, currently we support mmd and coral

        Arguments:
            X {tensor} -- source matrix
            Y {tensor} -- target matrix
            adapt_loss {string} -- loss type, 'mmd' or 'coral'. You can add your own loss

        Returns:
            [tensor] -- adaptation loss tensor
        """
        if adapt_loss == 'mmd':
            mmd_loss = mmd.MMD_loss()
            loss = mmd_loss(X, Y)
        elif adapt_loss == 'coral':
            loss = CORAL(X, Y)
        elif adapt_loss == 'DSAN':
            lmmd_loss = LMMD_loss(class_num=self.num_class)# X_label是原有标签
            loss = lmmd_loss.get_loss(X, Y, X_label, torch.nn.functional.softmax(Y_label, dim=1))
        elif adapt_loss == 'JDA':   # X_label是预测标签
            layer1_kernels = (GaussianKernel(alpha=0.5), GaussianKernel(1.), GaussianKernel(2.))
            layer2_kernels = (GaussianKernel(1.),)
            jmmd_loss = JointMultipleKernelMaximumMeanDiscrepancy((layer1_kernels, layer2_kernels))
            loss = jmmd_loss(
            (X, F.softmax(X_label, dim=1)),
            (Y, F.softmax(Y_label, dim=1))
        )
        else:
            loss = 0
        return loss

if __name__ == '__main__':
    Net = Transfer_Net(num_class=6)
    print(Net)