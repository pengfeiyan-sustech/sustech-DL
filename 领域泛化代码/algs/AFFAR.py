import torch
import torch.nn as nn
import torch.nn.functional as F
from alg.MMD_Loss import MMD_loss
from alg.modelopera import get_fea
from network import Adver_network, common_network
from alg.algs.base import Algorithm
import torch.optim as optim


class AFFAR(torch.nn.Module):

    def __init__(self, args):
        super(AFFAR, self).__init__()
        self.args = args
        self.cla_num = self.args.domain_num - len(self.args.test_envs)  # 分类器数量
        self.featurizer = get_fea(args)
        self.bottleneck = common_network.feat_bottleneck(
            self.featurizer.in_features, args.bottleneck, args.layer)
        self.classifier = nn.ModuleList()
        for _ in range(self.cla_num):
            classifier = common_network.feat_classifier(
                args.num_classes, args.bottleneck, args.classifier)
            self.classifier.append(classifier)  # 将每个模块添加到模块列表中

        self.discriminator = common_network.feat_classifier(
            self.cla_num, args.bottleneck, args.classifier)

        self.tfbd = args.bottleneck

        self.teaf = get_fea(args)
        self.teab = common_network.feat_bottleneck(
            self.featurizer.in_features, self.tfbd, args.layer)
        self.teac = common_network.feat_classifier(
            args.num_classes, self.tfbd, args.classifier)
        self.teaNet = nn.Sequential(
            self.teaf,
            self.teab,
            self.teac
        )

    def teanettrain(self, dataloaders, epochs, opt1, sch1):
        self.teaNet.train()
        minibatches_iterator = zip(*dataloaders)
        for epoch in range(epochs):
            minibatches = [(tdata) for tdata in next(minibatches_iterator)]
            all_x = torch.cat([data[0].cuda().float() for data in minibatches])
            all_z = torch.angle(torch.fft.fftn(all_x, dim=(2, 3)))
            all_y = torch.cat([data[1].cuda().long() for data in minibatches])
            all_p = self.teaNet(all_z)
            loss = F.cross_entropy(all_p, all_y, reduction='mean')
            opt1.zero_grad()
            loss.backward()
            if ((epoch + 1) % (int(self.args.steps_per_epoch * self.args.max_epoch * 0.7)) == 0 or (epoch + 1) % (
                    int(self.args.steps_per_epoch * self.args.max_epoch * 0.9)) == 0) and (not self.args.schuse):
                for param_group in opt1.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.1
            opt1.step()
            if sch1:
                sch1.step()

            if epoch % int(self.args.steps_per_epoch) == 0 or epoch == epochs - 1:
                print('epoch: %d, cls loss: %.4f' % (epoch, loss))
        self.teaNet.eval()

    def coral(self, x, y):
        mean_x = x.mean(0, keepdim=True)
        mean_y = y.mean(0, keepdim=True)
        cent_x = x - mean_x
        cent_y = y - mean_y
        cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
        cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

        mean_diff = (mean_x - mean_y).pow(2).mean()
        cova_diff = (cova_x - cova_y).pow(2).mean()

        return mean_diff + cova_diff

    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1),
                          x1,
                          x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100,
                                           1000]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):
        Kxx = self.gaussian_kernel(x, x).mean()
        Kyy = self.gaussian_kernel(y, y).mean()
        Kxy = self.gaussian_kernel(x, y).mean()
        return Kxx + Kyy - 2 * Kxy

    def update(self, minibatches, opt, sch):
        all_x = torch.cat([data[0].cuda().float() for data in minibatches])
        all_y = torch.cat([data[1].cuda().long() for data in minibatches])
        with torch.no_grad():
            all_x1 = torch.angle(torch.fft.fftn(all_x, dim=(2, 3)))
            tfea = self.teab(self.teaf(all_x1)).detach()

        all_z = self.bottleneck(self.featurizer(all_x))
        loss1 = F.mse_loss(all_z[:, :self.tfbd], tfea) * self.args.alpha

        # 每个领域数据分别计算高级特征
        outputs = []
        for i, classifier in enumerate(self.classifier):
            output = classifier(all_z[i * self.args.batch_size:(i + 1) * self.args.batch_size])
            outputs.append(output)
        # 每两个领域分别计算MMD损失
        loss2 = 0
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                loss2 += self.coral(outputs[i], outputs[j])
                # loss2 += self.mmd(outputs[i], outputs[j])
        loss2 = loss2 * 2 / (len(minibatches) *
                             (len(minibatches) - 1)) * self.args.lam
        # 计算领域判别损失
        disc_input = all_z
        # disc_input = Adver_network.ReverseLayerF.apply(
        #     disc_input, self.args.alpha)
        domain_weights = self.discriminator(disc_input)
        disc_labels = torch.cat([
            torch.full((data[0].shape[0],), i,
                       dtype=torch.int64, device='cuda')
            for i, data in enumerate(minibatches)
        ])
        loss3 = F.cross_entropy(domain_weights, disc_labels) * self.args.beta
        # 计算分类损失
        final_output = torch.cat([output for output in outputs])
        loss4 = F.cross_entropy(final_output, all_y)
        # 总的损失
        loss = loss1 + loss2 + loss3 + loss4
        # loss = loss4
        opt.zero_grad()
        loss.backward()
        opt.step()
        if sch:
            sch.step()
        return {'total': loss.item(), 'class': loss4.item(), 'dis': loss3.item(), 'mmd': loss2}

    def predict(self, x):
        feature = self.bottleneck(self.featurizer(x))
        outputs = []
        for classifier in self.classifier:
            output = classifier(feature)
            outputs.append(output)

        domain_weights = self.discriminator(feature)
        normalized_weights = torch.softmax(domain_weights, dim=1)

        weighted_outputs = []
        for i in range(self.args.domain_num - len(self.args.test_envs)):
            weighted_output = outputs[i] * (normalized_weights[:, i].unsqueeze(1))  # 将分类器的输出与对应的权重相乘
            weighted_outputs.append(weighted_output)

        final_output = sum(weighted_outputs)  # 将所有加权输出相加得到最终的预测输出
        # final_output = sum(outputs)  # 将所有加权输出相加得到最终的预测输出

        return final_output
